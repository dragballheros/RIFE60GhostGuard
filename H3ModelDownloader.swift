import Combine
import CryptoKit
import Foundation

private enum H3ModelDownloadError: LocalizedError {
    case listing(Int)
    case emptyManifest
    case badDigest(String)
    case badSize(String)
    case http(String, Int)
    case rangeIgnored
    case hashMismatch(String)

    var errorDescription: String? {
        switch self {
        case .listing(let status): return "Could not list official H3 model files (HTTP \(status))."
        case .emptyManifest: return "The pinned Ref2VA model manifest was empty."
        case .badDigest(let path): return "No verifiable checksum was supplied for \(path)."
        case .badSize(let path): return "Downloaded size did not match the manifest for \(path)."
        case .http(let path, let status): return "Download failed for \(path) (HTTP \(status))."
        case .rangeIgnored: return "The model server did not honor a resume request."
        case .hashMismatch(let path): return "Checksum verification failed for \(path); the incomplete file was discarded."
        }
    }
}


/// Downloads the root model index and Ref2VA task tree expected by h3_load_dir. Model weights are never bundled in the IPA.
/// The model revision is pinned so the listed sizes and digests cannot drift under us.
@MainActor
final class H3ModelDownloader: ObservableObject {
    enum State: Equatable {
        case idle, listing, ready, downloading, completed
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var downloadedBytes: Int64 = 0
    @Published private(set) var totalBytes: Int64 = 0
    @Published private(set) var completedFiles = 0
    @Published private(set) var totalFiles = 0

    static let revision = "42ed227ee7df40d41602854ae760620d6eb651fe"
    static let licenseURL = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE")!
    private static let repository = "MiniMaxAI/MiniMax-H3"
    private static let diskMargin: Int64 = 2 * 1_073_741_824
    private var files: [Entry] = []
    private var task: Task<Void, Never>?
    private let fileManager = FileManager.default

    var destinationURL: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MiniMax-H3", isDirectory: true)
    }

    var progress: Double {
        guard totalBytes > 0 else { return 0 }
        return min(1, max(0, Double(downloadedBytes) / Double(totalBytes)))
    }

    var totalGiB: Double { Double(totalBytes) / 1_073_741_824 }
    var canStart: Bool {
        guard !files.isEmpty else { return false }
        if state == .ready { return true }
        if case .failed = state { return true }
        return false
    }

    func prepare() {
        guard state != .listing && state != .downloading else { return }
        state = .listing
        task?.cancel()
        task = Task {
            do {
                let entries = try await Self.fetchEntries()
                try Task.checkCancellation()
                guard !entries.isEmpty else { throw H3ModelDownloadError.emptyManifest }
                files = entries.sorted { $0.path < $1.path }
                totalBytes = files.reduce(0) { $0 + $1.expectedSize }
                totalFiles = files.count
                downloadedBytes = existingBytes()
                completedFiles = 0
                state = .ready
            } catch is CancellationError {
                state = .idle
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    func startDownload() {
        guard canStart else { return }
        let root = destinationURL
        do {
            try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
            var modelRoot = root
            var resourceValues = URLResourceValues()
            resourceValues.isExcludedFromBackup = true
            try modelRoot.setResourceValues(resourceValues)
            let attributes = try fileManager.attributesOfFileSystem(forPath: root.path)
            let free = (attributes[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
            let remaining = max(0, totalBytes - existingBytes())
            guard free > 0 else {
                state = .failed("Could not verify free iPhone storage. Free up space and retry after storage information is available.")
                return
            }
            guard free >= remaining + Self.diskMargin else {
                state = .failed("Not enough free storage. Ref2VA needs about \(String(format: "%.1f", Double(remaining + Self.diskMargin) / 1_073_741_824)) GiB including safety margin.")
                return
            }
        } catch {
            state = .failed("Cannot prepare local model storage: \(error.localizedDescription)")
            return
        }

        downloadedBytes = existingBytes()
        completedFiles = 0
        state = .downloading
        task = Task {
            do {
                for entry in files {
                    try Task.checkCancellation()
                    try await Self.download(entry, root: root) { [weak self] delta in
                        Task { @MainActor in
                            guard let self else { return }
                            self.downloadedBytes = max(0, min(self.totalBytes, self.downloadedBytes + delta))
                        }
                    }
                    completedFiles += 1
                }
                state = .completed
            } catch is CancellationError {
                downloadedBytes = existingBytes()
                state = .ready
            } catch {
                downloadedBytes = existingBytes()
                state = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
    }

    func deleteDownloadedModel() throws {
        guard state != .downloading && state != .listing else { return }
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        files.removeAll()
        downloadedBytes = 0
        totalBytes = 0
        completedFiles = 0
        totalFiles = 0
        state = .idle
    }

    private func existingBytes() -> Int64 {
        files.reduce(into: Int64(0)) { total, entry in
            let path = destinationURL.appendingPathComponent(entry.path).path
            let size = (try? fileManager.attributesOfItem(atPath: path)[.size] as? NSNumber)?.int64Value ?? 0
            total += min(size, entry.expectedSize)
        }
    }

    private struct Entry: Decodable {
        struct LFS: Decodable { let oid: String? }
        let type: String
        let path: String
        let size: Int64?
        let oid: String?
        let lfs: LFS?

        var expectedSize: Int64 { size ?? 0 }
        var sha256: String? {
            guard let oid = lfs?.oid else { return nil }
            return oid.replacingOccurrences(of: "sha256:", with: "").lowercased()
        }
        var gitSHA1: String? { oid?.lowercased() }
    }


    private static func fetchEntries() async throws -> [Entry] {
        let urlString = "https://huggingface.co/api/models/\(repository)/tree/\(revision)?recursive=true&expand=true"
        guard let initialURL = URL(string: urlString) else { throw H3ModelDownloadError.listing(-1) }

        // The Hub tree endpoint is paginated. Fetch every page before accepting the
        // manifest; otherwise a first page can look valid while silently omitting
        // later transformer shards, leaving a multi-hundred-GiB checkpoint unusable.
        var nextURL: URL? = initialURL
        var visitedPages = Set<String>()
        var raw: [Entry] = []
        while let pageURL = nextURL {
            guard visitedPages.insert(pageURL.absoluteString).inserted,
                  visitedPages.count <= 10_000 else {
                throw H3ModelDownloadError.emptyManifest
            }
            let (data, response) = try await URLSession.shared.data(from: pageURL)
            guard let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode) else {
                throw H3ModelDownloadError.listing((response as? HTTPURLResponse)?.statusCode ?? -1)
            }
            raw.append(contentsOf: try JSONDecoder().decode([Entry].self, from: data))
            nextURL = Self.nextPageURL(from: http)
        }

        // h3_load_dir expects the repository root, with the task-specific checkpoint
        // under Ref2VA/. Do not flatten the subtree into the root; that layout
        // can pass file hashes but still fail native model discovery.
        let entries = raw.filter {
            guard $0.type == "file", $0.expectedSize > 0 else { return false }
            let components = $0.path.split(separator: "/", omittingEmptySubsequences: false)
            guard !components.isEmpty,
                  components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                  !$0.path.hasPrefix("/"),
                  !$0.path.contains("\\") else { return false }
            return $0.path == "model_index.json" || $0.path.hasPrefix("Ref2VA/")
        }
        guard entries.contains(where: { $0.path == "model_index.json" }) &&
              entries.contains(where: { $0.path == "Ref2VA/model_index.json" }) &&
              entries.allSatisfy({ $0.sha256 != nil || $0.gitSHA1 != nil }) else {
            throw H3ModelDownloadError.emptyManifest
        }
        let uniquePaths = Set(entries.map(\.path))
        guard uniquePaths.count == entries.count else {
            throw H3ModelDownloadError.emptyManifest
        }
        return entries
    }

    private static func nextPageURL(from response: HTTPURLResponse) -> URL? {
        guard let linkHeader = response.value(forHTTPHeaderField: "Link") else { return nil }
        for link in linkHeader.split(separator: ",") {
            guard link.contains("rel=\"next\"") || link.contains("rel=next"),
                  let left = link.firstIndex(of: "<"),
                  let right = link.firstIndex(of: ">"),
                  left < right else { continue }
            let target = String(link[link.index(after: left)..<right])
            return URL(string: target, relativeTo: response.url)?.absoluteURL
        }
        return nil
    }

    private static func download(_ entry: Entry, root: URL, onProgress: @escaping (Int64) -> Void) async throws {
        let finalURL = root.appendingPathComponent(entry.path)
        try FileManager.default.createDirectory(at: finalURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        var attempt = 0
        while true {
            try Task.checkCancellation()
            attempt += 1
            do {
                try await attemptDownload(entry, finalURL: finalURL, onProgress: onProgress)
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if attempt >= 5 { throw error }
                try await Task.sleep(nanoseconds: UInt64(min(30, pow(2, Double(attempt))) * 1_000_000_000))
            }
        }
    }

    private static func attemptDownload(_ entry: Entry, finalURL: URL, onProgress: @escaping (Int64) -> Void) async throws {
        let fm = FileManager.default
        var size = (try? fm.attributesOfItem(atPath: finalURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        if size == entry.expectedSize {
            if try verify(finalURL, entry: entry) { return }
            try? fm.removeItem(at: finalURL)
            onProgress(-size)
            size = 0
        } else if size > entry.expectedSize {
            try? fm.removeItem(at: finalURL)
            onProgress(-size)
            size = 0
        }

        let encodedPath = entry.path.split(separator: "/").map {
            String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0)
        }.joined(separator: "/")
        guard let url = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(encodedPath)") else {
            throw H3ModelDownloadError.http(entry.path, -1)
        }
        var request = URLRequest(url: url)
        let resuming = size > 0
        if resuming { request.setValue("bytes=\(size)-", forHTTPHeaderField: "Range") }

        if !fm.fileExists(atPath: finalURL.path) { fm.createFile(atPath: finalURL.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: finalURL)
        try handle.seekToEnd()
        defer { try? handle.close() }
        let delegate = H3StreamDelegate(handle: handle, expectPartial: resuming, onProgress: onProgress)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }

        do {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    delegate.continuation = continuation
                    session.dataTask(with: request).resume()
                }
            } onCancel: {
                session.invalidateAndCancel()
            }
        } catch H3ModelDownloadError.rangeIgnored {
            try? fm.removeItem(at: finalURL)
            onProgress(-size)
            throw H3ModelDownloadError.http(entry.path, 200)
        }

        let actualSize = (try? fm.attributesOfItem(atPath: finalURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        guard actualSize == entry.expectedSize else { throw H3ModelDownloadError.badSize(entry.path) }
        guard try verify(finalURL, entry: entry) else {
            try? fm.removeItem(at: finalURL)
            onProgress(-actualSize)
            throw H3ModelDownloadError.hashMismatch(entry.path)
        }
    }

    private static func verify(_ url: URL, entry: Entry) throws -> Bool {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        if let expected = entry.sha256 {
            var hasher = SHA256()
            while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty { hasher.update(data: data) }
            let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            return digest == expected
        }
        guard let expected = entry.gitSHA1 else { throw H3ModelDownloadError.badDigest(entry.path) }
        var hasher = Insecure.SHA1()
        hasher.update(data: Data("blob \(entry.expectedSize)\0".utf8))
        while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty { hasher.update(data: data) }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return digest == expected
    }
}

private final class H3StreamDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let handle: FileHandle
    private let expectPartial: Bool
    private let onProgress: (Int64) -> Void
    var continuation: CheckedContinuation<Void, Error>?
    private var finished = false

    init(handle: FileHandle, expectPartial: Bool, onProgress: @escaping (Int64) -> Void) {
        self.handle = handle
        self.expectPartial = expectPartial
        self.onProgress = onProgress
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            finish(H3ModelDownloadError.http("model file", -1))
            return
        }
        if expectPartial && http.statusCode != 206 {
            completionHandler(.cancel)
            finish(H3ModelDownloadError.rangeIgnored)
            return
        }
        guard (200..<300).contains(http.statusCode) else {
            completionHandler(.cancel)
            finish(H3ModelDownloadError.http("model file", http.statusCode))
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try handle.write(contentsOf: data)
            onProgress(Int64(data.count))
        } catch {
            dataTask.cancel()
            finish(error)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        finish(error)
    }

    private func finish(_ error: Error?) {
        guard !finished else { return }
        finished = true
        guard let continuation else { return }
        self.continuation = nil
        if let error { continuation.resume(throwing: error) }
        else { continuation.resume() }
    }
}
