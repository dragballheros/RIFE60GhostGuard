import Combine
import CryptoKit
import Foundation

private enum H3ModelDownloadError: LocalizedError {
    case listing(Int)
    case emptyManifest
    case http(String, Int)

    var errorDescription: String? {
        switch self {
        case .listing(let status): return "Could not list official H3 model files (HTTP \(status))."
        case .emptyManifest: return "The pinned H3 model manifest did not contain all required FL2VA base and Ref2VA transformer assets or verifiable checksums."
        case .http(let path, let status): return "Download failed for \(path) (HTTP \(status))."
        }
    }
}


/// Downloads the FL2VA base assets plus the Ref2VA transformer used by native reference-conditioned generation. Model weights are never bundled in the IPA.
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
    @Published private(set) var currentFile = ""
    @Published private(set) var shardIndex = 0
    @Published private(set) var fileBytes: Int64 = 0
    @Published private(set) var fileTotalBytes: Int64 = 0
    @Published private(set) var tapNote = ""

    static let revision = "42ed227ee7df40d41602854ae760620d6eb651fe"
    static let licenseURL = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE")!
    private static let repository = "MiniMaxAI/MiniMax-H3"
    private var files: [Entry] = []
    private var task: Task<Void, Never>?
    private var running = false
    private let fileManager = FileManager.default

    init() {
        H3Tier1BackgroundTransfer.shared.activate()
        running = true
        task = Task { await resumeSavedTier1Job() }
    }

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

    func reportBlocked(_ message: String) {
        tapNote = ""
        state = .failed(message)
    }

    /// Tier 1: one published compressed shard at a time. Does not fetch the full official tree.
    func startDownload() {
        if running || state == .downloading || state == .listing {
            let name = currentFile.isEmpty ? "current shard" : currentFile
            let count = max(totalFiles, shardIndex)
            tapNote = "Download is already running: shard \(max(shardIndex, 1)) of \(max(count, 1)), \(name), \(fileBytes) of \(fileTotalBytes) bytes."
            return
        }
        tapNote = ""
        running = true
        state = .listing
        task = Task { await runTier1Download(resumingSavedManifest: false) }
    }

    private var savedManifestURL: URL {
        destinationURL.appendingPathComponent("tier1-manifest.json")
    }

    private func resumeSavedTier1Job() async {
        let tasks = await H3Tier1BackgroundTransfer.shared.outstandingDownloadCount()
        guard tasks > 0 || fileManager.fileExists(atPath: savedManifestURL.path) else {
            running = false
            return
        }
        await runTier1Download(resumingSavedManifest: true)
    }

    private func runTier1Download(resumingSavedManifest: Bool) async {
        defer { running = false }
        do {
            let manifest: H3Tier1OnDeviceConverter.Manifest
            if resumingSavedManifest, let data = try? Data(contentsOf: savedManifestURL) {
                manifest = try JSONDecoder().decode(H3Tier1OnDeviceConverter.Manifest.self, from: data)
            } else {
                let raw = UserDefaults.standard.string(forKey: H3Tier1OnDeviceConverter.manifestDefaultsKey)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let manifestString = raw.isEmpty ? H3Tier1OnDeviceConverter.defaultManifestURLString : raw
                guard let url = URL(string: manifestString), url.scheme == "https" else {
                    throw H3Tier1OnDeviceConverter.Failure.missingManifest
                }
                currentFile = "manifest"
                state = .listing
                let data: Data
                if let fetched = try? await Self.fetchManifest(url) {
                    data = fetched
                } else if manifestString == H3Tier1OnDeviceConverter.defaultManifestURLString,
                          let bundled = Bundle.main.url(forResource: "h3-tier1-manifest", withExtension: "json"),
                          let local = try? Data(contentsOf: bundled) {
                    data = local
                    tapNote = "Using the Q8 Tier 1 manifest shipped with this build."
                } else {
                    throw H3Tier1OnDeviceConverter.Failure.http("manifest", -1)
                }
                try Task.checkCancellation()
                manifest = try JSONDecoder().decode(H3Tier1OnDeviceConverter.Manifest.self, from: data)
                guard !manifest.files.isEmpty else { throw H3ModelDownloadError.emptyManifest }
                try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true)
                try data.write(to: savedManifestURL, options: .atomic)
            }
            guard !manifest.files.isEmpty else { throw H3ModelDownloadError.emptyManifest }
            totalFiles = manifest.files.count
            totalBytes = manifest.files.reduce(Int64(0)) { $0 + $1.bytes }
            state = .downloading
            try await H3Tier1OnDeviceConverter.convert(manifest: manifest, into: destinationURL) { [weak self] message in
                Task { @MainActor in
                    self?.applyTier1Progress(message, files: manifest.files)
                }
            }
            try Task.checkCancellation()
            try? fileManager.removeItem(at: savedManifestURL)
            currentFile = ""
            fileBytes = totalBytes
            fileTotalBytes = totalBytes
            shardIndex = totalFiles
            completedFiles = totalFiles
            downloadedBytes = totalBytes
            state = .completed
            tapNote = "Tier 1 download finished. \(totalFiles) shard(s) kept. Scratch deleted."
        } catch is CancellationError {
            if state == .listing || state == .downloading {
                state = .ready
            }
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private static func fetchManifest(_ url: URL) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: url)
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard (200..<300).contains(status) else {
            throw H3Tier1OnDeviceConverter.Failure.http("manifest", status)
        }
        return data
    }

    private func applyTier1Progress(_ message: String, files: [H3Tier1OnDeviceConverter.Manifest.File]) {
        if message.hasPrefix("bytes ") {
            let body = message.dropFirst("bytes ".count)
            let pieces = body.split(separator: " · ", maxSplits: 1, omittingEmptySubsequences: false)
            let counts = pieces.first?.split(separator: "/", maxSplits: 1) ?? []
            if counts.count == 2 {
                fileBytes = Int64(counts[0]) ?? fileBytes
                fileTotalBytes = Int64(counts[1]) ?? fileTotalBytes
            }
            if pieces.count == 2 { currentFile = String(pieces[1]) }
        } else if message.hasPrefix("Downloading ") {
            currentFile = String(message.dropFirst("Downloading ".count))
            fileBytes = 0
        } else if message.hasPrefix("Kept ") {
            let name = String(message.dropFirst("Kept ".count)).replacingOccurrences(of: ". Scratch deleted.", with: "")
            currentFile = name
        }
        if let index = files.firstIndex(where: { $0.path == currentFile }) {
            shardIndex = index + 1
            completedFiles = message.hasPrefix("Kept ") ? index + 1 : index
            let prior = files.prefix(index).reduce(Int64(0)) { $0 + $1.bytes }
            downloadedBytes = prior + (message.hasPrefix("Kept ") ? files[index].bytes : fileBytes)
        }
        tapNote = "Shard \(max(shardIndex, 1)) of \(files.count): \(currentFile) · \(fileBytes) / \(fileTotalBytes) bytes"
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
        // later transformer shards, leaving an incomplete checkpoint unusable.
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
        // under Ref2VA/transformer/. Do not flatten the subtree into the root; that layout
        // can pass file hashes but still fail native model discovery.
        let entries = raw.filter {
            guard $0.type == "file", $0.expectedSize > 0 else { return false }
            let components = $0.path.split(separator: "/", omittingEmptySubsequences: false)
            guard !components.isEmpty,
                  components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                  !$0.path.hasPrefix("/"),
                  !$0.path.contains("\\") else { return false }
            // The pinned native h3_load_dir implementation uses FL2VA for the
            // shared tokenizer, text encoder, VAEs, and base transformer. For
            // reference-conditioned generation it inventories only the Ref2VA
            // transformer checkpoint; it does not load Ref2VA's duplicate tokenizer,
            // text encoder, or VAEs. Avoid downloading those redundant task assets.
            return $0.path == "model_index.json" ||
                $0.path.hasPrefix("FL2VA/") ||
                $0.path.hasPrefix("Ref2VA/transformer/")
        }
        let requiredNativeFiles = [
            "model_index.json",
            "FL2VA/transformer/config.json",
            "FL2VA/tokenizer/tokenizer.json",
            "Ref2VA/transformer/config.json",
            "Ref2VA/transformer/model.safetensors.index.json"
        ]
        guard requiredNativeFiles.allSatisfy({ required in entries.contains(where: { $0.path == required }) }) &&
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

    /// Appends to a partial file when a background task is no longer in the system session.
    static func resumeRequest(for url: URL, existingBytes: Int64) -> URLRequest {
        var request = URLRequest(url: url)
        if existingBytes > 0 {
            request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
        }
        return request
    }

    static func checksumMatches(url: URL, sha256 expectedSHA: String?, gitSHA1: String?, size: Int64) throws -> Bool {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        if let expectedSHA {
            var hasher = SHA256()
            while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty {
                hasher.update(data: data)
            }
            let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            return digest == expectedSHA.lowercased()
        }
        guard let gitSHA1 else { return false }
        var hasher = Insecure.SHA1()
        hasher.update(data: Data("blob \(size)\0".utf8))
        while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty {
            hasher.update(data: data)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return digest == gitSHA1.lowercased()
    }
}
