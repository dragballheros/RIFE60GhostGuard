import CryptoKit
import Darwin
import Foundation

/// Tier 1 MiniMax H3 install. No paid API, no cloud inference, no encoder-student
/// distillation, and no DMD training. GitHub Actions is not used to quantize
/// weights: a free runner cannot hold the official tree, and weights are not
/// uploaded as artifacts.
///
/// The phone downloads one manifest file into a scratch path, checks it, keeps
/// it only when the stored set stays within the Tier 1 budget, then deletes the
/// scratch file before the next file starts.
enum H3Tier1OnDeviceConverter {
    static let targetStoreBytes: Int64 = 44 * 1_073_741_824
    /// Raw manifest on the repository default branch. The app also ships the same JSON.
    static let defaultManifestURLString = "https://raw.githubusercontent.com/dragballheros/RIFE60GhostGuard/main/Models/h3-tier1-manifest.json"

    /// The Q8 manifest does not include a tokenizer. The official tokenizer ships in the app
    /// and is copied next to the weights so generation can encode a prompt.
    static func installBundledTokenizer(into modelDirectory: URL) throws {
        guard let bundled = Bundle.main.url(forResource: "h3-tokenizer", withExtension: "json") else {
            return
        }
        let folders = ["FL2VA/tokenizer", "Ref2VA/tokenizer"]
        for relative in folders {
            let directory = modelDirectory.appendingPathComponent(relative, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: destination.path) { continue }
            try FileManager.default.copyItem(at: bundled, to: destination)
        }
    }
    static let peakScratchCeilingBytes: Int64 = 150 * 1_073_741_824
    static let manifestDefaultsKey = "RIFE60.H3.Tier1ManifestURL"

    struct Manifest: Codable, Equatable, Sendable {
        var files: [File]

        struct File: Codable, Equatable, Sendable {
            var path: String
            var url: String
            var sha256: String
            var bytes: Int64
        }
    }

    enum Failure: LocalizedError {
        case missingManifest
        case unsafePath(String)
        case shardTooLarge(String)
        case notEnoughFreeSpace
        case storeBudgetExceeded(String)
        case checksum(String)
        case http(String, Int)

        var errorDescription: String? {
            switch self {
            case .missingManifest:
                return "Set a Tier 1 manifest URL for already-published compressed H3 weights. This phone will not download the full official tree as the stored model."
            case .unsafePath(let path):
                return "Refusing model path \(path)."
            case .shardTooLarge(let path):
                return "\(path) is larger than the 150 GB scratch ceiling. It was not downloaded."
            case .notEnoughFreeSpace:
                return "Not enough free space to hold this one shard plus a 2 GB margin and still delete the scratch file. The kept store can grow to about 44 GB, one file at a time."
            case .storeBudgetExceeded(let path):
                return "Keeping \(path) would pass the about 44 GB Tier 1 store. The scratch file was deleted."
            case .checksum(let path):
                return "Checksum failed for \(path). The scratch file was deleted."
            case .http(let path, let status):
                return "Download failed for \(path) (HTTP \(status))."
            }
        }
    }

    static func convert(
        manifest: Manifest,
        into directory: URL,
        progress: @escaping @Sendable (String) -> Void
    ) async throws {
        let fm = FileManager.default
        let scratch = directory.appendingPathComponent("tier1-scratch", isDirectory: true)
        let kept = directory.appendingPathComponent("tier1-weights", isDirectory: true)
        try fm.createDirectory(at: scratch, withIntermediateDirectories: true)
        try fm.createDirectory(at: kept, withIntermediateDirectories: true)
        var stored = try directorySize(kept)
        for file in manifest.files {
            try Task.checkCancellation()
            let scratchURL = scratch.appendingPathComponent(safeName(file.path))
            let keptURL = kept.appendingPathComponent(safeName(file.path))
            CrashCapture.step("H3 shard \(file.path) bytes=\(file.bytes)")
            if fm.fileExists(atPath: keptURL.path), try await alreadyVerified(file, at: keptURL, directory: directory) {
                progress("Kept \(file.path). Scratch deleted.")
                continue
            }
            if fm.fileExists(atPath: keptURL.path) {
                let digest = try await hashFile(keptURL) { read in
                    progress("bytes \(read)/\(file.bytes) · \(file.path)")
                }
                guard digest == file.sha256.lowercased() else { throw Failure.checksum(file.path) }
                try rememberVerified(file, at: keptURL, digest: digest, directory: directory)
                progress("Kept \(file.path). Scratch deleted.")
                continue
            }
            let present = (try? fm.attributesOfItem(atPath: scratchURL.path)[.size] as? NSNumber)?.int64Value ?? 0
            let bytesStillNeeded = present > 0 && present < file.bytes ? file.bytes - present : (present == file.bytes ? 0 : file.bytes)
            try validate(file, bytesStillNeeded: bytesStillNeeded)
            // A killed app must keep the partial shard. A failed checksum still deletes it.
            defer {
                if !Task.isCancelled {
                    try? fm.removeItem(at: scratchURL)
                }
            }
            let scratchSize = (try? fm.attributesOfItem(atPath: scratchURL.path)[.size] as? NSNumber)?.int64Value ?? 0
            var digest: String?
            if scratchSize == file.bytes, try await alreadyVerified(file, at: scratchURL, directory: directory) {
                digest = file.sha256.lowercased()
                progress("bytes \(file.bytes)/\(file.bytes) · \(file.path)")
            } else if scratchSize == file.bytes {
                CrashCapture.step("H3 checksum \(file.path)")
                digest = try await hashFile(scratchURL) { read in
                    progress("bytes \(read)/\(file.bytes) · \(file.path)")
                }
            }
            if digest == file.sha256.lowercased() {
                progress("bytes \(file.bytes)/\(file.bytes) · \(file.path)")
            } else {
                if scratchSize > file.bytes { try? fm.removeItem(at: scratchURL) }
                CrashCapture.step("H3 downloading \(file.path)")
                progress("Downloading \(file.path)")
                try await download(file, to: scratchURL) { written, expected in
                    progress("bytes \(written)/\(expected) · \(file.path)")
                }
                CrashCapture.step("H3 checksum \(file.path)")
                digest = try await hashFile(scratchURL) { read in
                    progress("bytes \(read)/\(file.bytes) · \(file.path)")
                }
            }
            guard digest == file.sha256.lowercased() else {
                throw Failure.checksum(file.path)
            }
            if file.path.lowercased().hasSuffix(".gguf") {
                CrashCapture.step("H3 verify GGUF header \(file.path)")
                try await Task.detached(priority: .utility) {
                    try H3GGUFStream.confirmDownloadableShard(at: scratchURL, path: file.path)
                }.value
            }
            guard stored + file.bytes <= targetStoreBytes else {
                throw Failure.storeBudgetExceeded(file.path)
            }
            let destination = keptURL
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: scratchURL, to: destination)
            if let digest {
                try rememberVerified(file, at: destination, digest: digest, directory: directory)
            }
            stored += file.bytes
            progress("Kept \(file.path). Scratch deleted.")
        }
    }

    private static func validate(_ file: Manifest.File, bytesStillNeeded: Int64) throws {
        if file.path.contains("..") || file.path.contains("\\") || file.path.hasPrefix("/") {
            throw Failure.unsafePath(file.path)
        }
        if file.bytes <= 0 || file.bytes > peakScratchCeilingBytes {
            throw Failure.shardTooLarge(file.path)
        }
        let margin = bytesStillNeeded == 0 ? Int64(0) : 2 * 1_073_741_824
        if freeBytes() < bytesStillNeeded + margin {
            throw Failure.notEnoughFreeSpace
        }
    }

    /// One ranged piece. A single background task for shard 1 (about 21 GB) is what
    /// iOS kept relaunching; this stays on a foreground session and survives a kill.
    private static let pieceBytes: Int64 = 64 * 1_024 * 1_024

    private static let pieceSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 180
        configuration.timeoutIntervalForResource = 60 * 60
        configuration.waitsForConnectivity = true
        return URLSession(configuration: configuration)
    }()

    private static func download(
        _ file: Manifest.File,
        to destination: URL,
        onBytes: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws {
        guard let url = URL(string: file.url) else { throw Failure.http(file.path, -1) }
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
        }
        var written = (try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? NSNumber)?.int64Value ?? 0
        if written > file.bytes {
            try FileManager.default.removeItem(at: destination)
            FileManager.default.createFile(atPath: destination.path, contents: nil)
            written = 0
        }
        onBytes(written, file.bytes)
        while written < file.bytes {
            try Task.checkCancellation()
            let pieceEnd = min(written + pieceBytes, file.bytes) - 1
            var request = H3ModelDownloader.resumeRequest(for: url, existingBytes: written, end: pieceEnd)
            request.timeoutInterval = 180
            let (temporary, response) = try await pieceSession.download(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            let body = (try? FileManager.default.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            if status == 206 {
                let expectedPiece = pieceEnd - written + 1
                guard body == expectedPiece else {
                    try? FileManager.default.removeItem(at: temporary)
                    throw Failure.http(file.path, status)
                }
                written = try appendFile(temporary, to: destination, already: written, expected: file.bytes, onBytes: onBytes)
            } else if status == 200, body == file.bytes {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: temporary, to: destination)
                written = file.bytes
                onBytes(written, file.bytes)
            } else {
                try? FileManager.default.removeItem(at: temporary)
                throw Failure.http(file.path, status)
            }
        }
        guard written == file.bytes else { throw Failure.http(file.path, -1) }
    }

    private static func appendFile(
        _ temporary: URL,
        to destination: URL,
        already: Int64,
        expected: Int64,
        onBytes: @escaping @Sendable (Int64, Int64) -> Void
    ) throws -> Int64 {
        let input = try FileHandle(forReadingFrom: temporary)
        defer {
            try? input.close()
            try? FileManager.default.removeItem(at: temporary)
        }
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        try output.seekToEnd()
        var written = already
        while true {
            let chunk = try input.read(upToCount: 8 * 1024 * 1024) ?? Data()
            if chunk.isEmpty { break }
            try output.write(contentsOf: chunk)
            written += Int64(chunk.count)
            if written == expected || written % (32 * 1_024 * 1_024) < Int64(chunk.count) {
                onBytes(written, expected)
            }
        }
        return written
    }

    private static let checksumChunkBytes = 4 * 1024 * 1024

    private struct VerifiedRecord: Codable {
        var bytes: Int64
        var modifiedMillis: Int64
        var sha256: String
    }

    private static func verifiedMarkerURL(_ directory: URL) -> URL {
        directory.appendingPathComponent("tier1-verified.json")
    }

    private static func modificationMillis(_ url: URL) -> Int64? {
        guard let date = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate else { return nil }
        return Int64((date.timeIntervalSince1970 * 1000).rounded())
    }

    private static func alreadyVerified(_ file: Manifest.File, at url: URL, directory: URL) async throws -> Bool {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value,
              size == file.bytes,
              let modified = modificationMillis(url),
              let data = try? Data(contentsOf: verifiedMarkerURL(directory)),
              let map = try? JSONDecoder().decode([String: VerifiedRecord].self, from: data),
              let record = map[file.path],
              record.bytes == size,
              record.modifiedMillis == modified,
              record.sha256 == file.sha256.lowercased() else {
            return false
        }
        CrashCapture.step("H3 checksum skipped \(file.path); verified marker matches size and modification date")
        return true
    }

    private static func rememberVerified(_ file: Manifest.File, at url: URL, digest: String, directory: URL) throws {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value,
              let modified = modificationMillis(url) else { return }
        let markerURL = verifiedMarkerURL(directory)
        var map = (try? Data(contentsOf: markerURL)).flatMap { try? JSONDecoder().decode([String: VerifiedRecord].self, from: $0) } ?? [:]
        map[file.path] = VerifiedRecord(bytes: size, modifiedMillis: modified, sha256: digest.lowercased())
        let data = try JSONEncoder().encode(map)
        try data.write(to: markerURL, options: .atomic)
    }

    private static func hashFile(_ url: URL, progress: @escaping @Sendable (Int64) -> Void) async throws -> String {
        try await Task.detached(priority: .utility) {
            try await sha256Streaming(of: url, progress: progress)
        }.value
    }

    private static func sha256Streaming(of url: URL, progress: @escaping @Sendable (Int64) -> Void) async throws -> String {
        var handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        armUncached(handle)
        CrashCapture.step("H3 allocating checksum buffer bytes=\(checksumChunkBytes) for \(url.lastPathComponent)")
        var hasher = SHA256()
        var readTotal: Int64 = 0
        var loggedFirstChunk = false
        while true {
            if CrashCapture.memoryWarningActive {
                try handle.close()
                CrashCapture.step("H3 checksum paused; released \(checksumChunkBytes)B read buffer")
                try await CrashCapture.waitWhileMemoryWarning()
                handle = try FileHandle(forReadingFrom: url)
                armUncached(handle)
                try handle.seek(toOffset: UInt64(readTotal))
                CrashCapture.step("H3 checksum resumed at byte \(readTotal)")
            }
            let chunk: Data = try autoreleasepool {
                try handle.read(upToCount: checksumChunkBytes) ?? Data()
            }
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
            readTotal += Int64(chunk.count)
            if !loggedFirstChunk {
                loggedFirstChunk = true
                CrashCapture.step("H3 checksum first chunk bytes=\(chunk.count) released")
            }
            if readTotal % (32 * 1_024 * 1_024) < Int64(chunk.count) {
                progress(readTotal)
                CrashCapture.note("H3 checksum \(url.lastPathComponent) \(readTotal)")
            }
        }
        CrashCapture.step("H3 checksum finished \(url.lastPathComponent) bytes=\(readTotal)")
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func armUncached(_ handle: FileHandle) {
        _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        armUncached(handle)
        var hasher = SHA256()
        while true {
            let chunk: Data = try autoreleasepool {
                try handle.read(upToCount: checksumChunkBytes) ?? Data()
            }
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func directorySize(_ url: URL) throws -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let item as URL in enumerator {
            let size = try item.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            total += Int64(size)
        }
        return total
    }

    private static func freeBytes() -> Int64 {
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let attributes = try? FileManager.default.attributesOfFileSystem(forPath: root.path)
        return (attributes?[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
    }

    private static func safeName(_ path: String) -> String {
        path.replacingOccurrences(of: "/", with: "_")
    }
}

/// Drains the old whole-file background session. New shards use ranged foreground
/// downloads. The session is created only after the background completion handler
/// is stored, then every legacy task is cancelled so iOS stops relaunching on shard 1.
final class H3Tier1BackgroundTransfer: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    static let shared = H3Tier1BackgroundTransfer()
    static let identifier = "com.za.rife60ghostguard.h3.tier1"

    private let lock = NSLock()
    private var session: URLSession?
    private var backgroundCompletion: (() -> Void)?
    private var handlingBackgroundEvents = false
    private var deliveredCompletion = false

    func beginBackgroundEvents(_ completion: @escaping () -> Void) {
        lock.lock()
        backgroundCompletion = completion
        handlingBackgroundEvents = true
        deliveredCompletion = false
        createSessionLocked()
        let active = session
        lock.unlock()
        active?.invalidateAndCancel()
    }

    private func createSessionLocked() {
        guard session == nil else { return }
        let configuration = URLSessionConfiguration.background(withIdentifier: Self.identifier)
        configuration.sessionSendsLaunchEvents = false
        configuration.isDiscretionary = false
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        downloadTask.cancel()
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // Leave the file in place so the system can delete it. Moving a finished
        // multi-gigabyte shard here blocks the delegate queue and iOS relaunches the app.
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {}

    func urlSession(_ session: URLSession, didBecomeInvalidWithError error: Error?) {
        lock.lock()
        self.session = nil
        lock.unlock()
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        lock.lock()
        if deliveredCompletion {
            lock.unlock()
            return
        }
        deliveredCompletion = true
        let handler = backgroundCompletion
        backgroundCompletion = nil
        lock.unlock()
        DispatchQueue.main.async { handler?() }
    }

}
