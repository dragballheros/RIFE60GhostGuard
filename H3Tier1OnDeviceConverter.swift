import CryptoKit
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
    static let targetStoreBytes: Int64 = 30 * 1_073_741_824
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
                return "Not enough free space to hold one shard and still delete it after the check."
            case .storeBudgetExceeded(let path):
                return "Keeping \(path) would pass the about 30 GB Tier 1 store. The scratch file was deleted."
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
            try validate(file)
            let scratchURL = scratch.appendingPathComponent(safeName(file.path))
            defer { try? fm.removeItem(at: scratchURL) }
            progress("Downloading \(file.path)")
            try await download(file, to: scratchURL)
            guard try sha256(of: scratchURL) == file.sha256.lowercased() else {
                throw Failure.checksum(file.path)
            }
            guard stored + file.bytes <= targetStoreBytes else {
                throw Failure.storeBudgetExceeded(file.path)
            }
            let destination = kept.appendingPathComponent(safeName(file.path))
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: scratchURL, to: destination)
            stored += file.bytes
            progress("Kept \(file.path). Scratch deleted.")
        }
    }

    private static func validate(_ file: Manifest.File) throws {
        if file.path.contains("..") || file.path.contains("\\") || file.path.hasPrefix("/") {
            throw Failure.unsafePath(file.path)
        }
        if file.bytes <= 0 || file.bytes > peakScratchCeilingBytes {
            throw Failure.shardTooLarge(file.path)
        }
        let free = freeBytes()
        if free < file.bytes + (2 * 1_073_741_824) {
            throw Failure.notEnoughFreeSpace
        }
    }

    private static func download(_ file: Manifest.File, to destination: URL) async throws {
        guard let url = URL(string: file.url) else { throw Failure.http(file.path, -1) }
        let (temporary, response) = try await URLSession.shared.download(from: url)
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard (200..<300).contains(status) else {
            try? FileManager.default.removeItem(at: temporary)
            throw Failure.http(file.path, status)
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: temporary, to: destination)
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 8 * 1_024 * 1_024) ?? Data()
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
