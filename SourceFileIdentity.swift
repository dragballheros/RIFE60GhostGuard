import CryptoKit
import Foundation

/// Stable content identity for a media source, calculated incrementally so a
/// multi-gigabyte video is never loaded into RAM just to verify recovery.
enum SourceFileIdentity {
    static func sha256(of url: URL, chunkSize: Int = 4 * 1024 * 1024) throws -> String {
        guard chunkSize > 0 else {
            throw NSError(domain: "RIFE60SourceIdentity", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Hash chunk size must be positive."])
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard (attributes[.type] as? FileAttributeType) == .typeRegular else {
            throw NSError(domain: "RIFE60SourceIdentity", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Source identity requires a regular file."])
        }

        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: chunkSize) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func filesMatch(_ lhs: URL, _ rhs: URL) -> Bool {
        if lhs.standardizedFileURL == rhs.standardizedFileURL {
            return FileManager.default.fileExists(atPath: lhs.path)
        }
        guard let leftSize = fileSize(lhs), let rightSize = fileSize(rhs), leftSize == rightSize else {
            return false
        }
        do {
            return try sha256(of: lhs) == sha256(of: rhs)
        } catch {
            return false
        }
    }

    private static func fileSize(_ url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let number = attributes[.size] as? NSNumber else { return nil }
        return number.int64Value
    }
}
