import Foundation

/// Disk resume record for the x265 preset stage. A finished file is reused
/// after the app is killed; a partial file is discarded and encoded again.
struct X265PresetCacheRecord: Codable, Equatable {
    var sourceSHA256: String
    var preset: String
    var complete: Bool
    var outputBytes: Int64
}

enum X265PresetCache {
    static func directory(fileManager: FileManager = .default) -> URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("X265PresetCache", isDirectory: true)
    }

    static func outputURL(sourceSHA256: String, fileManager: FileManager = .default) -> URL {
        directory(fileManager: fileManager).appendingPathComponent(sourceSHA256).appendingPathExtension("mov")
    }

    static func partialURL(sourceSHA256: String, fileManager: FileManager = .default) -> URL {
        outputURL(sourceSHA256: sourceSHA256, fileManager: fileManager).appendingPathExtension("partial")
    }

    static func manifestURL(sourceSHA256: String, fileManager: FileManager = .default) -> URL {
        directory(fileManager: fileManager).appendingPathComponent(sourceSHA256).appendingPathExtension("json")
    }

    static func reusableOutput(sourceSHA256: String, preset: String, record: X265PresetCacheRecord?, outputBytes: Int64?) -> URL? {
        guard let record, record.complete, record.sourceSHA256 == sourceSHA256, record.preset == preset,
              let outputBytes, outputBytes == record.outputBytes, outputBytes > 0 else { return nil }
        return outputURL(sourceSHA256: sourceSHA256)
    }

    static func loadRecord(sourceSHA256: String, fileManager: FileManager = .default) -> X265PresetCacheRecord? {
        guard let data = try? Data(contentsOf: manifestURL(sourceSHA256: sourceSHA256, fileManager: fileManager)) else { return nil }
        return try? JSONDecoder().decode(X265PresetCacheRecord.self, from: data)
    }

    static func fileSize(_ url: URL, fileManager: FileManager = .default) -> Int64? {
        guard let number = try? fileManager.attributesOfItem(atPath: url.path)[.size] as? NSNumber else { return nil }
        return number.int64Value
    }
}
