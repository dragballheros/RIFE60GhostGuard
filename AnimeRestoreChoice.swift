import Foundation

enum AnimeRestoreVariant: String, Codable, CaseIterable, Identifiable {
    case apisr
    case span

    var id: String { rawValue }
    var title: String {
        switch self {
        case .apisr: return "APISR 2× RRDB"
        case .span: return "SPAN 2× HFA2k"
        }
    }
    var zipName: String {
        switch self {
        case .apisr: return "APISR2xRRDB.mlpackage.zip"
        case .span: return "SPAN2xHFA2k.mlpackage.zip"
        }
    }
    var packageDirectoryName: String {
        switch self {
        case .apisr: return "APISR2xRRDB.mlpackage"
        case .span: return "SPAN2xHFA2k.mlpackage"
        }
    }
}

enum AnimeRestoreQuality: String, CaseIterable, Identifiable {
    case auto
    case apisr
    case span

    var id: String { rawValue }
    var title: String {
        switch self {
        case .auto: return "Auto"
        case .apisr: return "Max (APISR)"
        case .span: return "Fast (SPAN)"
        }
    }
}

struct AnimeRestoreChoiceRecord: Codable {
    var variant: String
    var apisrMillisecondsPerFrame: Double
    var effectiveMillisecondsPerFrame: Double
    var deviceModel: String
    var systemVersion: String
    var buildNumber: String
    var needsRebenchmark: Bool
}

enum AnimeRestoreChoice {
    /// Unique-frame cost is divided by this because anime-on-2s repeats a frame.
    static let duplicateDivisor = 2.0
    static let budgetMilliseconds = 1000.0

    static var fileURL: URL = {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base
            .appendingPathComponent("RIFE60GhostGuard", isDirectory: true)
            .appendingPathComponent("anime-restore-choice.json")
    }()

    static func pick(quality: AnimeRestoreQuality, apisrMillisecondsPerFrame: Double) -> (AnimeRestoreVariant, Double) {
        let effective = apisrMillisecondsPerFrame / duplicateDivisor
        switch quality {
        case .apisr: return (.apisr, effective)
        case .span: return (.span, effective)
        case .auto: return (effective <= budgetMilliseconds ? .apisr : .span, effective)
        }
    }

    static func savedVariant() -> AnimeRestoreVariant? {
        guard let record = load(),
              RealCUGANStrategyStore.installIsCurrent(
                device: record.deviceModel,
                system: record.systemVersion,
                build: record.buildNumber,
                needsRebenchmark: record.needsRebenchmark
              ),
              let variant = AnimeRestoreVariant(rawValue: record.variant) else { return nil }
        return variant
    }

    static func save(variant: AnimeRestoreVariant, apisrMillisecondsPerFrame: Double, effective: Double) {
        let record = AnimeRestoreChoiceRecord(
            variant: variant.rawValue,
            apisrMillisecondsPerFrame: apisrMillisecondsPerFrame,
            effectiveMillisecondsPerFrame: effective,
            deviceModel: RealCUGANStrategyStore.deviceModel(),
            systemVersion: RealCUGANStrategyStore.systemVersion(),
            buildNumber: RealCUGANStrategyStore.buildNumber(),
            needsRebenchmark: false
        )
        let directory = fileURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(record) {
            try? data.write(to: fileURL, options: .atomic)
        }
    }

    static func load() -> AnimeRestoreChoiceRecord? {
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        return try? JSONDecoder().decode(AnimeRestoreChoiceRecord.self, from: data)
    }

    static func tileCount(length: Int, tile: Int = 256, overlap: Int = 16) -> Int {
        guard length > 0 else { return 0 }
        if length <= tile { return 1 }
        let stride = tile - overlap
        var count = 0
        var origin = 0
        while origin + tile < length {
            count += 1
            origin += stride
        }
        return count + 1
    }
}
