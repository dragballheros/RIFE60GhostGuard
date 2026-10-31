import Foundation

/// Saved Sharpie tile size and whether tiles are predicted together.
/// The default stays the shipped 256 sequential path until Benchmark mode
/// picks a faster bundled variant.
enum SharpieTileStrategy: String, Codable, CaseIterable, Identifiable {
    case sequential256
    case batched256
    case tile768
    case tile1024

    var id: String { rawValue }
    var title: String {
        switch self {
        case .sequential256: return "Sharpie 256 sequential"
        case .batched256: return "Sharpie 256 batched"
        case .tile768: return "Sharpie 768"
        case .tile1024: return "Sharpie 1024"
        }
    }
    var resourceName: String {
        switch self {
        case .sequential256, .batched256: return "AnimeSharpieOutlineSubtle"
        case .tile768: return "AnimeSharpieOutlineSubtle_Tile768"
        case .tile1024: return "AnimeSharpieOutlineSubtle_Tile1024"
        }
    }
    var tile: Int {
        switch self {
        case .sequential256, .batched256: return 256
        case .tile768: return 768
        case .tile1024: return 1024
        }
    }
    var usesUnbackedBatch: Bool { self == .batched256 }

    static func isBundled(_ strategy: SharpieTileStrategy) -> Bool {
        Bundle.main.url(forResource: strategy.resourceName, withExtension: "mlpackage") != nil
    }

    static func candidates() -> [SharpieTileStrategy] {
        allCases.filter(isBundled)
    }
}

struct SharpieTileRecord: Codable {
    var strategy: String
    var medianMilliseconds: Double
    var deviceModel: String
    var systemVersion: String
    var buildNumber: String
    var needsRebenchmark: Bool
}

enum SharpieTileStore {
    static var fileURL: URL = {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base
            .appendingPathComponent("RIFE60GhostGuard", isDirectory: true)
            .appendingPathComponent("sharpie-tile.json")
    }()

    static func plan() -> SharpieTileStrategy {
        guard let record = load(),
              !record.needsRebenchmark,
              record.deviceModel == RealCUGANStrategyStore.deviceModel(),
              record.systemVersion == RealCUGANStrategyStore.systemVersion(),
              record.buildNumber == RealCUGANStrategyStore.buildNumber(),
              let strategy = SharpieTileStrategy(rawValue: record.strategy),
              SharpieTileStrategy.isBundled(strategy) else {
            return .sequential256
        }
        return strategy
    }

    static func save(_ strategy: SharpieTileStrategy, medianMilliseconds: Double) {
        let record = SharpieTileRecord(
            strategy: strategy.rawValue,
            medianMilliseconds: medianMilliseconds,
            deviceModel: RealCUGANStrategyStore.deviceModel(),
            systemVersion: RealCUGANStrategyStore.systemVersion(),
            buildNumber: RealCUGANStrategyStore.buildNumber(),
            needsRebenchmark: false
        )
        if let data = try? JSONEncoder().encode(record) {
            try? FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? data.write(to: fileURL, options: .atomic)
        }
    }

    static func markNeedsRebenchmark() {
        var record = load() ?? SharpieTileRecord(
            strategy: SharpieTileStrategy.sequential256.rawValue,
            medianMilliseconds: 0,
            deviceModel: RealCUGANStrategyStore.deviceModel(),
            systemVersion: RealCUGANStrategyStore.systemVersion(),
            buildNumber: RealCUGANStrategyStore.buildNumber(),
            needsRebenchmark: true
        )
        record.needsRebenchmark = true
        if let data = try? JSONEncoder().encode(record) {
            try? data.write(to: fileURL, options: .atomic)
        }
    }

    private static func load() -> SharpieTileRecord? {
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        return try? JSONDecoder().decode(SharpieTileRecord.self, from: data)
    }
}
