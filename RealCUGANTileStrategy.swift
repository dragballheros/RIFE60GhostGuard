import Darwin
import Foundation

/// Interchangeable Real-CUGAN tile execution strategies. Crop and stitch stay on
/// `RealCUGANTileCopy`, so the pixels match the per-tile path within 1 LSB when
/// the model math matches. `sequentialBacked` is the default until Benchmark mode
/// saves a winner for this model, tile, device, OS, and app build.
enum RealCUGANTileStrategy: String, Codable, CaseIterable, Identifiable {
    case sequentialBacked
    case batchedUnbacked
    case concurrentBacked
    case fixedBatch2
    case fixedBatch4

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sequentialBacked: return "Sequential backed"
        case .batchedUnbacked: return "Batched, no backing"
        case .concurrentBacked: return "Concurrent backed"
        case .fixedBatch2: return "Fixed batch 2"
        case .fixedBatch4: return "Fixed batch 4"
        }
    }

    var fixedBatchCount: Int? {
        switch self {
        case .fixedBatch2: return 2
        case .fixedBatch4: return 4
        default: return nil
        }
    }

    func modelResourceName(base: String) -> String {
        guard let count = fixedBatchCount else { return base }
        return "\(base)_Batch\(count)"
    }

    static func isBundled(_ strategy: RealCUGANTileStrategy, modelName: String) -> Bool {
        guard strategy.fixedBatchCount != nil else { return true }
        return Bundle.main.url(forResource: strategy.modelResourceName(base: modelName), withExtension: "mlmodelc") != nil
    }

    /// Strategies worth timing. Fixed-batch models are included only when the
    /// compiled asset is in the bundle. Concurrent and batch-4 need spare memory
    /// on a 6 GB phone.
    static func candidates(modelName: String, availableMB: Double) -> [RealCUGANTileStrategy] {
        var items: [RealCUGANTileStrategy] = [.sequentialBacked]
        if availableMB >= 600 { items.append(.batchedUnbacked) }
        if availableMB >= 700 { items.append(.concurrentBacked) }
        for strategy in [RealCUGANTileStrategy.fixedBatch2, .fixedBatch4] where isBundled(strategy, modelName: modelName) {
            let needed = strategy == .fixedBatch4 ? 1_100.0 : 800.0
            if availableMB >= needed { items.append(strategy) }
        }
        return items
    }
}

struct RealCUGANStrategyPlan: Equatable {
    var strategy: RealCUGANTileStrategy
    var slots: Int
    var medianMilliseconds: Double?
}

struct RealCUGANStrategyRecord: Codable, Equatable {
    var strategy: String
    var slots: Int
    var medianMilliseconds: Double
    var peakExtraMB: Double
    var modelName: String
    var tileWidth: Int
    var tileHeight: Int
    var deviceModel: String
    var systemVersion: String
    var buildNumber: String
    var needsRebenchmark: Bool
}

enum RealCUGANStrategyStore {
    static var fileURL: URL = {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        return base
            .appendingPathComponent("RIFE60GhostGuard", isDirectory: true)
            .appendingPathComponent("cugan-strategy.json")
    }()

    static var deviceModelOverride: String?
    static var systemVersionOverride: String?
    static var buildNumberOverride: String?

    static func plan(modelName: String, tileWidth: Int, tileHeight: Int) -> RealCUGANStrategyPlan {
        guard let record = matchingRecord(modelName: modelName, tileWidth: tileWidth, tileHeight: tileHeight),
              !record.needsRebenchmark,
              let strategy = RealCUGANTileStrategy(rawValue: record.strategy),
              RealCUGANTileStrategy.isBundled(strategy, modelName: modelName) else {
            return RealCUGANStrategyPlan(strategy: .sequentialBacked, slots: 1, medianMilliseconds: nil)
        }
        let slots = strategy == .concurrentBacked ? min(4, max(2, record.slots)) : max(1, record.slots)
        return RealCUGANStrategyPlan(strategy: strategy, slots: slots, medianMilliseconds: record.medianMilliseconds)
    }

    static func save(_ record: RealCUGANStrategyRecord) {
        var records = loadRecords().filter { !sameKey($0, record) }
        records.append(record)
        persist(records)
    }

    static func markNeedsRebenchmark(modelName: String, tileWidth: Int, tileHeight: Int, failed: RealCUGANTileStrategy) {
        var records = loadRecords()
        if let index = records.firstIndex(where: { candidate in
            candidate.modelName == modelName
                && candidate.tileWidth == tileWidth
                && candidate.tileHeight == tileHeight
                && candidate.deviceModel == deviceModel()
                && candidate.systemVersion == systemVersion()
                && candidate.buildNumber == buildNumber()
        }) {
            records[index].needsRebenchmark = true
            records[index].strategy = failed.rawValue
        } else {
            records.append(RealCUGANStrategyRecord(
                strategy: failed.rawValue,
                slots: 1,
                medianMilliseconds: 0,
                peakExtraMB: 0,
                modelName: modelName,
                tileWidth: tileWidth,
                tileHeight: tileHeight,
                deviceModel: deviceModel(),
                systemVersion: systemVersion(),
                buildNumber: buildNumber(),
                needsRebenchmark: true
            ))
        }
        persist(records)
    }

    static func deviceModel() -> String {
        if let deviceModelOverride { return deviceModelOverride }
        var system = utsname()
        uname(&system)
        let machine = system.machine
        return withUnsafeBytes(of: machine) { raw in
            let bytes = raw.bindMemory(to: CChar.self)
            return String(cString: bytes.baseAddress!)
        }
    }

    static func systemVersion() -> String {
        systemVersionOverride ?? ProcessInfo.processInfo.operatingSystemVersionString
    }

    static func buildNumber() -> String {
        if let buildNumberOverride { return buildNumberOverride }
        return Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0"
    }

    static func allowedConcurrentSlots(availableMB: Double) -> Int {
        if availableMB >= 1_600 { return 4 }
        if availableMB >= 1_100 { return 3 }
        if availableMB >= 700 { return 2 }
        return 1
    }

    private static func matchingRecord(modelName: String, tileWidth: Int, tileHeight: Int) -> RealCUGANStrategyRecord? {
        loadRecords().first { record in
            record.modelName == modelName
                && record.tileWidth == tileWidth
                && record.tileHeight == tileHeight
                && record.deviceModel == deviceModel()
                && record.systemVersion == systemVersion()
                && record.buildNumber == buildNumber()
        }
    }

    private static func sameKey(_ lhs: RealCUGANStrategyRecord, _ rhs: RealCUGANStrategyRecord) -> Bool {
        lhs.modelName == rhs.modelName
            && lhs.tileWidth == rhs.tileWidth
            && lhs.tileHeight == rhs.tileHeight
            && lhs.deviceModel == rhs.deviceModel
            && lhs.systemVersion == rhs.systemVersion
            && lhs.buildNumber == rhs.buildNumber
    }

    private static func loadRecords() -> [RealCUGANStrategyRecord] {
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        return (try? JSONDecoder().decode([RealCUGANStrategyRecord].self, from: data)) ?? []
    }

    private static func persist(_ records: [RealCUGANStrategyRecord]) {
        let directory = fileURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(records) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}

/// Packs one BGRA tile into planar RGB floats in 0...255 and unpacks those floats
/// back to BGRA. Integer values round-trip exactly; the CI check rejects a max
/// channel delta above 1.
enum RealCUGANBatchTensor {
    static func pack(
        bgra: UnsafePointer<UInt8>,
        stride: Int,
        width: Int,
        height: Int,
        destination: UnsafeMutablePointer<Float>,
        planeFloats: Int
    ) {
        let green = destination.advanced(by: planeFloats)
        let blue = destination.advanced(by: planeFloats * 2)
        for y in 0..<height {
            let row = bgra.advanced(by: y * stride)
            let offset = y * width
            for x in 0..<width {
                let pixel = row.advanced(by: x * 4)
                destination[offset + x] = Float(pixel[2])
                green[offset + x] = Float(pixel[1])
                blue[offset + x] = Float(pixel[0])
            }
        }
    }

    static func unpack(
        source: UnsafePointer<Float>,
        planeFloats: Int,
        width: Int,
        height: Int,
        bgra: UnsafeMutablePointer<UInt8>,
        stride: Int
    ) {
        let green = source.advanced(by: planeFloats)
        let blue = source.advanced(by: planeFloats * 2)
        for y in 0..<height {
            let row = bgra.advanced(by: y * stride)
            let offset = y * width
            for x in 0..<width {
                let pixel = row.advanced(by: x * 4)
                pixel[0] = byte(blue[offset + x])
                pixel[1] = byte(green[offset + x])
                pixel[2] = byte(source[offset + x])
                pixel[3] = 255
            }
        }
    }

    static func maxRoundTripDelta(width: Int, height: Int) -> Int {
        let rowBytes = width * 4
        var source = [UInt8](repeating: 0, count: rowBytes * height)
        for index in 0..<source.count {
            source[index] = UInt8((index * 17 + 3) & 0xFF)
        }
        let plane = width * height
        var floats = [Float](repeating: 0, count: plane * 3)
        var restored = [UInt8](repeating: 0, count: source.count)
        source.withUnsafeBufferPointer { sourceBuffer in
            floats.withUnsafeMutableBufferPointer { floatBuffer in
                pack(bgra: sourceBuffer.baseAddress!, stride: rowBytes, width: width, height: height, destination: floatBuffer.baseAddress!, planeFloats: plane)
            }
        }
        floats.withUnsafeBufferPointer { floatBuffer in
            restored.withUnsafeMutableBufferPointer { restoredBuffer in
                unpack(source: floatBuffer.baseAddress!, planeFloats: plane, width: width, height: height, bgra: restoredBuffer.baseAddress!, stride: rowBytes)
            }
        }
        var maxDelta = 0
        for index in stride(from: 0, to: source.count, by: 4) {
            maxDelta = max(maxDelta, abs(Int(source[index]) - Int(restored[index])))
            maxDelta = max(maxDelta, abs(Int(source[index + 1]) - Int(restored[index + 1])))
            maxDelta = max(maxDelta, abs(Int(source[index + 2]) - Int(restored[index + 2])))
        }
        return maxDelta
    }

    private static func byte(_ value: Float) -> UInt8 {
        let rounded = Int(value.rounded())
        return UInt8(min(255, max(0, rounded)))
    }
}
