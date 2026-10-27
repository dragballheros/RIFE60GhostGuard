import Foundation

enum StrategyCheckError: Error, CustomStringConvertible {
    case message(String)
    var description: String { switch self { case .message(let text): return text } }
}

func expect(_ condition: Bool, _ message: String) throws {
    if !condition { throw StrategyCheckError.message(message) }
}

let directory = FileManager.default.temporaryDirectory.appendingPathComponent("cugan-strategy-\(UUID().uuidString)", isDirectory: true)
try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
RealCUGANStrategyStore.fileURL = directory.appendingPathComponent("cugan-strategy.json")
RealCUGANStrategyStore.deviceModelOverride = "iPhone15,3"
RealCUGANStrategyStore.systemVersionOverride = "Version 18.0 (Build 22A3354)"
RealCUGANStrategyStore.buildNumberOverride = "1101"

let model = "RealCUGAN2xNoise3_Tile704x608"
let initial = RealCUGANStrategyStore.plan(modelName: model, tileWidth: 704, tileHeight: 608)
try expect(initial.strategy == .sequentialBacked, "default strategy must be sequentialBacked")

RealCUGANStrategyStore.save(RealCUGANStrategyRecord(
    strategy: RealCUGANTileStrategy.concurrentBacked.rawValue,
    slots: 2,
    medianMilliseconds: 12.5,
    peakExtraMB: 40,
    modelName: model,
    tileWidth: 704,
    tileHeight: 608,
    deviceModel: RealCUGANStrategyStore.deviceModel(),
    systemVersion: RealCUGANStrategyStore.systemVersion(),
    buildNumber: RealCUGANStrategyStore.buildNumber(),
    needsRebenchmark: false
))
let saved = RealCUGANStrategyStore.plan(modelName: model, tileWidth: 704, tileHeight: 608)
try expect(saved.strategy == .concurrentBacked && saved.slots == 2, "saved winner should load")

RealCUGANStrategyStore.buildNumberOverride = "1102"
let afterUpdate = RealCUGANStrategyStore.plan(modelName: model, tileWidth: 704, tileHeight: 608)
try expect(afterUpdate.strategy == .sequentialBacked, "a new app build must re-benchmark")

RealCUGANStrategyStore.buildNumberOverride = "1101"
RealCUGANStrategyStore.markNeedsRebenchmark(modelName: model, tileWidth: 704, tileHeight: 608, failed: .concurrentBacked)
let marked = RealCUGANStrategyStore.plan(modelName: model, tileWidth: 704, tileHeight: 608)
try expect(marked.strategy == .sequentialBacked, "a failed saved strategy must fall back")

let delta = RealCUGANBatchTensor.maxRoundTripDelta(width: 32, height: 24)
try expect(delta <= 1, "batch tensor round trip delta \(delta) exceeds 1 LSB")

let slots = RealCUGANStrategyStore.allowedConcurrentSlots(availableMB: 900)
try expect(slots == 2, "900 MB headroom should allow 2 concurrent tiles, got \(slots)")
let wide = RealCUGANStrategyStore.allowedConcurrentSlots(availableMB: 1_700)
try expect(wide == 4, "1700 MB headroom should allow 4 concurrent tiles, got \(wide)")

print("Real-CUGAN strategy store and tensor round trip passed")
