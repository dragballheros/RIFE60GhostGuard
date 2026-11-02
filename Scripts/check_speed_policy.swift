import Foundation

enum SpeedPolicyError: Error, CustomStringConvertible {
    case message(String)
    var description: String { switch self { case .message(let text): return text } }
}

func expect(_ condition: Bool, _ message: String) throws {
    if !condition { throw SpeedPolicyError.message(message) }
}

try expect(SharpieTileStrategy.unbenchmarkedDefault() == .batched256, "unbenchmarked Sharpie must batch the 256 tile")
try expect(SharpieTileStrategy.batched256.tile == SharpieTileStrategy.sequential256.tile, "batched Sharpie tile must match sequential 256")
try expect(SharpieTileStrategy.batched256.resourceName == SharpieTileStrategy.sequential256.resourceName, "batched Sharpie must use the same model")
try expect(SharpieTileStrategy.batched256.usesUnbackedBatch, "batched Sharpie must take the unbacked batch path")

let cool = RealCUGANStrategyStore.concurrentCandidate(availableMB: 1_800, tierSafe: false)
try expect(cool?.strategy == .concurrentBacked && (cool?.slots ?? 0) >= 2, "enough memory should offer concurrent CUGAN slots")
let hot = RealCUGANStrategyStore.concurrentCandidate(availableMB: 1_800, tierSafe: true)
try expect(hot == nil, "thermal safe must not raise CUGAN concurrency")
let tight = RealCUGANStrategyStore.concurrentCandidate(availableMB: 500, tierSafe: false)
try expect(tight == nil, "low memory must stay sequential")

let untouched = RealCUGANStrategyStore.plan(modelName: "RealCUGAN2xNoise3_Tile704x608", tileWidth: 704, tileHeight: 608)
try expect(untouched.strategy == .sequentialBacked && untouched.slots == 1, "plan() with no record stays sequentialBacked")

print("Sharpie batch policy and CUGAN slot policy passed")
