import Foundation

func require(_ condition: Bool, _ message: String) { if !condition { fatalError(message) } }
require(ModelComputePreference.auto.units == .all, "Auto must preserve .all")
require(ModelComputePreference.gpu.units == .cpuAndGPU, "GPU policy")
require(ModelComputePreference.neuralEngine.units == .cpuAndNeuralEngine, "ANE policy")
require(UpscaleFirstPlan.stages(restoration: true, outline: true) == ["restored", "cugan", "rife", "outline"], "stage order")
require(UpscaleFirstPlan.stages(restoration: false, outline: false) == ["cugan", "rife"], "optional stages")
require(!UpscaleFirstPlan.admits(width: 3840, height: 2160, availableMB: 2000, performance: true), "4K low-memory refusal")
require(UpscaleFirstPlan.admits(width: 3840, height: 2160, availableMB: 4000, performance: true), "4K healthy admission")
require(!UpscaleFirstPlan.admits(width: 3840, height: 2160, availableMB: 4000, performance: false), "governor refusal")
require(!UpscaleFirstPlan.admits(width: 7680, height: 4320, availableMB: 12000, performance: true), "above-4K refusal")
for restoration in [false, true] {
    for outline in [false, true] {
        let stages = UpscaleFirstPlan.stages(restoration: restoration, outline: outline)
        for mask in 0..<(1 << stages.count) {
            let result = await UpscaleFirstPlan.furthestValid(stages: stages) { index, _ in mask & (1 << index) != 0 }
            let expected = stages.indices.last { mask & (1 << $0) != 0 }
            require(result == expected, "furthest checkpoint selection with deleted/invalid earlier stages")
        }
    }
}
print("SPEED_OPTIONS_PASS: policies, memory admission, every checkpoint combination")
