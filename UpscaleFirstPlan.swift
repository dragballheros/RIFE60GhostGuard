import Foundation

/// Small, testable admission and checkpoint planner shared by the real pipeline.
enum UpscaleFirstPlan {
    static func stages(restoration: Bool, outline: Bool) -> [String] {
        var result = restoration ? ["restored", "cugan", "rife"] : ["cugan", "rife"]
        if outline { result.append("outline") }
        return result
    }
    static func furthestValid(stages: [String], validate: (Int, String) async -> Bool) async -> Int? {
        for index in stages.indices.reversed() {
            if await validate(index, stages[index]) { return index }
        }
        return nil
    }
    static func requiredMemoryMB(width: Int, height: Int) -> Double {
        max(1_700, 900 + Double(width) * Double(height) * 192 / 1_048_576 + 700)
    }
    static func admits(width: Int, height: Int, availableMB: Double, performance: Bool) -> Bool {
        width > 0 && height > 0 && Double(width) * Double(height) <= 3840 * 2160 &&
        performance && availableMB >= requiredMemoryMB(width: width, height: height)
    }
}
