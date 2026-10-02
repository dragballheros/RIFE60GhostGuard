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
    static func admits(width: Int, height: Int) -> Bool {
        width > 0 && height > 0 && Double(width) * Double(height) <= 3840 * 2160
    }
}
