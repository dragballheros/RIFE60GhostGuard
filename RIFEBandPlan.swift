import Foundation

/// Headroom buckets choose a smaller working surface, not an estimated admission
/// limit. Only the governor's existing 700 MB emergency floor can stop a job.
enum RIFEBandPlan {
    static func bands(width: Int, height: Int, availableMB: Double, underPressure: Bool) -> Int {
        let pixels = Double(width) * Double(height)
        var count: Int
        if availableMB >= 3500 { count = 2 }
        else if availableMB >= 2200 { count = 4 }
        else if availableMB >= 1400 { count = 6 }
        else { count = 8 }
        if underPressure { count = max(count, 8) }
        if pixels <= 1920 * 1080 && availableMB >= 2200 && !underPressure { count = 2 }
        return min(max(count, 1), max(height, 1))
    }
}
