import CoreMedia
import Foundation

/// Compact luma signature used only to notice a hard camera cut.
struct VACEFrameSignature {
    var histogram: [Float]
    var luma: [UInt8]
}

/// On-device shot cuts from a luma histogram and a coarse frame difference.
/// This does not change the frames that are edited.
enum VACEShotCut {
    static let histogramCutDistance: Float = 0.55
    static let frameDifferenceCut: Float = 36

    static func signature(of frame: VACERGBAFrame) -> VACEFrameSignature {
        let bins = 32
        var histogram = [Float](repeating: 0, count: bins)
        let gridWidth = 80
        let gridHeight = max(1, gridWidth * frame.height / max(frame.width, 1))
        var sums = [Int](repeating: 0, count: gridWidth * gridHeight)
        var counts = [Int](repeating: 0, count: gridWidth * gridHeight)
        var samples = 0
        var y = 0
        while y < frame.height {
            var x = 0
            while x < frame.width {
                let offset = (y * frame.width + x) * 4
                let luma = (Int(frame.bytes[offset]) * 54
                    + Int(frame.bytes[offset + 1]) * 183
                    + Int(frame.bytes[offset + 2]) * 19) >> 8
                histogram[min(bins - 1, luma * bins / 256)] += 1
                samples += 1
                let gridX = min(gridWidth - 1, x * gridWidth / frame.width)
                let gridY = min(gridHeight - 1, y * gridHeight / frame.height)
                let index = gridY * gridWidth + gridX
                sums[index] += luma
                counts[index] += 1
                x += 2
            }
            y += 2
        }
        if samples > 0 {
            for index in histogram.indices {
                histogram[index] /= Float(samples)
            }
        }
        var grid = [UInt8](repeating: 0, count: sums.count)
        for index in grid.indices where counts[index] > 0 {
            grid[index] = UInt8(sums[index] / counts[index])
        }
        return VACEFrameSignature(histogram: histogram, luma: grid)
    }

    static func isCut(from previous: VACEFrameSignature, to current: VACEFrameSignature) -> (cut: Bool, histogramDistance: Float, frameDifference: Float) {
        let count = min(previous.histogram.count, current.histogram.count)
        var histogramDistance: Float = 0
        for index in 0..<count {
            histogramDistance += abs(previous.histogram[index] - current.histogram[index])
        }
        let lumaCount = min(previous.luma.count, current.luma.count)
        var difference: Float = 0
        if lumaCount > 0 {
            for index in 0..<lumaCount {
                difference += abs(Float(previous.luma[index]) - Float(current.luma[index]))
            }
            difference /= Float(lumaCount)
        }
        let cut = histogramDistance >= histogramCutDistance || difference >= frameDifferenceCut
        return (cut, histogramDistance, difference)
    }

    static func logCut(at time: CMTime, histogramDistance: Float, frameDifference: Float) {
        let seconds = time.seconds.isFinite ? time.seconds : 0
        DiagnosticsLogger.shared.log(
            String(format: "VACE shot cut at %.3fs (histogram L1 %.3f, frame difference %.1f)",
                   seconds, histogramDistance, frameDifference)
        )
    }
}
