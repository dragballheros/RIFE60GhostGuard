import Foundation
import CoreVideo

/// Detects a real source-frame outlier before interpolation.
/// A legitimate scene cut normally has a large A→C discontinuity while B
/// remains temporally consistent with one side. A stray one-frame "ghost"
/// instead creates a detour: A→B→C is substantially larger than A→C.
final class SourceFrameOutlierGuard {
    struct Result {
        let isOutlier: Bool
        let score: Double
        let reason: String
    }

    private let sampleColumns = 64
    private let sampleRows = 36

    func inspect(previous: CVPixelBuffer, candidate: CVPixelBuffer, next: CVPixelBuffer) -> Result {
        guard let a = Sample(previous), let b = Sample(candidate), let c = Sample(next) else {
            return Result(isOutlier: false, score: 0, reason: "analysis unavailable")
        }

        let ab = a.meanAbsoluteDifference(to: b)
        let bc = b.meanAbsoluteDifference(to: c)
        let ac = a.meanAbsoluteDifference(to: c)
        guard ab > 0.025, bc > 0.025 else {
            return Result(isOutlier: false, score: 0, reason: "candidate close to a neighbor")
        }

        let detourRatio = (ab + bc) / max(ac, 0.012)
        let midpointError = b.midpointError(with: a, and: c)
        let neighborAgreement = min(ab, bc) / max(ac, 0.012)

        // Strongest case: the neighbors are nearly the same but the middle
        // frame is a flash/ghost. This is deliberately independent of scene
        // cut detection.
        let flashLike = ac < 0.055 && ab > 0.10 && bc > 0.10

        // For an ordinary moving sequence, A→B→C is approximately the direct
        // temporal path A→C. An inserted frame makes a large detour.
        let detourLike = detourRatio > 1.42 &&
            midpointError > 0.075 &&
            neighborAgreement > 0.34 &&
            ab > 0.055 &&
            bc > 0.055

        let score = min(max(
            0.55 * min(max((detourRatio - 1.0) / 1.0, 0), 1) +
            0.25 * min(max(midpointError / 0.16, 0), 1) +
            0.20 * min(max(neighborAgreement / 0.75, 0), 1),
            0
        ), 1)

        if flashLike {
            return Result(isOutlier: true, score: max(score, 0.92), reason: "isolated source-frame flash/ghost")
        }
        if detourLike {
            return Result(isOutlier: true, score: score, reason: "temporal source-frame outlier")
        }
        return Result(isOutlier: false, score: score, reason: "normal temporal path")
    }

    /// Builds a temporary bridge frame at the suspect frame's original time.
    /// RIFE receives this cleaned source sequence afterward, so it can perform
    /// the high-quality motion reconstruction rather than carrying the bad
    /// source frame into interpolation.
    func bridge(previous: CVPixelBuffer, next: CVPixelBuffer, progress: Double) -> CVPixelBuffer? {
        let width = min(CVPixelBufferGetWidth(previous), CVPixelBufferGetWidth(next))
        let height = min(CVPixelBufferGetHeight(previous), CVPixelBufferGetHeight(next))
        guard width > 0, height > 0,
              CVPixelBufferGetPixelFormatType(previous) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(next) == kCVPixelFormatType_32BGRA else { return nil }

        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        guard CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attrs as CFDictionary, &output
        ) == kCVReturnSuccess, let output else { return nil }

        let t = min(max(progress, 0), 1)
        CVPixelBufferLockBaseAddress(previous, .readOnly)
        CVPixelBufferLockBaseAddress(next, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(output, [])
            CVPixelBufferUnlockBaseAddress(next, .readOnly)
            CVPixelBufferUnlockBaseAddress(previous, .readOnly)
        }

        guard let pBase = CVPixelBufferGetBaseAddress(previous),
              let nBase = CVPixelBufferGetBaseAddress(next),
              let oBase = CVPixelBufferGetBaseAddress(output) else { return nil }

        let pRow = CVPixelBufferGetBytesPerRow(previous)
        let nRow = CVPixelBufferGetBytesPerRow(next)
        let oRow = CVPixelBufferGetBytesPerRow(output)

        for y in 0..<height {
            let p = pBase.advanced(by: y * pRow).assumingMemoryBound(to: UInt8.self)
            let n = nBase.advanced(by: y * nRow).assumingMemoryBound(to: UInt8.self)
            let o = oBase.advanced(by: y * oRow).assumingMemoryBound(to: UInt8.self)
            for x in 0..<(width * 4) {
                o[x] = UInt8(min(255, max(0, Int((1.0 - t) * Double(p[x]) + t * Double(n[x]) + 0.5))))
            }
        }
        CVBufferPropagateAttachments(previous, output)
        return output
    }

    private struct Sample {
        let width: Int
        let height: Int
        let values: [Float]

        init?(_ buffer: CVPixelBuffer) {
            guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { return nil }
            let width = CVPixelBufferGetWidth(buffer)
            let height = CVPixelBufferGetHeight(buffer)
            guard width >= 64, height >= 64 else { return nil }

            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            let p = base.assumingMemoryBound(to: UInt8.self)
            let xStep = max(1, width / 64)
            let yStep = max(1, height / 36)
            let xOffset = xStep / 2
            let yOffset = yStep / 2
            var values: [Float] = []
            values.reserveCapacity(64 * 36)

            for row in 0..<36 {
                let y = min(height - 1, yOffset + row * yStep)
                let source = p.advanced(by: y * rowBytes)
                for column in 0..<64 {
                    let x = min(width - 1, xOffset + column * xStep)
                    let i = x * 4
                    let b = Float(source[i]) / 255
                    let g = Float(source[i + 1]) / 255
                    let r = Float(source[i + 2]) / 255
                    values.append(0.0722 * b + 0.7152 * g + 0.2126 * r)
                }
            }
            self.width = 64
            self.height = 36
            self.values = values
        }

        func meanAbsoluteDifference(to other: Sample) -> Double {
            let n = min(values.count, other.values.count)
            guard n > 0 else { return 0 }
            var sum: Float = 0
            for i in 0..<n { sum += abs(values[i] - other.values[i]) }
            return Double(sum / Float(n))
        }

        func midpointError(with first: Sample, and last: Sample) -> Double {
            let n = min(values.count, min(first.values.count, last.values.count))
            guard n > 0 else { return 0 }
            var sum: Float = 0
            for i in 0..<n {
                let midpoint = (first.values[i] + last.values[i]) * 0.5
                sum += abs(values[i] - midpoint)
            }
            return Double(sum / Float(n))
        }
    }
}
