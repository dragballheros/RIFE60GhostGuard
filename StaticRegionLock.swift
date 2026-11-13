import CoreVideo
import Foundation

/// Pins background that did not move between the two sources. RIFE bends
/// static tile lines when something fast passes next to them. Where A and C
/// agree, including in a small neighborhood, the in-between is replaced with
/// their average. A solid foreground blob is left alone, and a detected cut
/// skips the pass.
final class StaticRegionLock {
    let sensitivity: Double

    init(sensitivity: Double) {
        let value = sensitivity.isFinite ? sensitivity : 1.35
        self.sensitivity = min(max(value, 0.65), 1.35)
    }

    func lock(previous: CVPixelBuffer, interpolated: CVPixelBuffer, next: CVPixelBuffer, progress: Double) -> CVPixelBuffer? {
        let format = CVPixelBufferGetPixelFormatType(previous)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA,
              CVPixelBufferGetPixelFormatType(interpolated) == format,
              CVPixelBufferGetPixelFormatType(next) == format else { return nil }

        let width = CVPixelBufferGetWidth(previous)
        let height = CVPixelBufferGetHeight(previous)
        guard width >= 8, height >= 8,
              CVPixelBufferGetWidth(interpolated) == width, CVPixelBufferGetWidth(next) == width,
              CVPixelBufferGetHeight(interpolated) == height, CVPixelBufferGetHeight(next) == height else { return nil }

        if let cut = StructuralSceneCut.assess(previous: previous, next: next, sensitivity: sensitivity, mid: interpolated),
           cut.isCut {
            return nil
        }

        let rgba = format == kCVPixelFormatType_32RGBA
        CVPixelBufferLockBaseAddress(previous, .readOnly)
        CVPixelBufferLockBaseAddress(interpolated, .readOnly)
        CVPixelBufferLockBaseAddress(next, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(next, .readOnly)
            CVPixelBufferUnlockBaseAddress(interpolated, .readOnly)
            CVPixelBufferUnlockBaseAddress(previous, .readOnly)
        }
        guard let previousBase = CVPixelBufferGetBaseAddress(previous),
              let midBase = CVPixelBufferGetBaseAddress(interpolated),
              let nextBase = CVPixelBufferGetBaseAddress(next) else { return nil }

        let previousRow = CVPixelBufferGetBytesPerRow(previous)
        let midRow = CVPixelBufferGetBytesPerRow(interpolated)
        let nextRow = CVPixelBufferGetBytesPerRow(next)
        let previousPixels = previousBase.assumingMemoryBound(to: UInt8.self)
        let midPixels = midBase.assumingMemoryBound(to: UInt8.self)
        let nextPixels = nextBase.assumingMemoryBound(to: UInt8.self)

        // 0.026 at sensitivity 1.35, about 0.030 at 0.65. The 5×5 mean is
        // tighter so one noisy pixel does not unlock a static line.
        let pixelLimit = Float(0.034 - 0.006 * sensitivity)
        let neighborhoodLimit = Float(0.026 - 0.004 * sensitivity)
        let count = width * height
        var metric = [Float](repeating: 0, count: count)
        var disagreement = [Float](repeating: 0, count: count)
        for y in 0..<height {
            let previousLine = previousPixels.advanced(by: y * previousRow)
            let nextLine = nextPixels.advanced(by: y * nextRow)
            let midLine = midPixels.advanced(by: y * midRow)
            for x in 0..<width {
                let pixel = x * 4
                let a = channels(previousLine, pixel, rgba: rgba)
                let b = channels(nextLine, pixel, rgba: rgba)
                let m = channels(midLine, pixel, rgba: rgba)
                let color = (abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b)) / 3
                metric[y * width + x] = max(abs(a.y - b.y), color)
                let midColor = (abs(m.r - a.r) + abs(m.g - a.g) + abs(m.b - a.b)) / 3
                disagreement[y * width + x] = max(abs(m.y - a.y), midColor) > 0.15 ? 1 : 0
            }
        }

        let localAgreement = boxMean(metric, width: width, height: height, radius: 2)
        let localChange = boxMean(disagreement, width: width, height: height, radius: 2)
        var mask = [UInt8](repeating: 0, count: count)
        var locked = 0
        for index in 0..<count {
            // A and C agree here and nearby, and M's difference is a thin
            // stroke rather than a solid body sitting between the two poses.
            if metric[index] < pixelLimit && localAgreement[index] < neighborhoodLimit && localChange[index] < 0.72 {
                mask[index] = 1
                locked += 1
            }
        }
        guard locked > 0 else { return nil }

        let coverage = boxMean(mask.map { Float($0) }, width: width, height: height, radius: 2)
        var output: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format,
                                  attributes as CFDictionary, &output) == kCVReturnSuccess,
              let output else { return nil }
        CVPixelBufferLockBaseAddress(output, [])
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        guard let outputBase = CVPixelBufferGetBaseAddress(output) else { return nil }
        let outputRow = CVPixelBufferGetBytesPerRow(output)
        let outputPixels = outputBase.assumingMemoryBound(to: UInt8.self)
        let nearerIsNext = progress > 0.5

        for y in 0..<height {
            let source = midPixels.advanced(by: y * midRow)
            let dest = outputPixels.advanced(by: y * outputRow)
            dest.update(from: source, count: width * 4)
            let previousLine = previousPixels.advanced(by: y * previousRow)
            let nextLine = nextPixels.advanced(by: y * nextRow)
            let nearerLine = nearerIsNext ? nextLine : previousLine
            for x in 0..<width {
                let index = y * width + x
                guard mask[index] != 0 else { continue }
                // Full strength inside the static area, about half within 2px
                // of a moving pixel so the seam is not a hard edge.
                let weight: Float = coverage[index] > 0.99 ? 1 : 0.45
                let keep = 1 - weight
                let pixel = x * 4
                for channel in 0..<3 {
                    dest[pixel + channel] = byte(keep * Float(dest[pixel + channel]) + weight * Float(nearerLine[pixel + channel]))
                }
            }
        }
        CVBufferPropagateAttachments(interpolated, output)
        return output
    }

    private func channels(_ row: UnsafePointer<UInt8>, _ pixel: Int, rgba: Bool) -> (r: Float, g: Float, b: Float, y: Float) {
        let c0 = Float(row[pixel]) / 255
        let c1 = Float(row[pixel + 1]) / 255
        let c2 = Float(row[pixel + 2]) / 255
        let r = rgba ? c0 : c2
        let b = rgba ? c2 : c0
        return (r, c1, b, 0.2126 * r + 0.7152 * c1 + 0.0722 * b)
    }

    private func byte(_ value: Float) -> UInt8 {
        UInt8(min(255, max(0, value + 0.5)))
    }

    private func boxMean(_ values: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        let stride = width + 1
        var integral = [Float](repeating: 0, count: stride * (height + 1))
        for y in 0..<height {
            var rowSum: Float = 0
            let current = (y + 1) * stride
            let above = y * stride
            let source = y * width
            for x in 0..<width {
                rowSum += values[source + x]
                integral[current + x + 1] = integral[above + x + 1] + rowSum
            }
        }
        var mean = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let y0 = max(0, y - radius)
            let y1 = min(height - 1, y + radius)
            for x in 0..<width {
                let x0 = max(0, x - radius)
                let x1 = min(width - 1, x + radius)
                let sum = integral[(y1 + 1) * stride + (x1 + 1)]
                    - integral[y0 * stride + (x1 + 1)]
                    - integral[(y1 + 1) * stride + x0]
                    + integral[y0 * stride + x0]
                let area = Float((x1 - x0 + 1) * (y1 - y0 + 1))
                mean[y * width + x] = sum / area
            }
        }
        return mean
    }
}
