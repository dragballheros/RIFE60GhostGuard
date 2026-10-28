import CoreVideo
import Foundation

/// Reconstructs chroma from the clean luma instead of blurring the whole frame.
/// Flat anime cels are consolidated, speckles and bands are pulled toward the
/// edge-aware local color, and line art keeps its chroma because strong luma
/// edges reduce the mix. Skin tones are mixed less so they do not wash out.
/// Video frames blend stable pixels toward the previous rebuilt chroma.
final class ColorRebuild {
    private var previousLuma: [UInt8] = []
    private var previousCb: [UInt8] = []
    private var previousCr: [UInt8] = []
    private var previousWidth = 0
    private var previousHeight = 0
    private var hasHistory = false

    func apply(_ source: CVPixelBuffer, strength: Double) -> CVPixelBuffer {
        let amount = min(max(strength, 0), 1)
        guard amount > 0.001 else { return source }
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        guard width > 4, height > 4 else { return source }
        guard let destination = Self.makeBuffer(width: width, height: height) else { return source }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination) else { return source }
        let src = sourceBase.assumingMemoryBound(to: UInt8.self)
        let dst = destinationBase.assumingMemoryBound(to: UInt8.self)
        let srcRow = CVPixelBufferGetBytesPerRow(source)
        let dstRow = CVPixelBufferGetBytesPerRow(destination)
        let count = width * height
        if previousWidth != width || previousHeight != height || previousLuma.count != count {
            previousLuma = [UInt8](repeating: 0, count: count)
            previousCb = [UInt8](repeating: 128, count: count)
            previousCr = [UInt8](repeating: 128, count: count)
            previousWidth = width
            previousHeight = height
            hasHistory = false
        }

        let radius = amount > 0.7 ? 4 : 3
        var luma = [UInt8](repeating: 0, count: count)
        var cb = [Int16](repeating: 0, count: count)
        var cr = [Int16](repeating: 0, count: count)
        for y in 0..<height {
            let row = src.advanced(by: y * srcRow)
            let base = y * width
            for x in 0..<width {
                let p = row.advanced(by: x * 4)
                let b = Int(p[0]), g = Int(p[1]), r = Int(p[2])
                let y8 = UInt8(min(255, (77 * r + 150 * g + 29 * b) >> 8))
                luma[base + x] = y8
                cb[base + x] = Int16(((-43 * r - 85 * g + 128 * b) >> 8) + 128)
                cr[base + x] = Int16(((128 * r - 107 * g - 21 * b) >> 8) + 128)
            }
        }

        let guidedCb = Self.edgeAware(channel: cb, guide: luma, width: width, height: height, radius: radius)
        let guidedCr = Self.edgeAware(channel: cr, guide: luma, width: width, height: height, radius: radius)
        let hasPrevious = hasHistory

        for y in 0..<height {
            let row = src.advanced(by: y * srcRow)
            let out = dst.advanced(by: y * dstRow)
            let base = y * width
            for x in 0..<width {
                let index = base + x
                let y8 = Int(luma[index])
                let edge = Self.lumaEdge(luma, x: x, y: y, width: width, height: height)
                var mix = amount * (edge < 12 ? 0.82 : (edge < 28 ? 0.38 : 0.12))
                let cb0 = Int(cb[index])
                let cr0 = Int(cr[index])
                if Self.isSkin(y: y8, cb: cb0, cr: cr0) { mix *= 0.42 }
                var outCb = Int((Double(cb0) * (1 - mix) + Double(guidedCb[index]) * mix).rounded())
                var outCr = Int((Double(cr0) * (1 - mix) + Double(guidedCr[index]) * mix).rounded())
                if hasPrevious {
                    let motion = abs(y8 - Int(previousLuma[index]))
                    if motion < 8 {
                        let hold = 0.34
                        outCb = Int((Double(outCb) * (1 - hold) + Double(previousCb[index]) * hold).rounded())
                        outCr = Int((Double(outCr) * (1 - hold) + Double(previousCr[index]) * hold).rounded())
                    }
                }
                outCb = min(255, max(0, outCb))
                outCr = min(255, max(0, outCr))
                let p = row.advanced(by: x * 4)
                let rgb = Self.rgb(y: y8, cb: outCb, cr: outCr)
                let o = out.advanced(by: x * 4)
                o[0] = rgb.b
                o[1] = rgb.g
                o[2] = rgb.r
                o[3] = p[3]
                previousLuma[index] = UInt8(y8)
                previousCb[index] = UInt8(outCb)
                previousCr[index] = UInt8(outCr)
            }
        }
        hasHistory = true
        return destination
    }

    private static func isSkin(y: Int, cb: Int, cr: Int) -> Bool {
        y > 40 && y < 235 && cb > 85 && cb < 135 && cr > 140 && cr < 180
    }

    private static func lumaEdge(_ luma: [UInt8], x: Int, y: Int, width: Int, height: Int) -> Int {
        let cx = min(max(x, 1), width - 2)
        let cy = min(max(y, 1), height - 2)
        let c = Int(luma[cy * width + cx])
        let dx = abs(Int(luma[cy * width + cx + 1]) - Int(luma[cy * width + cx - 1]))
        let dy = abs(Int(luma[(cy + 1) * width + cx]) - Int(luma[(cy - 1) * width + cx]))
        return max(dx, dy, abs(c - c))
    }

    private static func edgeAware(channel: [Int16], guide: [UInt8], width: Int, height: Int, radius: Int) -> [Int16] {
        var horizontal = [Int16](repeating: 0, count: width * height)
        var output = [Int16](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var sum = 0
                var weight = 0
                let center = Int(guide[y * width + x])
                let x0 = max(0, x - radius)
                let x1 = min(width - 1, x + radius)
                for xx in x0...x1 {
                    let w = 64 / (1 + abs(center - Int(guide[y * width + xx])) / 3)
                    sum += Int(channel[y * width + xx]) * w
                    weight += w
                }
                horizontal[y * width + x] = Int16(sum / max(weight, 1))
            }
        }
        for y in 0..<height {
            let y0 = max(0, y - radius)
            let y1 = min(height - 1, y + radius)
            for x in 0..<width {
                var sum = 0
                var weight = 0
                let center = Int(guide[y * width + x])
                for yy in y0...y1 {
                    let w = 64 / (1 + abs(center - Int(guide[yy * width + x])) / 3)
                    sum += Int(horizontal[yy * width + x]) * w
                    weight += w
                }
                output[y * width + x] = Int16(sum / max(weight, 1))
            }
        }
        return output
    }

    private static func rgb(y: Int, cb: Int, cr: Int) -> (r: UInt8, g: UInt8, b: UInt8) {
        let c = cb - 128
        let d = cr - 128
        let r = min(255, max(0, y + ((359 * d) >> 8)))
        let g = min(255, max(0, y - ((88 * c + 183 * d) >> 8)))
        let b = min(255, max(0, y + ((454 * c) >> 8)))
        return (UInt8(r), UInt8(g), UInt8(b))
    }

    private static func makeBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &buffer) == kCVReturnSuccess else { return nil }
        return buffer
    }
}
