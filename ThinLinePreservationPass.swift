import CoreVideo
import Foundation

/// Restores thin dark strokes that both real frames still share when an
/// in-between fades them out. Hair wisps are a handful of pixels, so the
/// whole-frame guards never see the flicker. Moving lines are left to RIFE:
/// a stroke qualifies only when it sits in the same place in both sources.
final class ThinLinePreservationPass {
    let sensitivity: Double

    init(sensitivity: Double) {
        let value = sensitivity.isFinite ? sensitivity : 1.35
        self.sensitivity = min(max(value, 0.65), 1.35)
    }

    /// Returns a new buffer with only the lost strokes repainted, or nil when
    /// the frame should be kept as-is (cut, sanity bounds, or nothing lost).
    func preserve(previous: CVPixelBuffer,
                  interpolated: CVPixelBuffer,
                  next: CVPixelBuffer,
                  progress: Double) -> CVPixelBuffer? {
        let format = CVPixelBufferGetPixelFormatType(previous)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA,
              CVPixelBufferGetPixelFormatType(interpolated) == format,
              CVPixelBufferGetPixelFormatType(next) == format else { return nil }

        let width = CVPixelBufferGetWidth(previous)
        let height = CVPixelBufferGetHeight(previous)
        guard width >= 8, height >= 8,
              CVPixelBufferGetWidth(interpolated) == width,
              CVPixelBufferGetWidth(next) == width,
              CVPixelBufferGetHeight(interpolated) == height,
              CVPixelBufferGetHeight(next) == height else { return nil }

        let rgba = format == kCVPixelFormatType_32RGBA
        let t = min(max(progress.isFinite ? progress : 0.5, 0), 1)

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

        // Same gate GhostGuard uses for a shot change. A cross-fade here would
        // paint one scene's line art onto the other.
        if discontinuity(previousPixels, nextPixels, width: width, height: height,
                         previousRow: previousRow, nextRow: nextRow, rgba: rgba) > 0.20 / sensitivity {
            return nil
        }

        let previousLuma = lumaPlane(previousPixels, width: width, height: height, rowBytes: previousRow, rgba: rgba)
        let nextLuma = lumaPlane(nextPixels, width: width, height: height, rowBytes: nextRow, rgba: rgba)
        let midLuma = lumaPlane(midPixels, width: width, height: height, rowBytes: midRow, rgba: rgba)
        let previousBlur = boxBlur(previousLuma, width: width, height: height, radius: 2)
        let nextBlur = boxBlur(nextLuma, width: width, height: height, radius: 2)

        // Black top-hat on a 5×5 window: a thin dark stroke is darker than its
        // neighborhood, while the inside of a large dark region is not.
        let lineContrast = Float(max(0.055, 0.12 / sensitivity))
        let previousLines = lineMask(previousLuma, blur: previousBlur, width: width, height: height, contrast: lineContrast)
        var nextLines = lineMask(nextLuma, blur: nextBlur, width: width, height: height, contrast: lineContrast)
        dilate3x3(&nextLines, width: width, height: height)

        var agreement = 0
        let count = width * height
        for index in 0..<count where previousLines[index] != 0 && nextLines[index] != 0 {
            agreement += 1
        }
        let fraction = Double(agreement) / Double(count)
        // A speckle is noise. A mask covering a large share of the frame is not
        // a set of thin strokes, so leave that frame alone.
        if agreement < 12 || fraction > 0.06 { return nil }

        let fadeGap = Float(max(0.045, 0.10 / sensitivity))
        var hits: [Hit] = []
        hits.reserveCapacity(agreement)
        let preferPrevious = t <= 0.5
        for y in 2..<(height - 2) {
            let previousRowPixels = previousPixels.advanced(by: y * previousRow)
            let nextRowPixels = nextPixels.advanced(by: y * nextRow)
            for x in 2..<(width - 2) {
                let index = y * width + x
                guard previousLines[index] != 0, nextLines[index] != 0 else { continue }
                let yA = previousLuma[index]
                let yB = nextLuma[index]
                let yM = midLuma[index]
                let darker = min(yA, yB)
                let average = (yA + yB) * 0.5
                // Lost or faded: the in-between is lighter than both the darker
                // source stroke and the average of the two.
                guard yM > average + fadeGap, yM > darker + fadeGap else { continue }

                let pixel = x * 4
                let nearY = preferPrevious ? yA : yB
                let near = preferPrevious ? previousRowPixels : nextRowPixels
                let dark = yA <= yB ? previousRowPixels : nextRowPixels
                // The nearer frame is right when its stroke is still on this
                // pixel. A 1px tremble puts the stroke on the other side.
                let source = nearY <= darker + 0.04 ? near : dark
                hits.append(Hit(x: x, y: y, b: source[pixel], g: source[pixel + 1], r: source[pixel + 2]))
            }
        }
        guard !hits.isEmpty else { return nil }

        var weight = [Float](repeating: 0, count: count)
        var blue = [UInt8](repeating: 0, count: count)
        var green = [UInt8](repeating: 0, count: count)
        var red = [UInt8](repeating: 0, count: count)
        for hit in hits {
            for dy in -2...2 {
                let yy = hit.y + dy
                guard yy >= 0, yy < height else { continue }
                for dx in -2...2 {
                    let xx = hit.x + dx
                    guard xx >= 0, xx < width else { continue }
                    let distance = max(abs(dx), abs(dy))
                    let sampleWeight: Float = distance == 0 ? 1 : (distance == 1 ? 0.58 : 0.26)
                    let dest = yy * width + xx
                    if sampleWeight > weight[dest] {
                        weight[dest] = sampleWeight
                        blue[dest] = hit.b
                        green[dest] = hit.g
                        red[dest] = hit.r
                    }
                }
            }
        }

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

        for y in 0..<height {
            let source = midPixels.advanced(by: y * midRow)
            let dest = outputPixels.advanced(by: y * outputRow)
            dest.update(from: source, count: width * 4)
            let rowIndex = y * width
            for x in 0..<width {
                let sampleWeight = weight[rowIndex + x]
                if sampleWeight <= 0 { continue }
                let pixel = x * 4
                let keep = 1 - sampleWeight
                let repaired = rowIndex + x
                dest[pixel] = byte(keep * Float(dest[pixel]) + sampleWeight * Float(blue[repaired]))
                dest[pixel + 1] = byte(keep * Float(dest[pixel + 1]) + sampleWeight * Float(green[repaired]))
                dest[pixel + 2] = byte(keep * Float(dest[pixel + 2]) + sampleWeight * Float(red[repaired]))
            }
        }
        CVBufferPropagateAttachments(interpolated, output)
        return output
    }

    private struct Hit {
        var x: Int
        var y: Int
        var b: UInt8
        var g: UInt8
        var r: UInt8
    }

    private func byte(_ value: Float) -> UInt8 {
        UInt8(min(255, max(0, value + 0.5)))
    }

    private func luma(b: Float, g: Float, r: Float) -> Float {
        0.0722 * b + 0.7152 * g + 0.2126 * r
    }

    private func lumaPlane(_ pixels: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int, rgba: Bool) -> [Float] {
        var plane = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = pixels.advanced(by: y * rowBytes)
            for x in 0..<width {
                let pixel = x * 4
                let c0 = Float(row[pixel]) / 255
                let c1 = Float(row[pixel + 1]) / 255
                let c2 = Float(row[pixel + 2]) / 255
                let r = rgba ? c0 : c2
                let b = rgba ? c2 : c0
                plane[y * width + x] = luma(b: b, g: c1, r: r)
            }
        }
        return plane
    }

    private func lineMask(_ luma: [Float], blur: [Float], width: Int, height: Int, contrast: Float) -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: width * height)
        guard width > 4, height > 4 else { return mask }
        for y in 2..<(height - 2) {
            for x in 2..<(width - 2) {
                let index = y * width + x
                let value = luma[index]
                // Absolute cap keeps mid-tone texture from counting as a black stroke.
                if value < 0.58 && blur[index] - value >= contrast {
                    mask[index] = 1
                }
            }
        }
        return mask
    }

    private func dilate3x3(_ mask: inout [UInt8], width: Int, height: Int) {
        var dilated = mask
        for y in 0..<height {
            for x in 0..<width where mask[y * width + x] != 0 {
                for dy in -1...1 {
                    let yy = y + dy
                    guard yy >= 0, yy < height else { continue }
                    for dx in -1...1 {
                        let xx = x + dx
                        guard xx >= 0, xx < width else { continue }
                        dilated[yy * width + xx] = 1
                    }
                }
            }
        }
        mask = dilated
    }

    private func boxBlur(_ luma: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        let stride = width + 1
        var integral = [Float](repeating: 0, count: stride * (height + 1))
        for y in 0..<height {
            var rowSum: Float = 0
            let source = y * width
            let current = (y + 1) * stride
            let above = y * stride
            for x in 0..<width {
                rowSum += luma[source + x]
                integral[current + x + 1] = integral[above + x + 1] + rowSum
            }
        }
        var blurred = [Float](repeating: 0, count: width * height)
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
                blurred[y * width + x] = sum / area
            }
        }
        return blurred
    }

    private func discontinuity(_ previous: UnsafePointer<UInt8>,
                               _ next: UnsafePointer<UInt8>,
                               width: Int,
                               height: Int,
                               previousRow: Int,
                               nextRow: Int,
                               rgba: Bool) -> Double {
        let step = max(1, max(width, height) / 160)
        var lumaSum: Float = 0
        var colorSum: Float = 0
        var samples: Float = 0
        var y = 0
        while y < height {
            let previousRowPixels = previous.advanced(by: y * previousRow)
            let nextRowPixels = next.advanced(by: y * nextRow)
            var x = 0
            while x < width {
                let pixel = x * 4
                let p0 = Float(previousRowPixels[pixel]) / 255
                let p1 = Float(previousRowPixels[pixel + 1]) / 255
                let p2 = Float(previousRowPixels[pixel + 2]) / 255
                let n0 = Float(nextRowPixels[pixel]) / 255
                let n1 = Float(nextRowPixels[pixel + 1]) / 255
                let n2 = Float(nextRowPixels[pixel + 2]) / 255
                let pr = rgba ? p0 : p2
                let pb = rgba ? p2 : p0
                let nr = rgba ? n0 : n2
                let nb = rgba ? n2 : n0
                lumaSum += abs(luma(b: pb, g: p1, r: pr) - luma(b: nb, g: n1, r: nr))
                colorSum += (abs(p0 - n0) + abs(p1 - n1) + abs(p2 - n2)) / 3
                samples += 1
                x += step
            }
            y += step
        }
        guard samples > 0 else { return 0 }
        return Double(max(lumaSum, colorSum) / samples)
    }
}
