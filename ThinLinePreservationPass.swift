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

        // Same combined cut signal as GhostGuard, including different shots
        // that share a palette. Restoring lines across that boundary would
        // paint one scene onto the other.
        let cut = StructuralSceneCut.assess(
            previous: previousPixels,
            next: nextPixels,
            width: width,
            height: height,
            previousRow: previousRow,
            nextRow: nextRow,
            rgba: rgba,
            sensitivity: sensitivity,
            mid: midPixels,
            midRow: midRow
        )
        if cut.isCut { return nil }

        let previousLuma = lumaPlane(previousPixels, width: width, height: height, rowBytes: previousRow, rgba: rgba)
        let nextLuma = lumaPlane(nextPixels, width: width, height: height, rowBytes: nextRow, rgba: rgba)
        let midLuma = lumaPlane(midPixels, width: width, height: height, rowBytes: midRow, rgba: rgba)
        let previousBlur = boxBlur(previousLuma, width: width, height: height, radius: 2)
        let nextBlur = boxBlur(nextLuma, width: width, height: height, radius: 2)

        // Only restore strokes that occupy the same pixels in both source frames.
        // Nearby-but-shifted lines belong to motion interpolation, not source pinning.
        let lineContrast = Float(max(0.055, 0.12 / sensitivity))
        let previousLines = lineMask(previousLuma, blur: previousBlur, width: width, height: height, contrast: lineContrast)
        let nextLines = lineMask(nextLuma, blur: nextBlur, width: width, height: height, contrast: lineContrast)
        let tFloat = Float(t)
        let count = width * height
        var agreement = 0
        var previousLineCount = 0
        var nextLineCount = 0
        for index in 0..<count {
            let inPrevious = previousLines[index] != 0
            let inNext = nextLines[index] != 0
            if inPrevious { previousLineCount += 1 }
            if inNext { nextLineCount += 1 }
            if inPrevious && inNext { agreement += 1 }
        }
        let fraction = Double(agreement) / Double(count)
        if agreement < 12 || fraction > 0.06 { return nil }
        let previousOverlap = Double(agreement) / Double(previousLineCount)
        let nextOverlap = Double(agreement) / Double(nextLineCount)
        // Exact-coordinate overlap is the static-line fast path. For moving
        // contours, try a bounded, reciprocal local correspondence repair rather
        // than pinning a moving stroke to either source endpoint.
        if min(previousOverlap, nextOverlap) < 0.92 {
            return preserveMotionCompensatedContours(
                previous: previousPixels, interpolated: midPixels, next: nextPixels,
                attachmentsFrom: interpolated,
                previousLuma: previousLuma, midLuma: midLuma, nextLuma: nextLuma,
                previousLines: previousLines, nextLines: nextLines,
                width: width, height: height,
                previousRow: previousRow, midRow: midRow, nextRow: nextRow,
                format: format, progress: t
            )
        }

        let fadeGap = Float(max(0.045, 0.10 / sensitivity))
        var hits: [Hit] = []
        hits.reserveCapacity(agreement)
        for y in 2..<(height - 2) {
            let previousRowPixels = previousPixels.advanced(by: y * previousRow)
            let nextRowPixels = nextPixels.advanced(by: y * nextRow)
            for x in 2..<(width - 2) {
                let index = y * width + x
                // The exact same source pixel must be a line in both frames.
                guard previousLines[index] != 0 && nextLines[index] != 0 else { continue }
                let yS = previousLuma[index] * (1 - tFloat) + nextLuma[index] * tFloat
                let yM = midLuma[index]
                guard yM > yS + fadeGap else { continue }
                let pixel = x * 4
                let b = byte(Float(previousRowPixels[pixel]) * (1 - tFloat) + Float(nextRowPixels[pixel]) * tFloat)
                let g = byte(Float(previousRowPixels[pixel + 1]) * (1 - tFloat) + Float(nextRowPixels[pixel + 1]) * tFloat)
                let r = byte(Float(previousRowPixels[pixel + 2]) * (1 - tFloat) + Float(nextRowPixels[pixel + 2]) * tFloat)
                hits.append(Hit(x: x, y: y, b: b, g: g, r: r, luma: yS))
            }
        }
        guard !hits.isEmpty else { return nil }

        var weight = [Float](repeating: 0, count: count)
        var blue = [UInt8](repeating: 0, count: count)
        var green = [UInt8](repeating: 0, count: count)
        var red = [UInt8](repeating: 0, count: count)
        var floorLuma = [Float](repeating: 0, count: count)
        for hit in hits {
            let index = hit.y * width + hit.x
            weight[index] = 1
            blue[index] = hit.b
            green[index] = hit.g
            red[index] = hit.r
            floorLuma[index] = hit.luma
            // Alpha feather only, one pixel. The color is the source pixel at
            // the neighbor, not the stroke color, so the line cannot grow.
            for dy in -1...1 {
                for dx in -1...1 where dx != 0 || dy != 0 {
                    let xx = hit.x + dx
                    let yy = hit.y + dy
                    guard xx >= 0, yy >= 0, xx < width, yy < height else { continue }
                    let dest = yy * width + xx
                    guard weight[dest] < 0.35 else { continue }
                    let previousRowPixels = previousPixels.advanced(by: yy * previousRow)
                    let nextRowPixels = nextPixels.advanced(by: yy * nextRow)
                    let pixel = xx * 4
                    weight[dest] = 0.35
                    blue[dest] = byte(Float(previousRowPixels[pixel]) * (1 - tFloat) + Float(nextRowPixels[pixel]) * tFloat)
                    green[dest] = byte(Float(previousRowPixels[pixel + 1]) * (1 - tFloat) + Float(nextRowPixels[pixel + 1]) * tFloat)
                    red[dest] = byte(Float(previousRowPixels[pixel + 2]) * (1 - tFloat) + Float(nextRowPixels[pixel + 2]) * tFloat)
                    floorLuma[dest] = previousLuma[dest] * (1 - tFloat) + nextLuma[dest] * tFloat
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
                var b = keep * Float(dest[pixel]) + sampleWeight * Float(blue[repaired])
                var g = keep * Float(dest[pixel + 1]) + sampleWeight * Float(green[repaired])
                var r = keep * Float(dest[pixel + 2]) + sampleWeight * Float(red[repaired])
                // Never darker than the time-weighted source pixel at this location.
                if luma(b: b / 255, g: g / 255, r: r / 255) < floorLuma[repaired] {
                    b = Float(blue[repaired])
                    g = Float(green[repaired])
                    r = Float(red[repaired])
                }
                dest[pixel] = byte(b)
                dest[pixel + 1] = byte(g)
                dest[pixel + 2] = byte(r)
            }
        }
        CVBufferPropagateAttachments(interpolated, output)
        return output
    }

    /// Repairs only source-supported thin contours whose local position changes
    /// between endpoints. A reciprocal 3x3 patch match rejects ambiguous matches;
    /// a strict candidate-count cap keeps this exceptional path bounded on 4K frames.
    private func preserveMotionCompensatedContours(
        previous: UnsafePointer<UInt8>, interpolated: UnsafePointer<UInt8>, next: UnsafePointer<UInt8>,
        attachmentsFrom: CVPixelBuffer,
        previousLuma: [Float], midLuma: [Float], nextLuma: [Float],
        previousLines: [UInt8], nextLines: [UInt8],
        width: Int, height: Int,
        previousRow: Int, midRow: Int, nextRow: Int,
        format: OSType, progress: Double
    ) -> CVPixelBuffer? {
        let previousLineCount = previousLines.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        guard previousLineCount >= 12, previousLineCount <= 60_000 else { return nil }

        let radius = 4
        let t = Float(min(max(progress, 0), 1))
        let fadeGap = Float(max(0.045, 0.10 / sensitivity))

        func patchCost(_ a: [Float], _ b: [Float], ax: Int, ay: Int, bx: Int, by: Int) -> Float {
            var sum: Float = 0
            for oy in -2...2 {
                for ox in -2...2 {
                    let ai = (ay + oy) * width + ax + ox
                    let bi = (by + oy) * width + bx + ox
                    sum += abs(a[ai] - b[bi])
                }
            }
            return sum / 25
        }

        func bestMatch(
            x: Int, y: Int,
            sourceLuma: [Float], targetLuma: [Float], targetMask: [UInt8]
        ) -> (x: Int, y: Int, cost: Float)? {
            var best: (x: Int, y: Int, cost: Float)?
            for dy in -radius...radius {
                let yy = y + dy
                guard yy >= 2, yy < height - 2 else { continue }
                for dx in -radius...radius {
                    let xx = x + dx
                    guard xx >= 2, xx < width - 2,
                          targetMask[yy * width + xx] != 0 else { continue }
                    let cost = patchCost(sourceLuma, targetLuma, ax: x, ay: y, bx: xx, by: yy)
                    let distance = dx * dx + dy * dy
                    let bestDistance = best.map { ($0.x - x) * ($0.x - x) + ($0.y - y) * ($0.y - y) } ?? Int.max
                    if best == nil || cost < best!.cost - 0.0001
                        || (abs(cost - best!.cost) <= 0.0001 && distance < bestDistance) {
                        best = (xx, yy, cost)
                    }
                }
            }
            guard let best, best.cost <= 0.14 else { return nil }
            return best
        }

        var pairs: [(Int, Int, Int, Int)] = []
        pairs.reserveCapacity(previousLineCount)
        for y in 2..<(height - 2) {
            for x in 2..<(width - 2) {
                let sourceIndex = y * width + x
                guard previousLines[sourceIndex] != 0 else { continue }
                guard let match = bestMatch(
                    x: x, y: y, sourceLuma: previousLuma,
                    targetLuma: nextLuma, targetMask: nextLines
                ) else { continue }
                guard let reverse = bestMatch(
                    x: match.x, y: match.y, sourceLuma: nextLuma,
                    targetLuma: previousLuma, targetMask: previousLines
                ), reverse.x == x, reverse.y == y else { continue }
                pairs.append((x, y, match.x, match.y))
            }
        }
        guard pairs.count >= 12 else { return nil }

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
        let out = outputBase.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            out.advanced(by: y * outputRow).update(from: interpolated.advanced(by: y * midRow), count: width * 4)
        }

        var repaired = 0
        for (px, py, nx, ny) in pairs {
            let tx = Int((Float(px) * (1 - t) + Float(nx) * t).rounded())
            let ty = Int((Float(py) * (1 - t) + Float(ny) * t).rounded())
            guard tx >= 1, ty >= 1, tx < width - 1, ty < height - 1 else { continue }
            let targetIndex = ty * width + tx
            let expectedLuma = previousLuma[py * width + px] * (1 - t)
                + nextLuma[ny * width + nx] * t
            guard midLuma[targetIndex] > expectedLuma + fadeGap else { continue }

            let p = previous.advanced(by: py * previousRow + px * 4)
            let n = next.advanced(by: ny * nextRow + nx * 4)
            let d = out.advanced(by: ty * outputRow + tx * 4)
            for channel in 0..<3 {
                d[channel] = UInt8(min(255, max(0,
                    (Float(p[channel]) * (1 - t) + Float(n[channel]) * t).rounded()
                )))
            }
            repaired += 1
        }
        guard repaired >= 6 else { return nil }
        CVBufferPropagateAttachments(attachmentsFrom, output)
        return output
    }

    private struct Hit {
        var x: Int
        var y: Int
        var b: UInt8
        var g: UInt8
        var r: UInt8
        var luma: Float
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
}
