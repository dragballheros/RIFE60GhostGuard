import CoreVideo
import Foundation

/// Keeps thin line art crisp during a slow pan or a small local tremble.
/// RIFE's in-between is left alone when the motion is large or the match is poor.
/// A detected cut returns nil.
final class SmallMotionLineStabilizer {
    let sensitivity: Double

    init(sensitivity: Double) {
        let value = sensitivity.isFinite ? sensitivity : 1.35
        self.sensitivity = min(max(value, 0.65), 1.35)
    }

    func stabilize(previous: CVPixelBuffer, interpolated: CVPixelBuffer, next: CVPixelBuffer, progress: Double) -> CVPixelBuffer? {
        let format = CVPixelBufferGetPixelFormatType(previous)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA,
              CVPixelBufferGetPixelFormatType(interpolated) == format,
              CVPixelBufferGetPixelFormatType(next) == format else { return nil }
        let width = CVPixelBufferGetWidth(previous)
        let height = CVPixelBufferGetHeight(previous)
        guard width >= 32, height >= 32,
              CVPixelBufferGetWidth(interpolated) == width, CVPixelBufferGetWidth(next) == width,
              CVPixelBufferGetHeight(interpolated) == height, CVPixelBufferGetHeight(next) == height else { return nil }
        if let cut = StructuralSceneCut.assess(previous: previous, next: next, sensitivity: sensitivity, mid: interpolated),
           cut.isCut {
            return nil
        }

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

        // Higher sensitivity accepts a slightly worse residual and fainter lines.
        let wholeFrameLimit = Float(0.012 + 0.004 * (sensitivity - 0.65) / 0.70)
        let localLimit = Float(0.042 + 0.020 * (sensitivity - 0.65) / 0.70)
        let lineContrast = Float(max(0.055, 0.11 / sensitivity))

        let shift = globalShift(previousPixels, nextPixels, width: width, height: height,
                                previousRow: previousRow, nextRow: nextRow, rgba: rgba)
        let nearerIsNext = t > 0.5
        let nearerPixels = nearerIsNext ? nextPixels : previousPixels
        let nearerRow = nearerIsNext ? nextRow : previousRow
        // A is at shift 0, C is at `shift`. The nearer frame is moved onto time t.
        let placeScale = Float(nearerIsNext ? (t - 1) : t)
        let placeX = placeScale * Float(shift.dx)
        let placeY = placeScale * Float(shift.dy)

        if shift.explained {
            return composite(midPixels, midRow: midRow, source: nearerPixels, sourceRow: nearerRow,
                             width: width, height: height, rgba: rgba, format: format,
                             offsetX: placeX, offsetY: placeY,
                             linesOnly: shift.residual >= wholeFrameLimit,
                             lineContrast: lineContrast)
        }

        return localTiles(previousPixels, midPixels, nextPixels,
                          width: width, height: height,
                          previousRow: previousRow, midRow: midRow, nextRow: nextRow,
                          rgba: rgba, format: format, nearerIsNext: nearerIsNext,
                          lineContrast: lineContrast, localLimit: localLimit)
    }

    private struct Shift {
        var dx: Int
        var dy: Int
        var residual: Float
        var explained: Bool
    }

    /// Coarse-to-fine SAD. Half-resolution keeps the darkest sample in each
    /// 2×2 so a 1px stroke is still visible, then ±2px is checked at full size.
    private func globalShift(_ a: UnsafePointer<UInt8>, _ c: UnsafePointer<UInt8>,
                             width: Int, height: Int, previousRow: Int, nextRow: Int, rgba: Bool) -> Shift {
        let halfW = width / 2
        let halfH = height / 2
        guard halfW >= 8, halfH >= 8 else { return Shift(dx: 0, dy: 0, residual: 1, explained: false) }
        var halfA = [Float](repeating: 1, count: halfW * halfH)
        var halfC = [Float](repeating: 1, count: halfW * halfH)
        for y in 0..<height {
            let aLine = a.advanced(by: y * previousRow)
            let cLine = c.advanced(by: y * nextRow)
            let hy = min(halfH - 1, y / 2)
            for x in 0..<width {
                let hx = min(halfW - 1, x / 2)
                let index = hy * halfW + hx
                halfA[index] = min(halfA[index], luma(aLine, x * 4, rgba: rgba))
                halfC[index] = min(halfC[index], luma(cLine, x * 4, rgba: rgba))
            }
        }

        var best = (dx: 0, dy: 0, sad: Float.greatestFiniteMagnitude)
        for sy in -6...6 {
            for sx in -6...6 {
                let score = halfSAD(halfA, halfC, width: halfW, height: halfH, dx: sx, dy: sy)
                if score < best.sad { best = (sx * 2, sy * 2, score) }
            }
        }

        var refined = (dx: best.dx, dy: best.dy, sad: Float.greatestFiniteMagnitude)
        for dy in -2...2 {
            for dx in -2...2 {
                let sx = min(12, max(-12, best.dx + dx))
                let sy = min(12, max(-12, best.dy + dy))
                let score = fullSAD(a, c, width: width, height: height, aRow: previousRow, cRow: nextRow,
                                    rgba: rgba, dx: sx, dy: sy)
                if score < refined.sad { refined = (sx, sy, score) }
            }
        }
        let zero = fullSAD(a, c, width: width, height: height, aRow: previousRow, cRow: nextRow,
                           rgba: rgba, dx: 0, dy: 0)
        let sadLimit = Float(0.034 + 0.014 * (sensitivity - 0.65) / 0.70)
        let moved = refined.dx != 0 || refined.dy != 0
        let explained = moved && abs(refined.dx) <= 12 && abs(refined.dy) <= 12
            && refined.sad < sadLimit && refined.sad < zero * 0.55
        return Shift(dx: refined.dx, dy: refined.dy, residual: refined.sad, explained: explained)
    }

    private func halfSAD(_ a: [Float], _ c: [Float], width: Int, height: Int, dx: Int, dy: Int) -> Float {
        var sum: Float = 0
        var n: Float = 0
        for y in 0..<height {
            let yy = y + dy
            if yy < 0 || yy >= height { continue }
            for x in 0..<width {
                let xx = x + dx
                if xx < 0 || xx >= width { continue }
                sum += abs(a[y * width + x] - c[yy * width + xx])
                n += 1
            }
        }
        return n > 0 ? sum / n : 1
    }

    private func fullSAD(_ a: UnsafePointer<UInt8>, _ c: UnsafePointer<UInt8>,
                         width: Int, height: Int, aRow: Int, cRow: Int, rgba: Bool, dx: Int, dy: Int) -> Float {
        var sum: Float = 0
        var n: Float = 0
        var y = 0
        while y < height {
            let yy = y + dy
            if yy >= 0 && yy < height {
                let aLine = a.advanced(by: y * aRow)
                let cLine = c.advanced(by: yy * cRow)
                var x = 0
                while x < width {
                    let xx = x + dx
                    if xx >= 0 && xx < width {
                        sum += abs(luma(aLine, x * 4, rgba: rgba) - luma(cLine, xx * 4, rgba: rgba))
                        n += 1
                    }
                    x += 2
                }
            }
            y += 2
        }
        return n > 0 ? sum / n : 1
    }

    private func composite(_ mid: UnsafePointer<UInt8>, midRow: Int,
                           source: UnsafePointer<UInt8>, sourceRow: Int,
                           width: Int, height: Int, rgba: Bool, format: OSType,
                           offsetX: Float, offsetY: Float, linesOnly: Bool,
                           lineContrast: Float) -> CVPixelBuffer? {
        let sourceLuma = lumaPlane(source, width: width, height: height, rowBytes: sourceRow, rgba: rgba)
        let lines = lineMask(sourceLuma, width: width, height: height, contrast: lineContrast)
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
        var wrote = false
        for y in 0..<height {
            let midLine = mid.advanced(by: y * midRow)
            let dest = outputPixels.advanced(by: y * outputRow)
            dest.update(from: midLine, count: width * 4)
            for x in 0..<width {
                let sampleX = Float(x) - offsetX
                let sampleY = Float(y) - offsetY
                let nearestX = Int(sampleX.rounded())
                let nearestY = Int(sampleY.rounded())
                let onLine = nearestX >= 0 && nearestY >= 0 && nearestX < width && nearestY < height
                    && lines[nearestY * width + nearestX] != 0
                if linesOnly && !onLine && !nearLine(lines, x: nearestX, y: nearestY, width: width, height: height) {
                    continue
                }
                guard var color = bilinear(source, rowBytes: sourceRow, width: width, height: height,
                                           x: sampleX, y: sampleY, rgba: rgba) else { continue }
                if onLine {
                    color = sharpen(color, source: source, rowBytes: sourceRow, width: width, height: height,
                                    x: nearestX, y: nearestY, rgba: rgba)
                }
                let weight: Float = linesOnly ? (onLine ? 1 : 0.45) : 1
                let pixel = x * 4
                let keep = 1 - weight
                dest[pixel] = byte(keep * Float(dest[pixel]) + weight * color.0)
                dest[pixel + 1] = byte(keep * Float(dest[pixel + 1]) + weight * color.1)
                dest[pixel + 2] = byte(keep * Float(dest[pixel + 2]) + weight * color.2)
                wrote = true
            }
        }
        guard wrote else { return nil }
        CVBufferPropagateAttachments(mid, output)
        return output
    }

    /// 32×32 tiles. A tile is skipped when its lines moved by more than the
    /// ±3px search, or when the best match is no better than leaving M alone.
    private func localTiles(_ a: UnsafePointer<UInt8>, _ m: UnsafePointer<UInt8>, _ c: UnsafePointer<UInt8>,
                            width: Int, height: Int, previousRow: Int, midRow: Int, nextRow: Int,
                            rgba: Bool, format: OSType, nearerIsNext: Bool,
                            lineContrast: Float, localLimit: Float) -> CVPixelBuffer? {
        let lumaA = lumaPlane(a, width: width, height: height, rowBytes: previousRow, rgba: rgba)
        let lumaC = lumaPlane(c, width: width, height: height, rowBytes: nextRow, rgba: rgba)
        let lumaM = lumaPlane(m, width: width, height: height, rowBytes: midRow, rgba: rgba)
        let linesA = lineMask(lumaA, width: width, height: height, contrast: lineContrast)
        let linesC = lineMask(lumaC, width: width, height: height, contrast: lineContrast)
        let nearer = nearerIsNext ? c : a
        let nearerRow = nearerIsNext ? nextRow : previousRow
        let nearerLines = nearerIsNext ? linesC : linesA
        var weight = [Float](repeating: 0, count: width * height)
        var blue = [Float](repeating: 0, count: width * height)
        var green = [Float](repeating: 0, count: width * height)
        var red = [Float](repeating: 0, count: width * height)
        var any = false

        let tile = 32
        var ty = 0
        while ty < height {
            var tx = 0
            while tx < width {
                let x1 = min(width, tx + tile)
                let y1 = min(height, ty + tile)
                if !largeTileMotion(linesA, linesC, width: width, x0: tx, y0: ty, x1: x1, y1: y1),
                   let match = tileMatch(lumaM, lumaA, lumaC, nearerLines,
                                         width: width, x0: tx, y0: ty, x1: x1, y1: y1,
                                         nearerIsNext: nearerIsNext, localLimit: localLimit) {
                    any = true
                    for y in y0..<y1 {
                        for x in x0..<x1 {
                            let sourceX = x + match.dx
                            let sourceY = y + match.dy
                            guard sourceX >= 0, sourceY >= 0, sourceX < width, sourceY < height else { continue }
                            let sourceIndex = sourceY * width + sourceX
                            guard nearerLines[sourceIndex] != 0 else { continue }
                            let color = pixel(nearer, rowBytes: nearerRow, x: sourceX, y: sourceY)
                            splat(x: x, y: y, width: width, height: height, color: color,
                                  weight: &weight, blue: &blue, green: &green, red: &red)
                        }
                    }
                }
                tx += tile
            }
            ty += tile
        }
        guard any else { return nil }
        return blendWeights(m, midRow: midRow, width: width, height: height, format: format,
                            weight: weight, blue: blue, green: green, red: red)
    }

    private struct Match {
        var dx: Int
        var dy: Int
    }

    private func largeTileMotion(_ linesA: [UInt8], _ linesC: [UInt8], width: Int,
                                 x0: Int, y0: Int, x1: Int, y1: Int) -> Bool {
        var countA = 0
        var covered = 0
        for y in y0..<y1 {
            for x in x0..<x1 where linesA[y * width + x] != 0 {
                countA += 1
                if lineNearby(linesC, width: width, height: linesC.count / width, x: x, y: y, radius: 3) {
                    covered += 1
                }
            }
        }
        return countA >= 6 && Double(covered) / Double(countA) < 0.5
    }

    private func tileMatch(_ lumaM: [Float], _ lumaA: [Float], _ lumaC: [Float], _ nearerLines: [UInt8],
                           width: Int, x0: Int, y0: Int, x1: Int, y1: Int,
                           nearerIsNext: Bool, localLimit: Float) -> Match? {
        let nearer = nearerIsNext ? lumaC : lumaA
        let zero = tileSAD(lumaM, nearer, width: width, x0: x0, y0: y0, x1: x1, y1: y1, dx: 0, dy: 0)
        var best = (dx: 0, dy: 0, sad: zero)
        for dy in -3...3 {
            for dx in -3...3 where dx != 0 || dy != 0 {
                let score = tileSAD(lumaM, nearer, width: width, x0: x0, y0: y0, x1: x1, y1: y1, dx: dx, dy: dy)
                if score < best.sad { best = (dx, dy, score) }
            }
        }
        let hasLine = tileHasLine(nearerLines, width: width, x0: x0, y0: y0, x1: x1, y1: y1)
        guard hasLine else { return nil }
        let improved = best.sad < localLimit && best.sad < zero * 0.65 && (best.dx != 0 || best.dy != 0)
        let soft = best.dx == 0 && best.dy == 0 && best.sad < localLimit
            && lineSoftness(lumaM, nearer, nearerLines, width: width, x0: x0, y0: y0, x1: x1, y1: y1) > 0.06
        return (improved || soft) ? Match(dx: best.dx, dy: best.dy) : nil
    }

    private func tileSAD(_ a: [Float], _ b: [Float], width: Int, x0: Int, y0: Int, x1: Int, y1: Int, dx: Int, dy: Int) -> Float {
        var sum: Float = 0
        var n: Float = 0
        let height = a.count / width
        for y in y0..<y1 {
            let yy = y + dy
            if yy < 0 || yy >= height { continue }
            for x in x0..<x1 {
                let xx = x + dx
                if xx < 0 || xx >= width { continue }
                sum += abs(a[y * width + x] - b[yy * width + xx])
                n += 1
            }
        }
        return n > 0 ? sum / n : 1
    }

    private func tileHasLine(_ lines: [UInt8], width: Int, x0: Int, y0: Int, x1: Int, y1: Int) -> Bool {
        var count = 0
        for y in y0..<y1 {
            for x in x0..<x1 where lines[y * width + x] != 0 {
                count += 1
                if count >= 8 { return true }
            }
        }
        return false
    }

    private func lineSoftness(_ mid: [Float], _ source: [Float], _ lines: [UInt8],
                              width: Int, x0: Int, y0: Int, x1: Int, y1: Int) -> Float {
        var sum: Float = 0
        var n: Float = 0
        for y in y0..<y1 {
            for x in x0..<x1 where lines[y * width + x] != 0 {
                sum += mid[y * width + x] - source[y * width + x]
                n += 1
            }
        }
        return n > 0 ? sum / n : 0
    }

    private func lineNearby(_ lines: [UInt8], width: Int, height: Int, x: Int, y: Int, radius: Int) -> Bool {
        for dy in -radius...radius {
            let yy = y + dy
            if yy < 0 || yy >= height { continue }
            for dx in -radius...radius {
                let xx = x + dx
                if xx < 0 || xx >= width { continue }
                if lines[yy * width + xx] != 0 { return true }
            }
        }
        return false
    }

    private func splat(x: Int, y: Int, width: Int, height: Int, color: (Float, Float, Float),
                       weight: inout [Float], blue: inout [Float], green: inout [Float], red: inout [Float]) {
        for dy in -2...2 {
            let yy = y + dy
            guard yy >= 0, yy < height else { continue }
            for dx in -2...2 {
                let xx = x + dx
                guard xx >= 0, xx < width else { continue }
                let distance = max(abs(dx), abs(dy))
                let sampleWeight: Float = distance == 0 ? 1 : (distance == 1 ? 0.55 : 0.28)
                let index = yy * width + xx
                if sampleWeight > weight[index] {
                    weight[index] = sampleWeight
                    blue[index] = color.0
                    green[index] = color.1
                    red[index] = color.2
                }
            }
        }
    }

    private func blendWeights(_ mid: UnsafePointer<UInt8>, midRow: Int, width: Int, height: Int, format: OSType,
                              weight: [Float], blue: [Float], green: [Float], red: [Float]) -> CVPixelBuffer? {
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
        guard let base = CVPixelBufferGetBaseAddress(output) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(output)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            let source = mid.advanced(by: y * midRow)
            let dest = pixels.advanced(by: y * rowBytes)
            dest.update(from: source, count: width * 4)
            for x in 0..<width {
                let index = y * width + x
                let sampleWeight = weight[index]
                if sampleWeight <= 0 { continue }
                let keep = 1 - sampleWeight
                let pixel = x * 4
                dest[pixel] = byte(keep * Float(dest[pixel]) + sampleWeight * blue[index])
                dest[pixel + 1] = byte(keep * Float(dest[pixel + 1]) + sampleWeight * green[index])
                dest[pixel + 2] = byte(keep * Float(dest[pixel + 2]) + sampleWeight * red[index])
            }
        }
        CVBufferPropagateAttachments(mid, output)
        return output
    }

    private func bilinear(_ pixels: UnsafePointer<UInt8>, rowBytes: Int, width: Int, height: Int,
                          x: Float, y: Float, rgba: Bool) -> (Float, Float, Float)? {
        let snappedX = abs(x - x.rounded()) < 0.08 ? x.rounded() : x
        let snappedY = abs(y - y.rounded()) < 0.08 ? y.rounded() : y
        let x0 = Int(floor(snappedX))
        let y0 = Int(floor(snappedY))
        let x1 = x0 + 1
        let y1 = y0 + 1
        guard x0 >= 0, y0 >= 0, x1 < width, y1 < height else {
            guard x0 >= 0, y0 >= 0, x0 < width, y0 < height else { return nil }
            return pixel(pixels, rowBytes: rowBytes, x: x0, y: y0)
        }
        let tx = snappedX - Float(x0)
        let ty = snappedY - Float(y0)
        let p00 = pixel(pixels, rowBytes: rowBytes, x: x0, y: y0)
        let p10 = pixel(pixels, rowBytes: rowBytes, x: x1, y: y0)
        let p01 = pixel(pixels, rowBytes: rowBytes, x: x0, y: y1)
        let p11 = pixel(pixels, rowBytes: rowBytes, x: x1, y: y1)
        func mix(_ a: Float, _ b: Float, _ c: Float, _ d: Float) -> Float {
            let top = a + (b - a) * tx
            let bottom = c + (d - c) * tx
            return top + (bottom - top) * ty
        }
        var color = (mix(p00.0, p10.0, p01.0, p11.0), mix(p00.1, p10.1, p01.1, p11.1), mix(p00.2, p10.2, p01.2, p11.2))
        _ = rgba
        return color
    }

    /// Pull a half-pixel sample toward the darkest source tap so a stroke
    /// does not alternate soft and sharp. Integer samples are unchanged.
    private func sharpen(_ color: (Float, Float, Float), source: UnsafePointer<UInt8>, rowBytes: Int,
                         width: Int, height: Int, x: Int, y: Int, rgba: Bool) -> (Float, Float, Float) {
        var darkest = color
        var darkLuma = lumaValue(color)
        for dy in -1...1 {
            let yy = y + dy
            guard yy >= 0, yy < height else { continue }
            for dx in -1...1 {
                let xx = x + dx
                guard xx >= 0, xx < width else { continue }
                let sample = pixel(source, rowBytes: rowBytes, x: xx, y: yy)
                let sampleLuma = lumaValue(sample)
                if sampleLuma < darkLuma {
                    darkLuma = sampleLuma
                    darkest = sample
                }
            }
        }
        guard lumaValue(color) > darkLuma + 0.05, darkLuma < 0.45 else { return color }
        let pull: Float = 0.40
        return (
            color.0 + (darkest.0 - color.0) * pull,
            color.1 + (darkest.1 - color.1) * pull,
            color.2 + (darkest.2 - color.2) * pull
        )
    }

    private func nearLine(_ lines: [UInt8], x: Int, y: Int, width: Int, height: Int) -> Bool {
        guard x >= -2, y >= -2, x <= width + 1, y <= height + 1 else { return false }
        for dy in -2...2 {
            let yy = y + dy
            guard yy >= 0, yy < height else { continue }
            for dx in -2...2 {
                let xx = x + dx
                guard xx >= 0, xx < width else { continue }
                if lines[yy * width + xx] != 0 { return true }
            }
        }
        return false
    }

    private func lumaPlane(_ pixels: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int, rgba: Bool) -> [Float] {
        var plane = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = pixels.advanced(by: y * rowBytes)
            for x in 0..<width {
                plane[y * width + x] = luma(row, x * 4, rgba: rgba)
            }
        }
        return plane
    }

    private func lineMask(_ luma: [Float], width: Int, height: Int, contrast: Float) -> [UInt8] {
        let blur = boxMean(luma, width: width, height: height, radius: 2)
        var mask = [UInt8](repeating: 0, count: width * height)
        guard width > 4, height > 4 else { return mask }
        for y in 2..<(height - 2) {
            for x in 2..<(width - 2) {
                let index = y * width + x
                let value = luma[index]
                if value < 0.55 && blur[index] - value >= contrast {
                    mask[index] = 1
                }
            }
        }
        return mask
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
                mean[y * width + x] = sum / Float((x1 - x0 + 1) * (y1 - y0 + 1))
            }
        }
        return mean
    }

    private func pixel(_ pixels: UnsafePointer<UInt8>, rowBytes: Int, x: Int, y: Int) -> (Float, Float, Float) {
        let p = pixels.advanced(by: y * rowBytes + x * 4)
        return (Float(p[0]), Float(p[1]), Float(p[2]))
    }

    private func luma(_ row: UnsafePointer<UInt8>, _ pixel: Int, rgba: Bool) -> Float {
        let c0 = Float(row[pixel]) / 255
        let c1 = Float(row[pixel + 1]) / 255
        let c2 = Float(row[pixel + 2]) / 255
        let r = rgba ? c0 : c2
        let b = rgba ? c2 : c0
        return 0.2126 * r + 0.7152 * c1 + 0.0722 * b
    }

    private func lumaValue(_ color: (Float, Float, Float)) -> Float {
        // Stored channels are buffer order. BGRA luma uses B=0, G=1, R=2.
        (0.0722 * color.0 + 0.7152 * color.1 + 0.2126 * color.2) / 255
    }

    private func byte(_ value: Float) -> UInt8 {
        UInt8(min(255, max(0, value.rounded())))
    }
}
