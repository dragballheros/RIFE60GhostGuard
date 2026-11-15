import CoreVideo
import Foundation

/// Rejects an in-between whose outlines are not the sources' outlines.
/// A wavy or doubled stroke produces edge pixels that sit on neither frame.
/// Legitimate motion is covered by dilating the union of the source edges by
/// half the largest source-edge shift, capped, so a line that simply travels
/// from A to C is not novel.
enum WarpedOutlineGuard {
    static func rejects(previous: CVPixelBuffer, generated: CVPixelBuffer, next: CVPixelBuffer, sensitivity: Double) -> Bool {
        let format = CVPixelBufferGetPixelFormatType(previous)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA,
              CVPixelBufferGetPixelFormatType(generated) == format,
              CVPixelBufferGetPixelFormatType(next) == format else { return false }

        var width = min(CVPixelBufferGetWidth(previous), CVPixelBufferGetWidth(generated))
        width = min(width, CVPixelBufferGetWidth(next))
        var height = min(CVPixelBufferGetHeight(previous), CVPixelBufferGetHeight(generated))
        height = min(height, CVPixelBufferGetHeight(next))
        guard width >= 16, height >= 16 else { return false }

        CVPixelBufferLockBaseAddress(previous, .readOnly)
        CVPixelBufferLockBaseAddress(generated, .readOnly)
        CVPixelBufferLockBaseAddress(next, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(next, .readOnly)
            CVPixelBufferUnlockBaseAddress(generated, .readOnly)
            CVPixelBufferUnlockBaseAddress(previous, .readOnly)
        }
        guard let previousBase = CVPixelBufferGetBaseAddress(previous),
              let generatedBase = CVPixelBufferGetBaseAddress(generated),
              let nextBase = CVPixelBufferGetBaseAddress(next) else { return false }

        let step = max(2, max(width, height) / 480)
        let columns = (width - 1) / step + 1
        let rows = (height - 1) / step + 1
        guard columns >= 8, rows >= 8 else { return false }

        let rgba = format == kCVPixelFormatType_32RGBA
        let lumaA = plane(previousBase.assumingMemoryBound(to: UInt8.self), CVPixelBufferGetBytesPerRow(previous),
                          width: width, height: height, step: step, columns: columns, rows: rows, rgba: rgba)
        let lumaM = plane(generatedBase.assumingMemoryBound(to: UInt8.self), CVPixelBufferGetBytesPerRow(generated),
                          width: width, height: height, step: step, columns: columns, rows: rows, rgba: rgba)
        let lumaC = plane(nextBase.assumingMemoryBound(to: UInt8.self), CVPixelBufferGetBytesPerRow(next),
                          width: width, height: height, step: step, columns: columns, rows: rows, rgba: rgba)

        let edgesA = edges(lumaA, columns: columns, rows: rows)
        let edgesM = edges(lumaM, columns: columns, rows: rows)
        let edgesC = edges(lumaC, columns: columns, rows: rows)
        let countA = edgesA.reduce(0) { $0 + Int($1) }
        let countM = edgesM.reduce(0) { $0 + Int($1) }
        let countC = edgesC.reduce(0) { $0 + Int($1) }
        guard countM >= 24 else { return false }

        let tuned = min(max(sensitivity.isFinite ? sensitivity : 1.35, 0.65), 1.35)
        // 1.50 at sensitivity 1.35, 1.75 at 0.65. A and C must already agree
        // with each other, so a busier moving shot is not called a double.
        let doubleRatio = 1.50 + 0.25 * (1.35 - tuned) / 0.70
        let edgeMax = max(countA, countC)
        let edgeMin = min(countA, countC)
        let sourcesAgree = edgeMax >= 24 && edgeMin * 3 >= edgeMax * 2
        if sourcesAgree && Double(countM) > Double(edgeMax) * doubleRatio {
            return true
        }

        let shift = maxEdgeShift(from: edgesA, to: edgesC, columns: columns, rows: rows, search: 16)
        // Half the shift, plus one sample, covers the in-between position
        // and the gradient sitting beside the stroke. Never under 2px, never over 8.
        let radius = min(8, max(2, (shift + 3) / 2))
        var allowed = edgesA
        for index in 0..<allowed.count where edgesC[index] != 0 {
            allowed[index] = 1
        }
        dilate(&allowed, columns: columns, rows: rows, radius: radius)

        var novel = 0
        for index in 0..<edgesM.count where edgesM[index] != 0 && allowed[index] == 0 {
            novel += 1
        }
        // 0.25 at sensitivity 1.35, 0.35 at 0.65.
        let novelLimit = 0.25 + 0.10 * (1.35 - tuned) / 0.70
        return Double(novel) / Double(countM) > novelLimit
    }

    private static func plane(_ pixels: UnsafePointer<UInt8>, _ rowBytes: Int,
                              width: Int, height: Int, step: Int, columns: Int, rows: Int, rgba: Bool) -> [Float] {
        var luma = [Float](repeating: 0, count: columns * rows)
        for row in 0..<rows {
            let y = min(height - 1, row * step)
            let line = pixels.advanced(by: y * rowBytes)
            for column in 0..<columns {
                let x = min(width - 1, column * step)
                let pixel = x * 4
                let c0 = Float(line[pixel]) / 255
                let c1 = Float(line[pixel + 1]) / 255
                let c2 = Float(line[pixel + 2]) / 255
                let r = rgba ? c0 : c2
                let b = rgba ? c2 : c0
                luma[row * columns + column] = 0.2126 * r + 0.7152 * c1 + 0.0722 * b
            }
        }
        return luma
    }

    private static func edges(_ luma: [Float], columns: Int, rows: Int) -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: columns * rows)
        guard columns > 2, rows > 2 else { return mask }
        for y in 1..<(rows - 1) {
            for x in 1..<(columns - 1) {
                let index = y * columns + x
                let magnitude = abs(luma[index + 1] - luma[index - 1]) + abs(luma[index + columns] - luma[index - columns])
                if magnitude >= 0.12 { mask[index] = 1 }
            }
        }
        return mask
    }

    private static func maxEdgeShift(from source: [UInt8], to other: [UInt8], columns: Int, rows: Int, search: Int) -> Int {
        var farthest = 0
        for y in 0..<rows {
            for x in 0..<columns where source[y * columns + x] != 0 {
                var found = search
                if other[y * columns + x] != 0 {
                    found = 0
                } else {
                    ring: for radius in 1...search {
                        for dy in -radius...radius {
                            let yy = y + dy
                            if yy < 0 || yy >= rows { continue }
                            for dx in -radius...radius where max(abs(dx), abs(dy)) == radius {
                                let xx = x + dx
                                if xx < 0 || xx >= columns { continue }
                                if other[yy * columns + xx] != 0 {
                                    found = radius
                                    break ring
                                }
                            }
                        }
                    }
                }
                if found > farthest { farthest = found }
            }
        }
        return farthest
    }

    private static func dilate(_ mask: inout [UInt8], columns: Int, rows: Int, radius: Int) {
        var dilated = mask
        for y in 0..<rows {
            for x in 0..<columns where mask[y * columns + x] != 0 {
                for dy in -radius...radius {
                    let yy = y + dy
                    guard yy >= 0, yy < rows else { continue }
                    for dx in -radius...radius {
                        let xx = x + dx
                        guard xx >= 0, xx < columns else { continue }
                        dilated[yy * columns + xx] = 1
                    }
                }
            }
        }
        mask = dilated
    }
}
