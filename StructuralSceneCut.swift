import CoreVideo
import Foundation

/// Shot-boundary test for two frames that can share a palette. Coarse color
/// difference misses a dark-room or pink/beige cut, so this also compares
/// edge layout on a 64×36 luma grid. Callers OR the result into the existing
/// scene-cut path and then repeat a real frame instead of blending.
enum StructuralSceneCut {
    struct Assessment {
        var appearance: Double
        var overlap: Double?
        var blend: Double?
        /// Low edge overlap plus a modest color change. Nil overlap means both
        /// frames were too flat to judge structure.
        var structuralCut: Bool
        /// The original coarse MAD gate, `max(luma, RGB) > 0.20 / sensitivity`.
        var classicCut: Bool
        /// Mid frame sits on the A/C average while the shots do not share edges.
        var crossfadeCut: Bool

        var isCut: Bool { classicCut || structuralCut || crossfadeCut }
    }

    /// Dice overlap below this is a different layout. Higher sensitivity is stricter.
    /// 0.22 at 0.65, 0.30 at the default 1.35.
    static func overlapLimit(sensitivity: Double) -> Double {
        let s = clamped(sensitivity)
        return 0.22 + 0.08 * (s - 0.65) / 0.70
    }

    /// Cuts with almost no appearance change are left to the existing rules.
    /// 0.058 at 0.65, 0.050 at the default 1.35.
    static func appearanceFloor(sensitivity: Double) -> Double {
        let s = clamped(sensitivity)
        return 0.058 - 0.008 * (s - 0.65) / 0.70
    }

    static func assess(previous: CVPixelBuffer, next: CVPixelBuffer, sensitivity: Double,
                       mid: CVPixelBuffer? = nil) -> Assessment? {
        let format = CVPixelBufferGetPixelFormatType(previous)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA,
              CVPixelBufferGetPixelFormatType(next) == format else { return nil }
        if let mid, CVPixelBufferGetPixelFormatType(mid) != format { return nil }

        var width = min(CVPixelBufferGetWidth(previous), CVPixelBufferGetWidth(next))
        var height = min(CVPixelBufferGetHeight(previous), CVPixelBufferGetHeight(next))
        if let mid {
            width = min(width, CVPixelBufferGetWidth(mid))
            height = min(height, CVPixelBufferGetHeight(mid))
        }
        guard width >= 8, height >= 8 else { return nil }

        CVPixelBufferLockBaseAddress(previous, .readOnly)
        CVPixelBufferLockBaseAddress(next, .readOnly)
        if let mid { CVPixelBufferLockBaseAddress(mid, .readOnly) }
        defer {
            if let mid { CVPixelBufferUnlockBaseAddress(mid, .readOnly) }
            CVPixelBufferUnlockBaseAddress(next, .readOnly)
            CVPixelBufferUnlockBaseAddress(previous, .readOnly)
        }
        guard let previousBase = CVPixelBufferGetBaseAddress(previous),
              let nextBase = CVPixelBufferGetBaseAddress(next) else { return nil }
        let midBase = mid.flatMap { CVPixelBufferGetBaseAddress($0) }

        return assess(
            previous: previousBase.assumingMemoryBound(to: UInt8.self),
            next: nextBase.assumingMemoryBound(to: UInt8.self),
            width: width,
            height: height,
            previousRow: CVPixelBufferGetBytesPerRow(previous),
            nextRow: CVPixelBufferGetBytesPerRow(next),
            rgba: format == kCVPixelFormatType_32RGBA,
            sensitivity: sensitivity,
            mid: midBase?.assumingMemoryBound(to: UInt8.self),
            midRow: mid.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
        )
    }

    static func assess(previous: UnsafePointer<UInt8>,
                       next: UnsafePointer<UInt8>,
                       width: Int,
                       height: Int,
                       previousRow: Int,
                       nextRow: Int,
                       rgba: Bool,
                       sensitivity: Double,
                       mid: UnsafePointer<UInt8>? = nil,
                       midRow: Int = 0) -> Assessment {
        let columns = min(64, width)
        let rows = min(36, height)
        let count = columns * rows
        var lumaA = [Float](repeating: 0, count: count)
        var lumaB = [Float](repeating: 0, count: count)
        var lumaSum: Float = 0
        var colorSum: Float = 0
        var blendSum: Float = 0

        for row in 0..<rows {
            let y = min(height - 1, (row * height + height / 2) / rows)
            let previousLine = previous.advanced(by: y * previousRow)
            let nextLine = next.advanced(by: y * nextRow)
            let midLine = mid.map { $0.advanced(by: y * midRow) }
            for column in 0..<columns {
                let x = min(width - 1, (column * width + width / 2) / columns)
                let pixel = x * 4
                let a = sample(previousLine, pixel, rgba: rgba)
                let b = sample(nextLine, pixel, rgba: rgba)
                let index = row * columns + column
                lumaA[index] = a.y
                lumaB[index] = b.y
                lumaSum += abs(a.y - b.y)
                colorSum += (abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b)) / 3
                if let midLine {
                    let m = sample(midLine, pixel, rgba: rgba)
                    blendSum += abs(m.y - (a.y + b.y) * 0.5)
                }
            }
        }

        let samples = Float(max(count, 1))
        let appearance = Double(max(lumaSum, colorSum) / samples)
        let blend = mid == nil ? nil : Double(blendSum / samples)
        let tuned = clamped(sensitivity)
        let classicCut = appearance > 0.20 / tuned

        let edges = edgeMasks(lumaA, lumaB, columns: columns, rows: rows)
        let overlap: Double?
        if edges.fractionA < 0.012 && edges.fractionB < 0.012 {
            overlap = nil
        } else if edges.dilatedA + edges.dilatedB == 0 {
            overlap = nil
        } else {
            // Symmetric Dice after a 1px dilation. 2px left the missed
            // same-palette cuts at 0.28-0.41; at 1px those fall to 0.15-0.28
            // while ordinary motion stays near 1.
            overlap = Double(2 * edges.intersection) / Double(edges.dilatedA + edges.dilatedB)
        }

        let floor = appearanceFloor(sensitivity: tuned)
        let structuralCut = overlap.map { $0 < overlapLimit(sensitivity: tuned) && appearance > floor } ?? false
        let crossfadeCut: Bool
        if let overlap, let blend {
            let crossfadeOverlap = min(0.42, overlapLimit(sensitivity: tuned) + 0.08)
            // An endpoint frame sits about half of |A-C| off the average.
            // A real cross-fade sits on it. 0.28 * appearance stays under that half.
            crossfadeCut = appearance > floor && overlap < crossfadeOverlap && blend < max(0.018, appearance * 0.28)
        } else {
            crossfadeCut = false
        }

        return Assessment(
            appearance: appearance,
            overlap: overlap,
            blend: blend,
            structuralCut: structuralCut,
            classicCut: classicCut,
            crossfadeCut: crossfadeCut
        )
    }

    private static func clamped(_ sensitivity: Double) -> Double {
        min(max(sensitivity.isFinite ? sensitivity : 1.35, 0.65), 1.35)
    }

    private struct Sample {
        var r: Float
        var g: Float
        var b: Float
        var y: Float
    }

    private static func sample(_ row: UnsafePointer<UInt8>, _ pixel: Int, rgba: Bool) -> Sample {
        let c0 = Float(row[pixel]) / 255
        let c1 = Float(row[pixel + 1]) / 255
        let c2 = Float(row[pixel + 2]) / 255
        let r = rgba ? c0 : c2
        let b = rgba ? c2 : c0
        return Sample(r: r, g: c1, b: b, y: 0.2126 * r + 0.7152 * c1 + 0.0722 * b)
    }

    private struct Edges {
        var fractionA: Double
        var fractionB: Double
        var dilatedA: Int
        var dilatedB: Int
        var intersection: Int
    }

    private static func edgeMasks(_ lumaA: [Float], _ lumaB: [Float], columns: Int, rows: Int) -> Edges {
        let count = columns * rows
        var maskA = [UInt8](repeating: 0, count: count)
        var maskB = [UInt8](repeating: 0, count: count)
        var edgesA = 0
        var edgesB = 0
        if columns > 2, rows > 2 {
            for y in 1..<(rows - 1) {
                for x in 1..<(columns - 1) {
                    let index = y * columns + x
                    let magA = abs(lumaA[index + 1] - lumaA[index - 1]) + abs(lumaA[index + columns] - lumaA[index - columns])
                    let magB = abs(lumaB[index + 1] - lumaB[index - 1]) + abs(lumaB[index + columns] - lumaB[index - columns])
                    // A real contour, not a flat-field ripple.
                    if magA >= 0.12 {
                        maskA[index] = 1
                        edgesA += 1
                    }
                    if magB >= 0.12 {
                        maskB[index] = 1
                        edgesB += 1
                    }
                }
            }
        }
        dilate(&maskA, columns: columns, rows: rows, radius: 1)
        dilate(&maskB, columns: columns, rows: rows, radius: 1)
        var dilatedA = 0
        var dilatedB = 0
        var intersection = 0
        for index in 0..<count {
            let a = maskA[index] != 0
            let b = maskB[index] != 0
            if a { dilatedA += 1 }
            if b { dilatedB += 1 }
            if a && b { intersection += 1 }
        }
        return Edges(
            fractionA: Double(edgesA) / Double(max(count, 1)),
            fractionB: Double(edgesB) / Double(max(count, 1)),
            dilatedA: dilatedA,
            dilatedB: dilatedB,
            intersection: intersection
        )
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
