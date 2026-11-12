import Foundation

/// Low-contrast hatch drawn on a warm pink skin patch, such as cheek blush.
/// Strong dark outlines fail the contrast cap, and a stroke is ignored unless
/// both flanks are the same kind of pink, so hair, eyes, and ordinary line art
/// stay on the existing cleanup path.
enum BlushStrokeGuard {
    private struct Sample {
        var r: Int
        var g: Int
        var b: Int
        var y: Int
        var cb: Int
        var cr: Int
    }

    /// The stroke itself, plus the warm-pink pixels touching it. Those neighbors
    /// are where a guided color mix or a Sharpie dash would repaint the hatch edge.
    static func protects(
        pixels: UnsafePointer<UInt8>,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        rowBytes: Int,
        bgra: Bool
    ) -> Bool {
        if isStroke(pixels, x: x, y: y, width: width, height: height, rowBytes: rowBytes, bgra: bgra) {
            return true
        }
        let center = load(pixels, x: x, y: y, rowBytes: rowBytes, bgra: bgra)
        if !warmPink(center) { return false }
        if lumaRange(pixels, x: x, y: y, width: width, height: height, rowBytes: rowBytes, bgra: bgra) < 10 {
            return false
        }
        for dy in -1...1 {
            for dx in -1...1 where dx != 0 || dy != 0 {
                if isStroke(pixels, x: x + dx, y: y + dy, width: width, height: height, rowBytes: rowBytes, bgra: bgra) {
                    return true
                }
            }
        }
        return false
    }

    private static func isStroke(
        _ pixels: UnsafePointer<UInt8>,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        rowBytes: Int,
        bgra: Bool
    ) -> Bool {
        if x < 2 || y < 2 || x >= width - 2 || y >= height - 2 { return false }
        let center = load(pixels, x: x, y: y, rowBytes: rowBytes, bgra: bgra)
        if center.y < 80 || center.y > 205 { return false }
        if center.r < center.b + 6 { return false }
        if center.cr < 132 { return false }
        let axes = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for (dx, dy) in axes {
            for dist in [1, 2] {
                let ax = x - dx * dist
                let ay = y - dy * dist
                let bx = x + dx * dist
                let by = y + dy * dist
                if ax < 0 || ay < 0 || bx < 0 || by < 0 || ax >= width || ay >= height || bx >= width || by >= height {
                    continue
                }
                let flankA = load(pixels, x: ax, y: ay, rowBytes: rowBytes, bgra: bgra)
                let flankB = load(pixels, x: bx, y: by, rowBytes: rowBytes, bgra: bgra)
                if !warmPink(flankA) || !warmPink(flankB) { continue }
                if abs(flankA.y - flankB.y) > 14 { continue }
                let darker = min(flankA.y - center.y, flankB.y - center.y)
                if darker >= 8 && darker <= 32 { return true }
            }
        }
        return false
    }

    /// Mid-luma warm skin. Black line art and bright highlights fail this on purpose.
    private static func warmPink(_ sample: Sample) -> Bool {
        sample.y >= 105 && sample.y <= 225
            && sample.cr >= 148 && sample.cr <= 205
            && sample.cb >= 96 && sample.cb <= 142
            && sample.r >= sample.b + 12
            && sample.r + 6 >= sample.g
    }

    private static func lumaRange(
        _ pixels: UnsafePointer<UInt8>,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        rowBytes: Int,
        bgra: Bool
    ) -> Int {
        var low = 255
        var high = 0
        for dy in -1...1 {
            let sy = y + dy
            if sy < 0 || sy >= height { continue }
            for dx in -1...1 {
                let sx = x + dx
                if sx < 0 || sx >= width { continue }
                let luma = load(pixels, x: sx, y: sy, rowBytes: rowBytes, bgra: bgra).y
                low = min(low, luma)
                high = max(high, luma)
            }
        }
        return high - low
    }

    private static func load(
        _ pixels: UnsafePointer<UInt8>,
        x: Int,
        y: Int,
        rowBytes: Int,
        bgra: Bool
    ) -> Sample {
        let pixel = pixels.advanced(by: y * rowBytes + x * 4)
        let blue = Int(bgra ? pixel[0] : pixel[2])
        let green = Int(pixel[1])
        let red = Int(bgra ? pixel[2] : pixel[0])
        let luma = (77 * red + 150 * green + 29 * blue) >> 8
        let cb = ((-43 * red - 85 * green + 128 * blue) >> 8) + 128
        let cr = ((128 * red - 107 * green - 21 * blue) >> 8) + 128
        return Sample(r: red, g: green, b: blue, y: luma, cb: cb, cr: cr)
    }
}
