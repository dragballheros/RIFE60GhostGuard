import Foundation
import CoreGraphics

/// Stroke coordinates are normalized in the upright image, never the screen.
struct WatermarkBrushPoint: Codable, Sendable, Equatable {
    var x: Double
    var y: Double
    var isValid: Bool { x.isFinite && y.isFinite && (0...1).contains(x) && (0...1).contains(y) }
}
struct WatermarkBrushStroke: Codable, Sendable, Equatable {
    var points: [WatermarkBrushPoint]
    var radiusX: Double
    var radiusY: Double
    var erase: Bool
    var isValid: Bool {
        !points.isEmpty && points.count <= 4096 && points.allSatisfy { $0.isValid } &&
        radiusX.isFinite && radiusY.isFinite && radiusX > 0 && radiusY > 0 && radiusX <= 1 && radiusY <= 1
    }
}
struct WatermarkBrushMask: Codable, Sendable, Equatable {
    // Legacy rectangles remain editable with the same brush/eraser operations.
    var rectangles: [WatermarkRegion] = []
    var strokes: [WatermarkBrushStroke] = []
    var isValid: Bool {
        rectangles.count <= 8 && rectangles.allSatisfy { $0.brush == nil && $0.isValid } &&
        strokes.count <= 256 && strokes.reduce(0) { $0 + $1.points.count } <= 32768 &&
        strokes.allSatisfy { $0.isValid } && (!rectangles.isEmpty || strokes.contains { !$0.erase })
    }
    var region: WatermarkRegion? {
        var bounds = CGRect.null
        for box in rectangles { bounds = bounds.union(CGRect(x: box.x, y: box.y, width: box.width, height: box.height)) }
        for stroke in strokes where !stroke.erase {
            for p in stroke.points {
                bounds = bounds.union(CGRect(x: p.x - stroke.radiusX, y: p.y - stroke.radiusY,
                                             width: stroke.radiusX * 2, height: stroke.radiusY * 2))
            }
        }
        bounds = bounds.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard isValid, !bounds.isNull, bounds.width > 0, bounds.height > 0 else { return nil }
        return WatermarkRegion(x: bounds.minX, y: bounds.minY, width: bounds.width, height: bounds.height, brush: self)
    }
    func mapped(points: (WatermarkBrushPoint) -> WatermarkBrushPoint,
                radii: (Double, Double) -> (Double, Double)) -> WatermarkBrushMask {
        var result = self
        result.strokes = strokes.map { stroke in
            let radius = radii(stroke.radiusX, stroke.radiusY)
            return WatermarkBrushStroke(points: stroke.points.map(points), radiusX: radius.0, radiusY: radius.1, erase: stroke.erase)
        }
        result.rectangles = rectangles.map { box in
            let corners = [WatermarkBrushPoint(x: box.x, y: box.y), WatermarkBrushPoint(x: box.x + box.width, y: box.y),
                           WatermarkBrushPoint(x: box.x, y: box.y + box.height), WatermarkBrushPoint(x: box.x + box.width, y: box.y + box.height)].map(points)
            let xs = corners.map(\.x), ys = corners.map(\.y)
            let left = xs.min() ?? 0, top = ys.min() ?? 0
            return WatermarkRegion(x: left, y: top, width: (xs.max() ?? left) - left, height: (ys.max() ?? top) - top)
        }
        return result
    }

    /// Rasterized once per source size. Ordered eraser strokes preserve holes.
    /// No antialiasing or rectangular fill is used for the painted strokes.
    func raster(width: Int, height: Int, padding: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: width * height)
        func limits(_ lower: Double, _ upper: Double, _ size: Int) -> Range<Int> {
            max(0, min(size, Int(floor(lower * Double(size)))))..<max(0, min(size, Int(ceil(upper * Double(size)))))
        }
        for box in rectangles {
            for y in limits(box.y, box.y + box.height, height) {
                for x in limits(box.x, box.x + box.width, width) { pixels[y * width + x] = 255 }
            }
        }
        for stroke in strokes {
            for index in stroke.points.indices {
                let a = stroke.points[index], b = stroke.points[max(0, index - 1)]
                let ax = a.x / stroke.radiusX, ay = a.y / stroke.radiusY
                let bx = b.x / stroke.radiusX, by = b.y / stroke.radiusY
                let dx = bx - ax, dy = by - ay, length = dx * dx + dy * dy
                for y in limits(min(a.y, b.y) - stroke.radiusY, max(a.y, b.y) + stroke.radiusY, height) {
                    for x in limits(min(a.x, b.x) - stroke.radiusX, max(a.x, b.x) + stroke.radiusX, width) {
                        let px = (Double(x) + 0.5) / Double(width) / stroke.radiusX - ax
                        let py = (Double(y) + 0.5) / Double(height) / stroke.radiusY - ay
                        let t = length > 0 ? min(max((px * dx + py * dy) / length, 0), 1) : 0
                        let ex = px - t * dx, ey = py - t * dy
                        if ex * ex + ey * ey <= 1 { pixels[y * width + x] = stroke.erase ? 0 : 255 }
                    }
                }
            }
        }
        let amount = min(max(padding, 0), 16)
        if amount > 0 {
            var horizontal = pixels
            for y in 0..<height {
                for x in 0..<width where pixels[y * width + x] != 0 {
                    for dx in max(0, x - amount)...min(width - 1, x + amount) { horizontal[y * width + dx] = 255 }
                }
            }
            var expanded = horizontal
            for y in 0..<height {
                for x in 0..<width where horizontal[y * width + x] != 0 {
                    for dy in max(0, y - amount)...min(height - 1, y + amount) { expanded[dy * width + x] = 255 }
                }
            }
            pixels = expanded
        }
        return pixels
    }
}
