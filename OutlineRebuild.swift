import CoreVideo
import Foundation
import UIKit

/// Repairs broken line art before Sharpie. Sharpie only emphasizes lines that
/// already exist. This pass closes short gaps, restores faded line darkness
/// without shifting hue, and lifts isolated specks back toward the local paper.
final class OutlineRebuild {
    func apply(_ source: CVPixelBuffer, strength: Double) -> CVPixelBuffer {
        let amount = min(max(strength, 0), 1)
        guard amount > 0.001 else { return source }
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        guard width > 4, height > 4 else { return source }
        guard let destination = ColorRebuildBuffer.make(width: width, height: height) else { return source }
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
        let luma = Self.lumaPlane(src: src, width: width, height: height, srcRow: srcRow)
        let gap = amount > 0.66 ? 2 : 1
        let line = Self.lineMask(luma: luma, width: width, height: height, gap: gap)
        let closed = Self.closedMask(line: line, luma: luma, width: width, height: height, gap: gap)
        Self.paint(
            src: src, dst: dst, luma: luma, line: line, closed: closed,
            width: width, height: height, srcRow: srcRow, dstRow: dstRow, amount: amount
        )
        return destination
    }

    private static func lumaPlane(src: UnsafeMutablePointer<UInt8>, width: Int, height: Int, srcRow: Int) -> [UInt8] {
        var luma = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = src.advanced(by: y * srcRow)
            let base = y * width
            for x in 0..<width {
                let p = row.advanced(by: x * 4)
                let r = Int(p[2])
                let g = Int(p[1])
                let b = Int(p[0])
                luma[base + x] = UInt8(min(255, (77 * r + 150 * g + 29 * b) >> 8))
            }
        }
        return luma
    }

    private static func lineMask(luma: [UInt8], width: Int, height: Int, gap: Int) -> [UInt8] {
        var line = [UInt8](repeating: 0, count: width * height)
        for y in gap..<(height - gap) {
            for x in gap..<(width - gap) {
                let index = y * width + x
                let center = Int(luma[index])
                var localMax = center
                var yy = y - gap
                while yy <= y + gap {
                    var xx = x - gap
                    while xx <= x + gap {
                        localMax = max(localMax, Int(luma[yy * width + xx]))
                        xx += 1
                    }
                    yy += 1
                }
                if center < 150 && center + 16 < localMax {
                    line[index] = 1
                }
            }
        }
        return line
    }

    private static func closedMask(line: [UInt8], luma: [UInt8], width: Int, height: Int, gap: Int) -> [UInt8] {
        var closed = line
        for y in gap..<(height - gap) {
            for x in gap..<(width - gap) {
                let index = y * width + x
                if line[index] == 1 { continue }
                var hits = 0
                var yy = y - gap
                while yy <= y + gap {
                    var xx = x - gap
                    while xx <= x + gap {
                        if line[yy * width + xx] == 1 { hits += 1 }
                        xx += 1
                    }
                    yy += 1
                }
                if hits >= 3 && Int(luma[index]) < 190 {
                    closed[index] = 1
                }
            }
        }
        return closed
    }

    private static func paint(
        src: UnsafeMutablePointer<UInt8>, dst: UnsafeMutablePointer<UInt8>,
        luma: [UInt8], line: [UInt8], closed: [UInt8],
        width: Int, height: Int, srcRow: Int, dstRow: Int, amount: Double
    ) {
        for y in 0..<height {
            let row = src.advanced(by: y * srcRow)
            let out = dst.advanced(by: y * dstRow)
            for x in 0..<width {
                let index = y * width + x
                let p = row.advanced(by: x * 4)
                let o = out.advanced(by: x * 4)
                o[0] = p[0]
                o[1] = p[1]
                o[2] = p[2]
                o[3] = p[3]
                if y == 0 || x == 0 || y == height - 1 || x == width - 1 { continue }
                let stats = neighborStats(closed: closed, luma: luma, x: x, y: y, width: width)
                let center = Int(luma[index])
                if closed[index] == 1 && stats.neighbors >= 2 {
                    let lift = Int((1.0 - amount) * 18.0)
                    let target = min(center, stats.darkest + lift)
                    let keep = 1.0 - amount * 0.65
                    let mixed = Int((Double(center) * keep + Double(target) * (1.0 - keep)).rounded())
                    if mixed < center {
                        scale(pixel: o, by: Double(max(mixed, 1)) / Double(max(center, 1)))
                    }
                } else if line[index] == 1 && stats.neighbors <= 1 {
                    let paper = paperLuma(closed: closed, luma: luma, x: x, y: y, width: width)
                    if let paper {
                        let keep = 1.0 - amount * 0.7
                        let mixed = Int((Double(center) * keep + Double(paper) * (1.0 - keep)).rounded())
                        if mixed > center {
                            scale(pixel: o, by: Double(mixed) / Double(max(center, 1)))
                        }
                    }
                }
            }
        }
    }

    private static func neighborStats(closed: [UInt8], luma: [UInt8], x: Int, y: Int, width: Int) -> (neighbors: Int, darkest: Int) {
        var neighbors = 0
        var darkest = 255
        for yy in (y - 1)...(y + 1) {
            for xx in (x - 1)...(x + 1) {
                if xx == x && yy == y { continue }
                if closed[yy * width + xx] == 1 {
                    neighbors += 1
                    darkest = min(darkest, Int(luma[yy * width + xx]))
                }
            }
        }
        return (neighbors, darkest)
    }

    private static func paperLuma(closed: [UInt8], luma: [UInt8], x: Int, y: Int, width: Int) -> Int? {
        var paper = 0
        var paperCount = 0
        for yy in (y - 1)...(y + 1) {
            for xx in (x - 1)...(x + 1) {
                if closed[yy * width + xx] == 0 {
                    paper += Int(luma[yy * width + xx])
                    paperCount += 1
                }
            }
        }
        guard paperCount > 0 else { return nil }
        return paper / paperCount
    }

    private static func scale(pixel: UnsafeMutablePointer<UInt8>, by factor: Double) {
        pixel[0] = UInt8(min(255, max(0, Int((Double(pixel[0]) * factor).rounded()))))
        pixel[1] = UInt8(min(255, max(0, Int((Double(pixel[1]) * factor).rounded()))))
        pixel[2] = UInt8(min(255, max(0, Int((Double(pixel[2]) * factor).rounded()))))
    }

    static func apply(to image: UIImage, strength: Double) -> UIImage? {
        guard let buffer = pixelBuffer(from: image) else { return nil }
        let rebuilt = OutlineRebuild().apply(buffer, strength: strength)
        return uiImage(from: rebuilt, scale: image.scale)
    }

    private static func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let cg = image.cgImage else { return nil }
        let width = cg.width
        let height = cg.height
        guard let buffer = ColorRebuildBuffer.make(width: width, height: height) else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        guard let context = CGContext(
            data: base,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        context.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private static func uiImage(from buffer: CVPixelBuffer, scale: CGFloat) -> UIImage? {
        let ci = CIImage(cvPixelBuffer: buffer)
        let context = CIContext(options: [.cacheIntermediates: false])
        guard let cg = context.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg, scale: scale, orientation: .up)
    }
}

enum ColorRebuildBuffer {
    static func make(width: Int, height: Int) -> CVPixelBuffer? {
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
