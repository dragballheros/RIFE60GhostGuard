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
        let count = width * height
        var luma = [UInt8](repeating: 0, count: count)
        for y in 0..<height {
            let row = src.advanced(by: y * srcRow)
            for x in 0..<width {
                let p = row.advanced(by: x * 4)
                luma[y * width + x] = UInt8(min(255, (77 * Int(p[2]) + 150 * Int(p[1]) + 29 * Int(p[0])) >> 8))
            }
        }
        let gap = amount > 0.66 ? 2 : 1
        var line = [UInt8](repeating: 0, count: count)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let index = y * width + x
                let center = Int(luma[index])
                var localMax = center
                for yy in (y - gap)...(y + gap) {
                    for xx in (x - gap)...(x + gap) {
                        localMax = max(localMax, Int(luma[yy * width + xx]))
                    }
                }
                if center < 150 && center + 16 < localMax {
                    line[index] = 1
                }
            }
        }
        var closed = line
        for y in gap..<(height - gap) {
            for x in gap..<(width - gap) {
                let index = y * width + x
                if line[index] == 1 { continue }
                var hits = 0
                for yy in (y - gap)...(y + gap) {
                    for xx in (x - gap)...(x + gap) where line[yy * width + xx] == 1 {
                        hits += 1
                    }
                }
                let center = Int(luma[index])
                if hits >= 3 && center < 190 {
                    closed[index] = 1
                }
            }
        }

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
                guard y > 0, x > 0, y < height - 1, x < width - 1 else { continue }
                var neighbors = 0
                var darkest = Int(luma[index])
                for yy in (y - 1)...(y + 1) {
                    for xx in (x - 1)...(x + 1) where !(xx == x && yy == y) {
                        if closed[yy * width + xx] == 1 {
                            neighbors += 1
                            darkest = min(darkest, Int(luma[yy * width + xx]))
                        }
                    }
                }
                let center = Int(luma[index])
                if closed[index] == 1 && neighbors >= 2 {
                    let target = min(center, darkest + Int((1.0 - amount) * 18))
                    let mixed = Int((Double(center) * (1 - amount * 0.65) + Double(target) * amount * 0.65).rounded())
                    if mixed < center {
                        let scale = Double(max(mixed, 1)) / Double(max(center, 1))
                        o[0] = UInt8(min(255, max(0, Int((Double(p[0]) * scale).rounded()))))
                        o[1] = UInt8(min(255, max(0, Int((Double(p[1]) * scale).rounded()))))
                        o[2] = UInt8(min(255, max(0, Int((Double(p[2]) * scale).rounded()))))
                    }
                } else if line[index] == 1 && neighbors <= 1 {
                    var paper = 0
                    var paperCount = 0
                    for yy in (y - 1)...(y + 1) {
                        for xx in (x - 1)...(x + 1) where closed[yy * width + xx] == 0 {
                            paper += Int(luma[yy * width + xx])
                            paperCount += 1
                        }
                    }
                    if paperCount > 0 {
                        let lifted = paper / paperCount
                        let mixed = Int((Double(center) * (1 - amount * 0.7) + Double(lifted) * amount * 0.7).rounded())
                        if mixed > center {
                            let scale = Double(mixed) / Double(max(center, 1))
                            o[0] = UInt8(min(255, max(0, Int((Double(p[0]) * scale).rounded()))))
                            o[1] = UInt8(min(255, max(0, Int((Double(p[1]) * scale).rounded()))))
                            o[2] = UInt8(min(255, max(0, Int((Double(p[2]) * scale).rounded()))))
                        }
                    }
                }
            }
        }
        return destination
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
