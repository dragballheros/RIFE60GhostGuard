import CoreImage
import CoreVideo
import Foundation

enum RealCUGANTileCopyCheck {
    static func main() throws {
        let width = 96
        let height = 80
        let tile = 40
        let overlap = 8
        let source = try buffer(width: width, height: height)
        let copied = try buffer(width: tile, height: tile)
        let rendered = try buffer(width: tile, height: tile)
        fill(source)
        let ciX = 20 - overlap
        let ciY = 16 - overlap
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(copied, [])
        defer {
            CVPixelBufferUnlockBaseAddress(copied, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        RealCUGANTileCopy.crop(
            source: CVPixelBufferGetBaseAddress(source)!.assumingMemoryBound(to: UInt8.self),
            sourceStride: CVPixelBufferGetBytesPerRow(source),
            sourceWidth: width,
            sourceHeight: height,
            destination: CVPixelBufferGetBaseAddress(copied)!.assumingMemoryBound(to: UInt8.self),
            destinationStride: CVPixelBufferGetBytesPerRow(copied),
            ciOriginX: ciX,
            ciOriginY: ciY,
            tileWidth: tile,
            tileHeight: tile
        )
        CVPixelBufferUnlockBaseAddress(source, .readOnly)
        let image = CIImage(cvPixelBuffer: source).clampedToExtent()
            .cropped(to: CGRect(x: ciX, y: ciY, width: tile, height: tile))
            .transformed(by: CGAffineTransform(translationX: CGFloat(-ciX), y: CGFloat(-ciY)))
        CIContext(options: [.cacheIntermediates: false]).render(
            image,
            to: rendered,
            bounds: CGRect(x: 0, y: 0, width: tile, height: tile),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(rendered, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(rendered, .readOnly) }
        let maxDelta = delta(
            CVPixelBufferGetBaseAddress(copied)!.assumingMemoryBound(to: UInt8.self),
            CVPixelBufferGetBytesPerRow(copied),
            CVPixelBufferGetBaseAddress(rendered)!.assumingMemoryBound(to: UInt8.self),
            CVPixelBufferGetBytesPerRow(rendered),
            width: tile,
            height: tile
        )
        if maxDelta > 1 {
            fputs("FAIL: Real-CUGAN tile copy differs from Core Image by \(maxDelta)\n", stderr)
            exit(1)
        }
        print("PASS: Real-CUGAN tile copy matches Core Image within \(maxDelta) LSB")
    }

    static func buffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &buffer) == kCVReturnSuccess,
              let buffer else {
            throw NSError(domain: "tile", code: 1)
        }
        return buffer
    }

    static func fill(_ buffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let height = CVPixelBufferGetHeight(buffer)
        let width = CVPixelBufferGetWidth(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * stride + x * 4
                base[i] = UInt8((x * 3 + y) & 255)
                base[i + 1] = UInt8((x + y * 5) & 255)
                base[i + 2] = UInt8((x * 7) & 255)
                base[i + 3] = 255
            }
        }
    }

    static func delta(_ a: UnsafePointer<UInt8>, _ aStride: Int, _ b: UnsafePointer<UInt8>, _ bStride: Int, width: Int, height: Int) -> Int {
        var maxDelta = 0
        for y in 0..<height {
            for x in 0..<(width * 4) {
                let d = abs(Int(a[y * aStride + x]) - Int(b[y * bStride + x]))
                if d > maxDelta { maxDelta = d }
            }
        }
        return maxDelta
    }
}

do { try RealCUGANTileCopyCheck.main() } catch {
    fputs("FAIL: \(error)\n", stderr)
    exit(1)
}
