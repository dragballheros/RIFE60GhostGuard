import Foundation
import CoreML
import Vision
import CoreVideo
import CoreImage

/// Neural compression-restoration stage for downstream social transcoding.
///
/// This uses a 1x RealPLKSR Core ML restoration model. It preserves resolution
/// while removing compression artifacts and blur. Frames are tiled at 512x512
/// with overlap/feathering, matching the model's intended deployment pattern.
/// It runs before RIFE, so only source-cadence frames pay the restoration cost.
final class SocialCompressionRestorer {
    private let visionModel: VNCoreMLModel
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let tileSize = 512
    private let overlap = 64

    init(modelURL: URL) throws {
        let model = try MLModel(contentsOf: modelURL, configuration: {
            let c = MLModelConfiguration()
            c.computeUnits = .all
            return c
        }())
        visionModel = try VNCoreMLModel(for: model)
    }

    func apply(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let output = try makePixelBuffer(width: width, height: height)

        var accum = [Float](repeating: 0, count: width * height * 4)
        var weights = [Float](repeating: 0, count: width * height)

        for y in tileStarts(length: height) {
            for x in tileStarts(length: width) {
                try autoreleasepool {
                    let cropWidth = min(tileSize, width - x)
                    let cropHeight = min(tileSize, height - y)
                    let tile = try makeTile(source, originX: x, originY: y)
                    let restored = try infer(tile)
                    try blend(
                        restored,
                        into: &accum,
                        weights: &weights,
                        destinationWidth: width,
                        destinationHeight: height,
                        originX: x,
                        originY: y,
                        cropWidth: cropWidth,
                        cropHeight: cropHeight
                    )
                }
            }
        }

        try write(accum, weights: weights, to: output, width: width, height: height)
        return output
    }

    private func tileStarts(length: Int) -> [Int] {
        guard length > tileSize else { return [0] }
        let step = max(1, tileSize - overlap)
        var values: [Int] = []
        var p = 0
        while true {
            let last = max(0, length - tileSize)
            let start = min(p, last)
            if values.last != start { values.append(start) }
            if start >= last { break }
            p += step
        }
        return values
    }

    private func makeTile(_ source: CVPixelBuffer, originX: Int, originY: Int) throws -> CVPixelBuffer {
        let tile = try makePixelBuffer(width: tileSize, height: tileSize)
        let sourceImage = CIImage(cvPixelBuffer: source)
        let rect = CGRect(
            x: originX,
            y: originY,
            width: min(tileSize, CVPixelBufferGetWidth(source) - originX),
            height: min(tileSize, CVPixelBufferGetHeight(source) - originY)
        )
        let cropped = sourceImage
            .cropped(to: rect)
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
            .clampedToExtent()
            .cropped(to: CGRect(x: 0, y: 0, width: tileSize, height: tileSize))

        context.render(
            cropped,
            to: tile,
            bounds: CGRect(x: 0, y: 0, width: tileSize, height: tileSize),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )
        return tile
    }

    private func infer(_ tile: CVPixelBuffer) throws -> CVPixelBuffer {
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: tile, options: [:])
        try handler.perform([request])

        guard let observation = request.results?.compactMap({ $0 as? VNPixelBufferObservation }).first else {
            throw ProcessorError.conversionFailed("Social Compression Core ML returned no pixel buffer")
        }
        return observation.pixelBuffer
    }

    private func blend(
        _ tile: CVPixelBuffer,
        into accum: inout [Float],
        weights: inout [Float],
        destinationWidth: Int,
        destinationHeight: Int,
        originX: Int,
        originY: Int,
        cropWidth: Int,
        cropHeight: Int
    ) throws {
        CVPixelBufferLockBaseAddress(tile, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(tile, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(tile) else {
            throw ProcessorError.conversionFailed("Social Compression tile has no base address")
        }

        let src = base.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(tile)
        let width = CVPixelBufferGetWidth(tile)
        let height = CVPixelBufferGetHeight(tile)

        for y in 0..<cropHeight {
            for x in 0..<cropWidth {
                let dx = originX + x
                let dy = originY + y
                guard dx < destinationWidth && dy < destinationHeight else { continue }

                var weight: Float = 1
                if originX > 0 { weight *= min(1, Float(x + 1) / Float(overlap)) }
                if originY > 0 { weight *= min(1, Float(y + 1) / Float(overlap)) }
                if originX + cropWidth < destinationWidth {
                    weight *= min(1, Float(cropWidth - x) / Float(overlap))
                }
                if originY + cropHeight < destinationHeight {
                    weight *= min(1, Float(cropHeight - y) / Float(overlap))
                }

                let sx = min(width - 1, x)
                let sy = min(height - 1, y)
                let p = src.advanced(by: sy * bpr + sx * 4)
                let i = dy * destinationWidth + dx
                accum[i * 4] += Float(p[2]) / 255 * weight
                accum[i * 4 + 1] += Float(p[1]) / 255 * weight
                accum[i * 4 + 2] += Float(p[0]) / 255 * weight
                accum[i * 4 + 3] += weight
                weights[i] += weight
            }
        }
    }

    private func write(
        _ accum: [Float],
        weights: [Float],
        to output: CVPixelBuffer,
        width: Int,
        height: Int
    ) throws {
        CVPixelBufferLockBaseAddress(output, [])
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        guard let base = CVPixelBufferGetBaseAddress(output) else {
            throw ProcessorError.conversionFailed("Social Compression output has no base address")
        }

        let dst = base.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(output)

        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                let w = max(weights[i], 0.0001)
                let p = dst.advanced(by: y * bpr + x * 4)
                p[0] = UInt8(max(0, min(255, Int((accum[i * 4 + 2] / w * 255).rounded()))))
                p[1] = UInt8(max(0, min(255, Int((accum[i * 4 + 1] / w * 255).rounded()))))
                p[2] = UInt8(max(0, min(255, Int((accum[i * 4] / w * 255).rounded()))))
                p[3] = 255
            }
        }
    }

    private func render(_ source: CVPixelBuffer, into destination: CVPixelBuffer, rect: CGRect) throws {
        let image = CIImage(cvPixelBuffer: source).cropped(to: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(source), height: CVPixelBufferGetHeight(source)))
        context.render(
            image,
            to: destination,
            bounds: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(destination), height: CVPixelBufferGetHeight(destination)),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )
    }

    private func makePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var result: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &result
        ) == kCVReturnSuccess, let result else {
            throw ProcessorError.conversionFailed("could not allocate Social Compression pixel buffer")
        }
        return result
    }
}
