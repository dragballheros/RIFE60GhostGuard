import Foundation
import CoreML
import CoreVideo
import CoreImage

/// Mobile-first neural compression restoration.
///
/// The original implementation ran the 512x512 RealPLKSR model across the
/// full-resolution frame through Vision. On an iPhone that can create a large
/// GPU/memory workload before the rest of the pipeline even starts.
///
/// This implementation deliberately keeps the restoration stage mandatory, but
/// reduces its cost:
/// - restores a capped 960x540 working image for HD/4K source frames
/// - uses only four 512x512 model predictions for a 16:9 source
/// - calls Core ML directly instead of creating a Vision request per tile
/// - keeps Core ML off the GPU when possible (.cpuAndNeuralEngine)
/// - scales the restored result back to the original resolution with Lanczos
///
/// The output remains the original source resolution and cadence.
final class SocialCompressionRestorer {
    private let model: MLModel
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let tileSize = 512
    private let overlap = 64
    private let maxWorkingWidth = 960
    private let maxWorkingHeight = 540

    init(modelURL: URL) throws {
        let configuration = MLModelConfiguration()
        // RIFE/Real-CUGAN own the GPU later in the pipeline. Keeping this
        // restoration model on CPU + Neural Engine avoids competing with the
        // Metal renderer and is much friendlier to interactive iPhone use.
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    func apply(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let workingSize = workingSize(forWidth: sourceWidth, height: sourceHeight)

        let workingInput = try makePixelBuffer(width: workingSize.width, height: workingSize.height)
        let workingImage = CIImage(cvPixelBuffer: source)
            .transformed(
                by: CGAffineTransform(
                    scaleX: CGFloat(workingSize.width) / CGFloat(sourceWidth),
                    y: CGFloat(workingSize.height) / CGFloat(sourceHeight)
                )
            )

        context.render(
            workingImage,
            to: workingInput,
            bounds: CGRect(x: 0, y: 0, width: workingSize.width, height: workingSize.height),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )

        var accum = [Float](repeating: 0, count: workingSize.width * workingSize.height * 4)
        var weights = [Float](repeating: 0, count: workingSize.width * workingSize.height)

        for y in tileStarts(length: workingSize.height) {
            for x in tileStarts(length: workingSize.width) {
                try autoreleasepool {
                    let cropWidth = min(tileSize, workingSize.width - x)
                    let cropHeight = min(tileSize, workingSize.height - y)
                    let tile = try makeTile(workingInput, originX: x, originY: y)
                    let restored = try infer(tile)
                    try blend(
                        restored,
                        into: &accum,
                        weights: &weights,
                        destinationWidth: workingSize.width,
                        destinationHeight: workingSize.height,
                        originX: x,
                        originY: y,
                        cropWidth: cropWidth,
                        cropHeight: cropHeight
                    )
                }
            }
        }

        let restoredWorking = try makePixelBuffer(width: workingSize.width, height: workingSize.height)
        try write(
            accum,
            weights: weights,
            to: restoredWorking,
            width: workingSize.width,
            height: workingSize.height
        )

        let output = try makePixelBuffer(width: sourceWidth, height: sourceHeight)
        let restoredImage = CIImage(cvPixelBuffer: restoredWorking)
            .applyingFilter("CILanczosScaleTransform", parameters: [
                kCIInputScaleKey: CGFloat(sourceWidth) / CGFloat(workingSize.width),
                kCIInputAspectRatioKey: CGFloat(sourceHeight) / CGFloat(workingSize.height) *
                    CGFloat(workingSize.width) / CGFloat(sourceWidth)
            ])
            .cropped(to: CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight))

        context.render(
            restoredImage,
            to: output,
            bounds: CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )

        return output
    }

    private func workingSize(forWidth width: Int, height: Int) -> (width: Int, height: Int) {
        guard width > maxWorkingWidth || height > maxWorkingHeight else {
            return (width, height)
        }

        let scale = min(
            CGFloat(maxWorkingWidth) / CGFloat(width),
            CGFloat(maxWorkingHeight) / CGFloat(height)
        )

        return (
            max(1, Int((CGFloat(width) * scale).rounded())),
            max(1, Int((CGFloat(height) * scale).rounded()))
        )
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

    private func makeTile(
        _ source: CVPixelBuffer,
        originX: Int,
        originY: Int
    ) throws -> CVPixelBuffer {
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
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: tile)
        ])
        let result = try model.prediction(from: provider)

        guard let output = result.featureValue(for: "upscaled")?.imageBufferValue else {
            throw ProcessorError.conversionFailed(
                "Social Compression Core ML returned no image output"
            )
        }
        return output
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
            throw ProcessorError.conversionFailed(
                "could not allocate Social Compression pixel buffer"
            )
        }
        return result
    }
}
