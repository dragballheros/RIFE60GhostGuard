import Foundation
import CoreML
import CoreVideo

/// Neural preconditioning for downstream lossy transcodes.
/// Runs at the source cadence, before RIFE, so the restoration cost is paid only
/// for the original frames and the cleaned source becomes the temporal input to RIFE.
final class SocialCompressionRestorer {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputShape: [Int]

    init(modelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: configuration)

        guard let input = model.modelDescription.inputDescriptionsByName.values.first,
              let output = model.modelDescription.outputDescriptionsByName.values.first,
              let constraint = input.multiArrayConstraint else {
            throw ProcessorError.conversionFailed("Social Compression Core ML model must expose a MultiArray input/output")
        }

        inputName = input.name
        outputName = output.name
        inputShape = constraint.shape.map { $0.intValue }

        guard inputShape.count == 4, inputShape[0] == 1, inputShape[1] == 3 else {
            throw ProcessorError.conversionFailed("Social Compression Core ML model must use NCHW RGB input")
        }

        guard output.type == .multiArray else {
            throw ProcessorError.conversionFailed("Social Compression Core ML model must return a MultiArray")
        }
    }

    func apply(_ pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        // Prefer full-frame inference. A dynamic 1x restoration model is the only
        // configuration enabled here because arbitrary fixed-tile execution would
        // require an additional seam-safe compositor and can alter anime line weight.
        let requiredWidth = inputShape[3]
        let requiredHeight = inputShape[2]
        if requiredWidth > 0 && requiredHeight > 0 &&
            (requiredWidth != width || requiredHeight != height) {
            throw ProcessorError.conversionFailed(
                "Social Compression model requires (requiredWidth)x(requiredHeight); source is (width)x(height)"
            )
        }

        let input = try rgbArray(pixelBuffer, width: width, height: height)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: input)
        ])
        let prediction = try model.prediction(from: provider)

        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw ProcessorError.conversionFailed("Social Compression Core ML model returned no output")
        }

        let outputShape = output.shape.map { $0.intValue }
        guard outputShape.count == 4, outputShape[0] == 1, outputShape[1] == 3 else {
            throw ProcessorError.conversionFailed("Social Compression Core ML output is not NCHW RGB")
        }

        let outputHeight = outputShape[2]
        let outputWidth = outputShape[3]
        guard outputWidth == width && outputHeight == height else {
            throw ProcessorError.conversionFailed(
                "Social Compression model changed resolution to (outputWidth)x(outputHeight)"
            )
        }

        return try pixelBuffer(from: output, width: width, height: height)
    }

    private func rgbArray(_ source: CVPixelBuffer, width: Int, height: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )

        CVPixelBufferLockBaseAddress(source, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(source) else {
            throw ProcessorError.conversionFailed("Social Compression source has no base address")
        }

        let src = base.assumingMemoryBound(to: UInt8.self)
        let srcBPR = CVPixelBufferGetBytesPerRow(source)
        let dst = array.dataPointer.assumingMemoryBound(to: Float32.self)
        let plane = width * height

        for y in 0..<height {
            let row = src.advanced(by: y * srcBPR)
            for x in 0..<width {
                let p = row.advanced(by: x * 4)
                let i = y * width + x
                dst[i] = Float(p[2]) / 255.0
                dst[plane + i] = Float(p[1]) / 255.0
                dst[2 * plane + i] = Float(p[0]) / 255.0
            }
        }

        return array
    }

    private func pixelBuffer(from array: MLMultiArray, width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]

        var output: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            attrs as CFDictionary, &output
        ) == kCVReturnSuccess, let output else {
            throw ProcessorError.conversionFailed("could not allocate Social Compression output")
        }

        CVPixelBufferLockBaseAddress(output, [])
        defer { CVPixelBufferUnlockBaseAddress(output, []) }

        guard let base = CVPixelBufferGetBaseAddress(output) else {
            throw ProcessorError.conversionFailed("Social Compression output has no base address")
        }

        let dst = base.assumingMemoryBound(to: UInt8.self)
        let src = array.dataPointer.assumingMemoryBound(to: Float32.self)
        let strides = array.strides.map { $0.intValue }
        guard strides.count >= 4 else {
            throw ProcessorError.conversionFailed("Social Compression output has invalid strides")
        }

        let cStride = strides[1]
        let yStride = strides[2]
        let xStride = strides[3]
        let bpr = CVPixelBufferGetBytesPerRow(output)

        for y in 0..<height {
            for x in 0..<width {
                let r = src[y * yStride + x * xStride]
                let g = src[cStride + y * yStride + x * xStride]
                let b = src[2 * cStride + y * yStride + x * xStride]
                let p = dst.advanced(by: y * bpr + x * 4)

                p[0] = UInt8(max(0, min(255, Int((b * 255.0).rounded()))))
                p[1] = UInt8(max(0, min(255, Int((g * 255.0).rounded()))))
                p[2] = UInt8(max(0, min(255, Int((r * 255.0).rounded()))))
                p[3] = 255
            }
        }

        return output
    }
}
