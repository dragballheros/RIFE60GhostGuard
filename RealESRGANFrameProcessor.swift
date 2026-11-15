import Foundation
import CoreML
import CoreVideo

/// Single-frame tiled RealESRGAN_x2plus inference. The model is loaded once per
/// routed segment and released before the CUGAN backend is reactivated.
final class RealESRGANFrameProcessor: @unchecked Sendable {
    private let tileSize = 384
    private let overlap = 32
    private let model: MLModel
    private let inputArray: MLMultiArray
    private let inputStride: [Int]

    init() throws {
        guard let modelURL = Bundle.main.url(forResource: "RealESRGAN_x2plus_Tile384", withExtension: "mlmodelc") else {
            throw ProcessorError.conversionFailed("Bundled Real-ESRGAN x2plus Core ML model is missing.")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = ModelComputePreference.current.units
        configuration.allowLowPrecisionAccumulationOnGPU = true
        let loaded = try MLModel(contentsOf: modelURL, configuration: configuration)
        guard let inputDescription = loaded.modelDescription.inputDescriptionsByName["input"]?.multiArrayConstraint,
              let outputDescription = loaded.modelDescription.outputDescriptionsByName["output"]?.multiArrayConstraint,
              inputDescription.dataType == .float32,
              outputDescription.dataType == .float32,
              inputDescription.shape.map(\.intValue) == [1, 3, tileSize, tileSize],
              outputDescription.shape.map(\.intValue) == [1, 3, tileSize * 2, tileSize * 2] else {
            throw ProcessorError.conversionFailed("Real-ESRGAN model tensor contract does not match the 384×384 RGB tile profile.")
        }
        self.model = loaded
        self.inputArray = try MLMultiArray(shape: inputDescription.shape, dataType: .float32)
        self.inputStride = inputArray.strides.map(\.intValue)
        DiagnosticsLogger.shared.log("Real-ESRGAN x2plus loaded • tile=\(tileSize)x\(tileSize) • core=\(tileSize - overlap * 2)x\(tileSize - overlap * 2) • compute=\(configuration.computeUnits.rawValue)")
    }

    func upscaleFrame(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw ProcessorError.conversionFailed("Real-ESRGAN received an empty source frame.")
        }
        let targetWidth = sourceWidth * 2
        let targetHeight = sourceHeight * 2
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: targetWidth,
            kCVPixelBufferHeightKey as String: targetHeight,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var targetBuffer: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth,
            targetHeight,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &targetBuffer
        ) == kCVReturnSuccess, let target = targetBuffer else {
            throw ProcessorError.conversionFailed("Could not allocate the Real-ESRGAN 2× output frame.")
        }

        let tileInput = UnsafeMutablePointer<UInt8>.allocate(capacity: tileSize * tileSize * 4)
        let tileOutput = UnsafeMutablePointer<UInt8>.allocate(capacity: tileSize * 2 * tileSize * 2 * 4)
        defer {
            tileInput.deallocate()
            tileOutput.deallocate()
        }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(target, [])
        defer {
            CVPixelBufferUnlockBaseAddress(target, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let targetBase = CVPixelBufferGetBaseAddress(target) else {
            throw ProcessorError.conversionFailed("Real-ESRGAN source or destination pixel address is unavailable.")
        }

        let step = tileSize - overlap * 2
        let sourceStride = CVPixelBufferGetBytesPerRow(source)
        let targetStride = CVPixelBufferGetBytesPerRow(target)
        var y = 0
        while y < sourceHeight {
            let coreHeight = min(step, sourceHeight - y)
            var x = 0
            while x < sourceWidth {
                try Task.checkCancellation()
                let coreWidth = min(step, sourceWidth - x)
                RealCUGANTileCopy.crop(
                    source: sourceBase.assumingMemoryBound(to: UInt8.self),
                    sourceStride: sourceStride,
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    destination: tileInput,
                    destinationStride: tileSize * 4,
                    ciOriginX: x - overlap,
                    ciOriginY: y - overlap,
                    tileWidth: tileSize,
                    tileHeight: tileSize
                )
                packInput(tileInput)
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "input": MLFeatureValue(multiArray: inputArray)
                ])
                let prediction = try model.prediction(from: provider)
                guard let output = prediction.featureValue(for: "output")?.multiArrayValue,
                      output.shape.map(\.intValue) == [1, 3, tileSize * 2, tileSize * 2] else {
                    throw ProcessorError.conversionFailed("Real-ESRGAN returned an invalid output tensor.")
                }
                unpackOutput(output, into: tileOutput)

                RealCUGANTileCopy.stitch(
                    modelOutput: tileOutput,
                    outputStride: tileSize * 2 * 4,
                    outputHeight: tileSize * 2,
                    destination: targetBase.assumingMemoryBound(to: UInt8.self),
                    destinationStride: targetStride,
                    destinationHeight: targetHeight,
                    coreX: x,
                    coreY: y,
                    coreWidth: coreWidth,
                    coreHeight: coreHeight,
                    overlap: overlap
                )
                x += step
            }
            y += step
        }
        if CVPixelBufferGetWidth(target) != targetWidth || CVPixelBufferGetHeight(target) != targetHeight {
            throw ProcessorError.conversionFailed("Real-ESRGAN stitched frame dimensions are incorrect.")
        }
        return target
    }

    private func packInput(_ bgra: UnsafePointer<UInt8>) {
        let destination = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
        for y in 0..<tileSize {
            for x in 0..<tileSize {
                let sourceOffset = (y * tileSize + x) * 4
                let pixel = y * inputStride[2] + x * inputStride[3]
                destination[pixel] = Float(bgra[sourceOffset + 2]) / 255.0
                destination[inputStride[1] + pixel] = Float(bgra[sourceOffset + 1]) / 255.0
                destination[inputStride[1] * 2 + pixel] = Float(bgra[sourceOffset]) / 255.0
            }
        }
    }

    private func unpackOutput(_ output: MLMultiArray, into bgra: UnsafeMutablePointer<UInt8>) {
        let source = output.dataPointer.assumingMemoryBound(to: Float.self)
        let outputStride = output.strides.map(\.intValue)
        let width = tileSize * 2
        let height = tileSize * 2
        for y in 0..<height {
            for x in 0..<width {
                let pixel = y * outputStride[2] + x * outputStride[3]
                let destinationOffset = (y * width + x) * 4
                let red = min(max(source[pixel], 0), 1)
                let green = min(max(source[outputStride[1] + pixel], 0), 1)
                let blue = min(max(source[outputStride[1] * 2 + pixel], 0), 1)
                bgra[destinationOffset] = UInt8((blue * 255).rounded())
                bgra[destinationOffset + 1] = UInt8((green * 255).rounded())
                bgra[destinationOffset + 2] = UInt8((red * 255).rounded())
                bgra[destinationOffset + 3] = 255
            }
        }
    }
}
