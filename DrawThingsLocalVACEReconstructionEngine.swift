import Foundation
import _MediaGenerationKit
import DataModels
import ImageIO
import UIKit
import UniformTypeIdentifiers

enum DrawThingsVACEEngineError: LocalizedError {
    case notPrepared
    case invalidWindow
    case invalidInputFrame(Int)
    case invalidInputMask(Int)
    case missingGeneratedFrames(expected: Int, actual: Int)
    case couldNotDecodeOutput
    case missingGeneratedFrame(Int)

    var errorDescription: String? {
        switch self {
        case .notPrepared: return "The local Wan VACE model has not been prepared."
        case .invalidWindow: return "This on-device profile requires five-frame Wan VACE windows."
        case .invalidInputFrame(let index): return "Could not prepare source frame \(index) for local Wan VACE inference."
        case .invalidInputMask(let index): return "Could not prepare edit mask \(index) for local Wan VACE inference."
        case .missingGeneratedFrames(let expected, let actual):
            return "The local Wan VACE runtime returned \(actual) frames; \(expected) were required. No video was exported."
        case .couldNotDecodeOutput: return "Could not decode a frame returned by the local Wan VACE model."
        case .missingGeneratedFrame(let index): return "The local Wan VACE runtime did not return generated frame \(index)."
        }
    }
}

/// Real native Draw Things Metal/NNC Wan VACE inference.
/// This experimental profile is Wan 2.1 1.3B Q8 with 512x512 inference;
/// it is explicitly NOT equivalent to the requested 14B / 720p model.
actor DrawThingsLocalVACEReconstructionEngine: VACERegionReconstructionEngine {
    static let baseModel = "wan_v2.1_1.3b_480p_q8p.ckpt"
    static let vaceControl = "wan_v2.1_1.3b_vace_480p_q8p.ckpt"
    static let inferenceWidth = 512
    static let inferenceHeight = 512

    private var pipeline: MediaGenerationPipeline?
    private let modelSteps = 20

    func prepare(
        progress: @escaping @Sendable (String, Int64, Int64) -> Void = { _, _, _ in }
    ) async throws {
        if pipeline != nil { return }
        var localPipeline = try await MediaGenerationPipeline.fromPretrained(
            Self.baseModel, backend: .local
        )
        try await localPipeline.prepareLocalWanVACE13B(progress: progress)
        localPipeline.configuration.controls = [
            Control(file: Self.vaceControl, weight: 1.0, guidanceStart: 0.0,
                    guidanceEnd: 1.0, globalAveragePooling: false)
        ]
        configure(&localPipeline)
        pipeline = localPipeline
    }

    func reconstruct(_ request: VACERegionReconstructionRequest) async throws -> [VACERegionFrameOutput] {
        guard request.temporalWindowLength == 5, request.frames.count == 5,
              request.validFrameCount > 0, request.validFrameCount <= 5 else {
            throw DrawThingsVACEEngineError.invalidWindow
        }
        guard var localPipeline = pipeline else {
            throw DrawThingsVACEEngineError.notPrepared
        }

        var encodedFrames = [Data]()
        var encodedMasks = [Data]()
        encodedFrames.reserveCapacity(5)
        encodedMasks.reserveCapacity(5)
        for input in request.frames {
            let prepared = try VACEReconstructionCropPreprocessor.prepare(
                image: input.sourceCrop, mask: input.maskCrop,
                targetWidth: Self.inferenceWidth, targetHeight: Self.inferenceHeight,
                maximumInferencePixels: Self.inferenceWidth * Self.inferenceHeight
            )
            guard let imageData = Self.encode(prepared.image) else {
                throw DrawThingsVACEEngineError.invalidInputFrame(input.windowPosition)
            }
            // This app's mask means 0=preserve and 255=edit. Draw Things inpainting masks
            // use white=preserve and black=edit, so conversion is explicit and reversible.
            let inverseMask = try VACEMaskPlane(
                width: prepared.mask.width, height: prepared.mask.height,
                pixels: prepared.mask.pixels.map { 255 &- $0 }
            )
            guard let maskData = Self.encode(try Self.rgbaMaskFrame(inverseMask)) else {
                throw DrawThingsVACEEngineError.invalidInputMask(input.windowPosition)
            }
            encodedFrames.append(imageData)
            encodedMasks.append(maskData)
        }

        configure(&localPipeline)
        let resultSet = try await localPipeline.generate(
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            inputs: [
                MediaGenerationPipeline.VideoSequenceInput(frames: encodedFrames, masks: encodedMasks)
            ]
        )
        pipeline = localPipeline

        let generatedFrames: [MediaGenerationPipeline.Result]
        if resultSet.count == 1, resultSet[0].frameCount == request.temporalWindowLength {
            var unpacked = [MediaGenerationPipeline.Result]()
            unpacked.reserveCapacity(request.temporalWindowLength)
            for index in 0..<request.temporalWindowLength {
                guard let frame = resultSet[0].frame(at: index) else {
                    throw DrawThingsVACEEngineError.missingGeneratedFrame(index)
                }
                unpacked.append(frame)
            }
            generatedFrames = unpacked
        } else if resultSet.count == request.temporalWindowLength,
                  resultSet.allSatisfy({ $0.frameCount == 1 }) {
            generatedFrames = resultSet
        } else {
            let actual = resultSet.reduce(0) { $0 + $1.frameCount }
            throw DrawThingsVACEEngineError.missingGeneratedFrames(
                expected: request.temporalWindowLength, actual: actual
            )
        }

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE-VACE-frames-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        var outputs = [VACERegionFrameOutput]()
        outputs.reserveCapacity(request.validFrameCount)
        for position in 0..<request.validFrameCount {
            let path = tempDir.appendingPathComponent("frame-\(position).png")
            try generatedFrames[position].write(to: path, type: .png)
            guard let decoded = Self.decodeRGBA(from: path) else {
                throw DrawThingsVACEEngineError.couldNotDecodeOutput
            }
            let source = request.frames[position].sourceCrop
            let restored = try VACEReconstructionCropPreprocessor.restoreGeneratedCrop(
                decoded, toSourceWidth: source.width, height: source.height,
                maximumPixels: max(832 * 480, source.width * source.height)
            )
            outputs.append(VACERegionFrameOutput(windowPosition: position, reconstructedCrop: restored))
        }
        return outputs
    }

    private func configure(_ pipeline: inout MediaGenerationPipeline) {
        pipeline.configuration.width = Self.inferenceWidth
        pipeline.configuration.height = Self.inferenceHeight
        pipeline.configuration.numFrames = 5
        pipeline.configuration.fps = 16
        pipeline.configuration.steps = modelSteps
        pipeline.configuration.batchCount = 1
        pipeline.configuration.batchSize = 1
        pipeline.configuration.guidanceScale = 4.0
        pipeline.configuration.strength = 0.85
        pipeline.configuration.tiledDiffusion = false
        pipeline.configuration.tiledDecoding = false
        pipeline.configuration.controls = [
            Control(file: Self.vaceControl, weight: 1.0, guidanceStart: 0.0,
                    guidanceEnd: 1.0, globalAveragePooling: false)
        ]
    }

    private static func rgbaMaskFrame(_ mask: VACEMaskPlane) throws -> VACERGBAFrame {
        var bytes = [UInt8](repeating: 255, count: mask.pixelCount * 4)
        for pixel in 0..<mask.pixelCount {
            let value = mask.pixels[pixel]
            let offset = pixel * 4
            bytes[offset] = value
            bytes[offset + 1] = value
            bytes[offset + 2] = value
            bytes[offset + 3] = 255
        }
        return try VACERGBAFrame(width: mask.width, height: mask.height, bytes: bytes)
    }

    private static func encode(_ frame: VACERGBAFrame) -> Data? {
        let data = Data(frame.bytes)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let bitmap = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        )
        guard let image = CGImage(
            width: frame.width, height: frame.height, bitsPerComponent: 8,
            bitsPerPixel: 32, bytesPerRow: frame.width * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmap,
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ) else { return nil }
        return UIImage(cgImage: image, scale: 1, orientation: .up).pngData()
    }

    private static func decodeRGBA(from url: URL) -> VACERGBAFrame? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let width = image.width, height = image.height
        guard width > 0, height > 0, width * height <= 832 * 480 else { return nil }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let bitmap = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        let success = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmap.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard success else { return nil }
        return try? VACERGBAFrame(width: width, height: height, bytes: bytes)
    }
}
