import Foundation
import CoreML
import CoreImage
import CoreVideo

/// Masked anime/manga LaMa inpainting before interpolation/upscale/sharpening.
/// Only selected regions (plus explicit padding) are replaced. Every BGRA byte
/// outside those regions is copied unchanged, including alpha. Serial-use class.
final class AnimeWatermarkRemover {
    private let configuration: WatermarkConfiguration
    private let model: MLModel
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let modelSize = 512
    private var imagePool: CVPixelBufferPool?
    private var maskPool: CVPixelBufferPool?
    private var resultPool: CVPixelBufferPool?
    private var patchPool: CVPixelBufferPool?
    private var resultWidth = 0
    private var resultHeight = 0
    private var patchWidth = 0
    private var patchHeight = 0

    init(configuration: WatermarkConfiguration, modelURL: URL? = nil) throws {
        guard configuration.enabled, configuration.isValid else {
            throw ProcessorError.conversionFailed("mark at least one watermark region before enabling removal")
        }
        self.configuration = configuration
        guard let url = modelURL ?? Bundle.main.url(forResource: "AnimeWatermarkLaMa512", withExtension: "mlmodelc") else {
            throw ProcessorError.conversionFailed("the bundled Anime/Manga LaMa model is missing")
        }
        let options = MLModelConfiguration()
        // Keep Fourier calculations FP32 and avoid the older LaMa ANE/fp16 path.
        options.computeUnits = .cpuAndGPU
        self.model = try MLModel(contentsOf: url, configuration: options)
        guard let image = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint,
              let mask = model.modelDescription.inputDescriptionsByName["mask"]?.imageConstraint,
              image.pixelsWide == 512, image.pixelsHigh == 512,
              mask.pixelsWide == 512, mask.pixelsHigh == 512,
              model.modelDescription.outputDescriptionsByName["output"]?.type == .image else {
            throw ProcessorError.conversionFailed("Anime/Manga LaMa model has an incompatible input/output contract")
        }
        imagePool = try Self.makePool(width: 512, height: 512, format: kCVPixelFormatType_32BGRA)
        maskPool = try Self.makePool(width: 512, height: 512, format: kCVPixelFormatType_OneComponent8)
        DiagnosticsLogger.shared.log("Anime watermark removal active • model=\(WatermarkConfiguration.modelID) • regions=\(configuration.regions.count) • padding=\(configuration.paddingPixels)px • CPU+GPU FP32")
    }

    func apply(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA else {
            throw ProcessorError.conversionFailed("anime inpainting requires a BGRA source frame")
        }
        let width = CVPixelBufferGetWidth(source), height = CVPixelBufferGetHeight(source)
        if resultPool == nil || resultWidth != width || resultHeight != height {
            resultPool = try Self.makePool(width: width, height: height, format: kCVPixelFormatType_32BGRA)
            resultWidth = width; resultHeight = height
        }
        var current = source
        for region in configuration.regions {
            try Task.checkCancellation()
            current = try autoreleasepool { try remove(region, from: current) }
        }
        return current
    }

    private func remove(_ region: WatermarkRegion, from source: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source), height = CVPixelBufferGetHeight(source)
        let padding = CGFloat(min(max(configuration.paddingPixels, 0), 16))
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        // Core Image coordinates have a bottom-left origin; preview masks do not.
        let target = CGRect(x: region.x * Double(width),
                            y: (1 - region.y - region.height) * Double(height),
                            width: region.width * Double(width), height: region.height * Double(height))
            .insetBy(dx: -padding, dy: -padding).integral.intersection(bounds)
        guard !target.isNull, target.width >= 1, target.height >= 1 else { return source }
        // Give the model surrounding line/color context rather than resizing the
        // entire 4K frame. Border regions use replicated-edge padding.
        let side = Int(ceil(max(256, max(target.width, target.height) * 2)))
        let crop = CGRect(x: floor(target.midX - CGFloat(side) / 2),
                          y: floor(target.midY - CGFloat(side) / 2), width: side, height: side)
        let scale = CGFloat(modelSize) / CGFloat(side)
        let imageInput = try Self.allocate(imagePool)
        let maskInput = try Self.allocate(maskPool)
        let inputImage = CIImage(cvPixelBuffer: source, options: [.colorSpace: colorSpace])
            .clampedToExtent().cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        context.render(inputImage, to: imageInput, bounds: CGRect(x: 0, y: 0, width: modelSize, height: modelSize), colorSpace: colorSpace)
        try fillMask(maskInput, target: target, crop: crop, scale: scale)
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: imageInput),
            "mask": MLFeatureValue(pixelBuffer: maskInput)
        ])
        let prediction = try model.prediction(from: features)
        guard let predicted = prediction.featureValue(for: "output")?.imageBufferValue else {
            throw ProcessorError.conversionFailed("Anime/Manga LaMa returned no image")
        }
        let patchWidth = Int(target.width), patchHeight = Int(target.height)
        if patchPool == nil || self.patchWidth != patchWidth || self.patchHeight != patchHeight {
            patchPool = try Self.makePool(width: patchWidth, height: patchHeight, format: kCVPixelFormatType_32BGRA)
            self.patchWidth = patchWidth; self.patchHeight = patchHeight
        }
        let patch = try Self.allocate(patchPool)
        let restored = CIImage(cvPixelBuffer: predicted, options: [.colorSpace: colorSpace])
            .applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: CGFloat(side) / CGFloat(modelSize), kCIInputAspectRatioKey: 1])
        let translated = restored.transformed(by: CGAffineTransform(translationX: crop.minX - target.minX, y: crop.minY - target.minY))
        context.render(translated, to: patch, bounds: CGRect(x: 0, y: 0, width: patchWidth, height: patchHeight), colorSpace: colorSpace)
        let result = try Self.allocate(resultPool)
        try composite(patch: patch, source: source, destination: result, target: target)
        CVBufferPropagateAttachments(source, result)
        return result
    }

    private func fillMask(_ buffer: CVPixelBuffer, target: CGRect, crop: CGRect, scale: CGFloat) throws {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let address = CVPixelBufferGetBaseAddress(buffer) else { throw ProcessorError.conversionFailed("mask buffer address unavailable") }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        address.initializeMemory(as: UInt8.self, repeating: 0, count: stride * modelSize)
        let mask = address.assumingMemoryBound(to: UInt8.self)
        let left = max(0, Int(floor((target.minX - crop.minX) * scale)))
        let right = min(modelSize, Int(ceil((target.maxX - crop.minX) * scale)))
        let top = max(0, Int(floor((crop.maxY - target.maxY) * scale)))
        let bottom = min(modelSize, Int(ceil((crop.maxY - target.minY) * scale)))
        guard right > left, bottom > top else { return }
        for row in top..<bottom {
            mask.advanced(by: row * stride + left).initialize(repeating: 255, count: right - left)
        }
    }

    private func composite(patch: CVPixelBuffer, source: CVPixelBuffer, destination: CVPixelBuffer, target: CGRect) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(patch, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(patch, .readOnly)
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let src = CVPixelBufferGetBaseAddress(source), let patchBase = CVPixelBufferGetBaseAddress(patch),
              let dst = CVPixelBufferGetBaseAddress(destination) else { throw ProcessorError.conversionFailed("inpainting composite address unavailable") }
        let width = CVPixelBufferGetWidth(source), height = CVPixelBufferGetHeight(source)
        let srcStride = CVPixelBufferGetBytesPerRow(source), dstStride = CVPixelBufferGetBytesPerRow(destination)
        let patchStride = CVPixelBufferGetBytesPerRow(patch)
        for row in 0..<height { dst.advanced(by: row * dstStride).copyMemory(from: src.advanced(by: row * srcStride), byteCount: width * 4) }
        let left = Int(target.minX), right = Int(target.maxX)
        let top = height - Int(target.maxY), bottom = height - Int(target.minY)
        let original = src.assumingMemoryBound(to: UInt8.self)
        let replacement = patchBase.assumingMemoryBound(to: UInt8.self)
        let output = dst.assumingMemoryBound(to: UInt8.self)
        for row in top..<bottom {
            for col in left..<right {
                let edgeDistance = min(Double(col - left) + 0.5, Double(right - col) - 0.5,
                                       Double(row - top) + 0.5, Double(bottom - row) - 0.5)
                let alpha = min(edgeDistance / 2, 1) // Two-pixel feather stays inside the padded mask.
                let sourceOffset = row * srcStride + col * 4
                let targetOffset = row * dstStride + col * 4
                let patchOffset = (row - top) * patchStride + (col - left) * 4
                for channel in 0..<3 {
                    let value = Double(original[sourceOffset + channel]) * (1 - alpha) + Double(replacement[patchOffset + channel]) * alpha
                    output[targetOffset + channel] = UInt8(min(max(value.rounded(), 0), 255))
                }
            }
        }
    }

    private static func makePool(width: Int, height: Int, format: OSType) throws -> CVPixelBufferPool {
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool) == kCVReturnSuccess,
              let pool else { throw ProcessorError.conversionFailed("could not create anime inpainting buffer pool") }
        return pool
    }

    private static func allocate(_ pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        guard let pool else { throw ProcessorError.conversionFailed("anime inpainting buffer pool unavailable") }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer) == kCVReturnSuccess,
              let buffer else { throw ProcessorError.conversionFailed("could not allocate anime inpainting frame") }
        return buffer
    }
}
