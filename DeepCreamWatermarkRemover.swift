import Foundation
import CoreML
import CoreImage
import CoreVideo

/// DeepCreamPy v2 inference is isolated from the existing LaMa implementation.
/// It uses the original trained bar or mosaic checkpoint converted to Core ML.
final class DeepCreamWatermarkRemover: WatermarkRemovalApplying {
    private let configuration: WatermarkConfiguration
    private let model: MLModel
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let modelSize = 256
    private var resultPool: CVPixelBufferPool?
    private var resultWidth = 0
    private var resultHeight = 0
    private var brushRasters: [(region: WatermarkRegion, width: Int, height: Int, pixels: [UInt8])] = []

    init(configuration: WatermarkConfiguration, modelURL: URL? = nil) throws {
        guard configuration.enabled, configuration.isValid else {
            throw ProcessorError.conversionFailed("mark at least one region before enabling DeepCream")
        }
        guard configuration.model == .deepCreamBar || configuration.model == .deepCreamMosaic else {
            throw ProcessorError.conversionFailed("DeepCream requires the bar or mosaic model selection")
        }
        self.configuration = configuration
        let resourceName = configuration.model == .deepCreamBar ? "DeepCreamBar" : "DeepCreamMosaic"
        guard let url = modelURL ?? Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc") else {
            throw ProcessorError.conversionFailed("the bundled \(resourceName) Core ML model is missing")
        }
        let options = MLModelConfiguration()
        options.computeUnits = .all
        self.model = try MLModel(contentsOf: url, configuration: options)
        let inputNames = Set(model.modelDescription.inputDescriptionsByName.keys)
        guard Set(["Placeholder", "Placeholder_1", "Placeholder_2"]).isSubset(of: inputNames),
              model.modelDescription.outputDescriptionsByName["add"]?.type == .multiArray else {
            throw ProcessorError.conversionFailed("DeepCream Core ML input/output contract is incompatible")
        }
        DiagnosticsLogger.shared.log("DeepCream v2 active • model=\(resourceName) • regions=\(configuration.regions.count) • Core ML compute units=all")
    }

    func apply(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA else {
            throw ProcessorError.conversionFailed("DeepCream requires a BGRA source frame")
        }
        let width = CVPixelBufferGetWidth(source), height = CVPixelBufferGetHeight(source)
        if resultPool == nil || resultWidth != width || resultHeight != height {
            resultPool = try Self.makePool(width: width, height: height)
            resultWidth = width
            resultHeight = height
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
        let target = CGRect(x: region.x * Double(width),
                            y: (1 - region.y - region.height) * Double(height),
                            width: region.width * Double(width), height: region.height * Double(height))
            .insetBy(dx: -padding, dy: -padding).integral.intersection(bounds)
        guard !target.isNull, target.width >= 1, target.height >= 1 else { return source }

        let side = Int(ceil(max(256, max(target.width, target.height) * 2)))
        let crop = CGRect(x: floor(target.midX - CGFloat(side) / 2),
                          y: floor(target.midY - CGFloat(side) / 2),
                          width: CGFloat(side), height: CGFloat(side))
        let scale = CGFloat(modelSize) / CGFloat(side)
        let cropped = CIImage(cvPixelBuffer: source, options: [.colorSpace: colorSpace])
            .clampedToExtent().cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let inputCG = context.createCGImage(cropped, from: CGRect(x: 0, y: 0, width: modelSize, height: modelSize)) else {
            throw ProcessorError.conversionFailed("DeepCream could not prepare its 256×256 crop")
        }
        let rgba = try Self.rgbaBytes(from: inputCG, width: modelSize, height: modelSize)
        let brushPixels: [UInt8]?
        if let brush = region.brush {
            if let cached = brushRasters.first(where: { $0.region == region && $0.width == width && $0.height == height }) {
                brushPixels = cached.pixels
            } else {
                brushRasters.removeAll { $0.width != width || $0.height != height }
                let pixels = brush.raster(width: width, height: height, padding: configuration.paddingPixels)
                brushRasters.append((region: region, width: width, height: height, pixels: pixels))
                brushPixels = pixels
            }
        } else {
            brushPixels = nil
        }

        let shape: [NSNumber] = [1, NSNumber(value: modelSize), NSNumber(value: modelSize), 3]
        let maskedInput = try MLMultiArray(shape: shape, dataType: .float32)
        let originalInput = try MLMultiArray(shape: shape, dataType: .float32)
        let maskInput = try MLMultiArray(shape: shape, dataType: .float32)
        let maskedPtr = maskedInput.dataPointer.bindMemory(to: Float.self, capacity: modelSize * modelSize * 3)
        let originalPtr = originalInput.dataPointer.bindMemory(to: Float.self, capacity: modelSize * modelSize * 3)
        let maskPtr = maskInput.dataPointer.bindMemory(to: Float.self, capacity: modelSize * modelSize * 3)

        for row in 0..<modelSize {
            for col in 0..<modelSize {
                let xCI = crop.minX + (CGFloat(col) + 0.5) / scale
                let yCI = crop.maxY - (CGFloat(row) + 0.5) / scale
                let sx = min(max(Int(xCI.rounded(.down)), 0), width - 1)
                let sy = min(max(Int((CGFloat(height) - yCI).rounded(.down)), 0), height - 1)
                let masked = brushPixels.map { $0[sy * width + sx] != 0 } ?? target.contains(CGPoint(x: xCI, y: yCI))
                let known: Float = masked ? 0 : 1
                let pixel = (row * modelSize + col) * 3
                for channel in 0..<3 {
                    let value = Float(rgba[(row * modelSize + col) * 4 + channel]) / 255.0
                    originalPtr[pixel + channel] = value
                    maskPtr[pixel + channel] = known
                    maskedPtr[pixel + channel] = value * known
                }
            }
        }

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "Placeholder": MLFeatureValue(multiArray: maskedInput),
            "Placeholder_1": MLFeatureValue(multiArray: originalInput),
            "Placeholder_2": MLFeatureValue(multiArray: maskInput)
        ])
        let prediction = try model.prediction(from: features)
        guard let output = prediction.featureValue(for: "add")?.multiArrayValue,
              output.count >= modelSize * modelSize * 3 else {
            throw ProcessorError.conversionFailed("DeepCream returned no valid image tensor")
        }
        let outputPtr = output.dataPointer.bindMemory(to: Float.self, capacity: modelSize * modelSize * 3)
        var outputRGBA = [UInt8](repeating: 255, count: modelSize * modelSize * 4)
        for index in 0..<(modelSize * modelSize) {
            for channel in 0..<3 {
                let value = min(max(Double(outputPtr[index * 3 + channel]), 0), 1)
                outputRGBA[index * 4 + channel] = UInt8((value * 255).rounded())
            }
        }
        guard let outputCG = Self.makeCGImage(rgba: outputRGBA, width: modelSize, height: modelSize) else {
            throw ProcessorError.conversionFailed("DeepCream could not decode its predicted crop")
        }
        // The model predicts the entire square context crop. Extract only the
        // target rectangle from that prediction instead of stretching the whole
        // context crop into the marked rectangle.
        let restored = CIImage(cgImage: outputCG)
            .transformed(by: CGAffineTransform(scaleX: CGFloat(side) / CGFloat(modelSize),
                                               y: CGFloat(side) / CGFloat(modelSize)))
            .transformed(by: CGAffineTransform(translationX: crop.minX - target.minX,
                                               y: crop.minY - target.minY))
        guard let targetPatch = context.createCGImage(
            restored,
            from: CGRect(x: 0, y: 0, width: Int(target.width), height: Int(target.height))
        ) else {
            throw ProcessorError.conversionFailed("DeepCream could not extract the reconstructed target region")
        }
        let replacement = try Self.rgbaBytes(from: targetPatch, width: Int(target.width), height: Int(target.height))
        let destination = try Self.allocate(resultPool)
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let src = CVPixelBufferGetBaseAddress(source),
              let dst = CVPixelBufferGetBaseAddress(destination) else {
            throw ProcessorError.conversionFailed("DeepCream composite buffer address unavailable")
        }
        let srcStride = CVPixelBufferGetBytesPerRow(source)
        let dstStride = CVPixelBufferGetBytesPerRow(destination)
        for row in 0..<height {
            dst.advanced(by: row * dstStride).copyMemory(from: src.advanced(by: row * srcStride), byteCount: width * 4)
        }
        let left = max(0, Int(target.minX)), right = min(width, Int(target.maxX))
        let top = max(0, height - Int(target.maxY)), bottom = min(height, height - Int(target.minY))
        let sourceBytes = src.assumingMemoryBound(to: UInt8.self)
        let destBytes = dst.assumingMemoryBound(to: UInt8.self)
        guard right > left, bottom > top else { return destination }
        for row in top..<bottom {
            for col in left..<right {
                if let brushPixels, brushPixels[row * width + col] == 0 { continue }
                let localX = col - left, localY = row - top
                let srcIndex = row * srcStride + col * 4
                let dstIndex = row * dstStride + col * 4
                let patchIndex = (localY * Int(target.width) + localX) * 4
                destBytes[dstIndex] = replacement[patchIndex + 2]
                destBytes[dstIndex + 1] = replacement[patchIndex + 1]
                destBytes[dstIndex + 2] = replacement[patchIndex]
                destBytes[dstIndex + 3] = sourceBytes[srcIndex + 3]
            }
        }
        CVBufferPropagateAttachments(source, destination)
        return destination
    }

    private static func makePool(width: Int, height: Int) throws -> CVPixelBufferPool {
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool) == kCVReturnSuccess,
              let pool else { throw ProcessorError.conversionFailed("could not create DeepCream output buffer pool") }
        return pool
    }

    private static func allocate(_ pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        guard let pool else { throw ProcessorError.conversionFailed("DeepCream output buffer pool unavailable") }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer) == kCVReturnSuccess,
              let buffer else { throw ProcessorError.conversionFailed("could not allocate DeepCream output frame") }
        return buffer
    }

    private static func rgbaBytes(from image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(data: raw.baseAddress, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: width * 4,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else { throw ProcessorError.conversionFailed("DeepCream could not read image pixels") }
        return bytes
    }

    private static func makeCGImage(rgba: [UInt8], width: Int, height: Int) -> CGImage? {
        let data = Data(rgba) as CFData
        guard let provider = CGDataProvider(data: data),
              let color = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: color, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    private static func resize(_ image: CGImage, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let ok = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(data: raw.baseAddress, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: width * 4,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else { return nil }
        let data = Data(bytes) as CFData
        guard let provider = CGDataProvider(data: data),
              let color = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: color,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
