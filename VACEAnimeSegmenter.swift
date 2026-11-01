import CoreML
import CoreGraphics
import Foundation

/// SkyTNT / ISNet anime character segmenter. Apache-2.0 weights, converted to
/// Core ML and run on the device Neural Engine or GPU. Input is RGB / 255,
/// letterboxed to 1024 the same way the published inference script does.
enum VACEAnimeSegmenter {
    static let inputSide = 1024
    private static let lock = NSLock()
    private static var model: MLModel?
    private static var didLogMissing = false
    private static var didLogLoad = false

    /// Full-frame confidence, 0...255. Nil when the package is not in the app.
    static func confidence(cgImage: CGImage) throws -> [UInt8]? {
        lock.lock()
        defer { lock.unlock() }
        let loaded = try loadModelUnlocked()
        guard let loaded else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        let fitted = try letterbox(cgImage)
        let output = try predict(model: loaded, rgb: fitted.rgb)
        return try cropAndScale(
            output,
            padX: fitted.padX,
            padY: fitted.padY,
            contentWidth: fitted.contentWidth,
            contentHeight: fitted.contentHeight,
            targetWidth: width,
            targetHeight: height
        )
    }

    private static func loadModelUnlocked() throws -> MLModel? {
        if let model { return model }
        guard let url = Bundle.main.url(forResource: "VACEAnimeSeg", withExtension: "mlmodelc") else {
            if !didLogMissing {
                didLogMissing = true
                CrashCapture.step("VACE anime segmenter missing from the app bundle")
            }
            return nil
        }
        let options = MLModelConfiguration()
        options.computeUnits = .all
        let loaded = try MLModel(contentsOf: url, configuration: options)
        model = loaded
        if !didLogLoad {
            didLogLoad = true
            CrashCapture.step("VACE anime ISNet loaded computeUnits=all")
        }
        return loaded
    }

    private static func letterbox(_ image: CGImage) throws -> (rgb: [Float], padX: Int, padY: Int, contentWidth: Int, contentHeight: Int) {
        let width = image.width
        let height = image.height
        let contentWidth: Int
        let contentHeight: Int
        if height > width {
            contentHeight = inputSide
            contentWidth = max(1, inputSide * width / height)
        } else {
            contentWidth = inputSide
            contentHeight = max(1, inputSide * height / max(width, 1))
        }
        let padX = (inputSide - contentWidth) / 2
        let padY = (inputSide - contentHeight) / 2
        guard let context = CGContext(
            data: nil,
            width: contentWidth,
            height: contentHeight,
            bitsPerComponent: 8,
            bytesPerRow: contentWidth * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw VACESubjectSegmentationError.visionFailed("Could not allocate the anime segmenter input.")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: contentWidth, height: contentHeight))
        guard let data = context.data else {
            throw VACESubjectSegmentationError.visionFailed("Could not read the anime segmenter input.")
        }
        let source = data.bindMemory(to: UInt8.self, capacity: contentWidth * contentHeight * 4)
        var rgb = [Float](repeating: 0, count: 3 * inputSide * inputSide)
        let plane = inputSide * inputSide
        for y in 0..<contentHeight {
            for x in 0..<contentWidth {
                let sourceOffset = (y * contentWidth + x) * 4
                let destination = (padY + y) * inputSide + (padX + x)
                rgb[destination] = Float(source[sourceOffset]) / 255
                rgb[plane + destination] = Float(source[sourceOffset + 1]) / 255
                rgb[plane * 2 + destination] = Float(source[sourceOffset + 2]) / 255
            }
        }
        return (rgb, padX, padY, contentWidth, contentHeight)
    }

    private static func predict(model: MLModel, rgb: [Float]) throws -> [Float] {
        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: inputSide), NSNumber(value: inputSide)], dataType: .float32)
        let expected = rgb.count
        guard array.count == expected else {
            throw VACESubjectSegmentationError.visionFailed("Anime segmenter input length did not match 1024×1024.")
        }
        rgb.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            array.dataPointer.bindMemory(to: Float.self, capacity: expected).update(from: base, count: expected)
        }
        guard let inputName = model.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .multiArray })?.key else {
            throw VACESubjectSegmentationError.visionFailed("Anime segmenter has no tensor input.")
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [inputName: array])
        let result = try model.prediction(from: features)
        let output = try maskArray(in: result, model: model)
        let count = output.count
        var values = [Float](repeating: 0, count: count)
        let strides = output.strides.map { $0.intValue }
        if output.dataType == .float16 {
            let pointer = output.dataPointer.bindMemory(to: Float16.self, capacity: count)
            for index in 0..<count { values[index] = Float(pointer[strided(index, shape: output.shape, strides: strides)]) }
        } else {
            let pointer = output.dataPointer.bindMemory(to: Float.self, capacity: count)
            for index in 0..<count { values[index] = pointer[strided(index, shape: output.shape, strides: strides)] }
        }
        let shape = output.shape.map(\.intValue)
        var minimum = Float.greatestFiniteMagnitude
        var maximum = -Float.greatestFiniteMagnitude
        var sum = Float(0)
        for value in values {
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            sum += value
        }
        let mean = values.isEmpty ? 0 : sum / Float(values.count)
        if maximum > 1.5 || minimum < -0.2 {
            for index in values.indices {
                let logit = values[index]
                values[index] = 1 / (1 + exp(-logit))
            }
            CrashCapture.step(String(format: "VACE anime output looked like logits min %.3f max %.3f; applied sigmoid", minimum, maximum))
        }
        let active = values.reduce(0) { $0 + ($1 > 0.5 ? 1 : 0) }
        let rawCoverage = values.isEmpty ? 0 : Double(active) / Double(values.count)
        CrashCapture.step(String(
            format: "VACE anime run shape %@ min %.3f max %.3f mean %.3f rawCoverage %.1f%%",
            shape.map(String.init).joined(separator: "x"),
            minimum, maximum, mean, rawCoverage * 100
        ))
        return values
    }

    /// Picks the spatial mask plane when Core ML exposes more than one output.
    private static func maskArray(in result: MLFeatureProvider, model: MLModel) throws -> MLMultiArray {
        var chosen: MLMultiArray?
        var chosenArea = 0
        for name in model.modelDescription.outputDescriptionsByName.keys.sorted() {
            guard let array = result.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = array.shape.map(\.intValue)
            let area = shape.suffix(2).reduce(1, *)
            if area > chosenArea {
                chosen = array
                chosenArea = area
            }
        }
        guard let chosen else {
            throw VACESubjectSegmentationError.visionFailed("Anime segmenter returned no mask.")
        }
        return chosen
    }

    private static func strided(_ index: Int, shape: [NSNumber], strides: [Int]) -> Int {
        guard !strides.isEmpty, strides.count == shape.count else { return index }
        var remainder = index
        var offset = 0
        for dimension in stride(from: shape.count - 1, through: 0, by: -1) {
            let size = max(shape[dimension].intValue, 1)
            let coordinate = remainder % size
            remainder /= size
            offset += coordinate * strides[dimension]
        }
        return offset
    }

    private static func cropAndScale(
        _ values: [Float],
        padX: Int,
        padY: Int,
        contentWidth: Int,
        contentHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) throws -> [UInt8] {
        guard values.count >= inputSide * inputSide else {
            throw VACESubjectSegmentationError.visionFailed("Anime segmenter mask was shorter than 1024×1024.")
        }
        var cropped = [UInt8](repeating: 0, count: contentWidth * contentHeight)
        for y in 0..<contentHeight {
            for x in 0..<contentWidth {
                let sample = min(max(values[(padY + y) * inputSide + (padX + x)], 0), 1)
                cropped[y * contentWidth + x] = UInt8(sample * 255)
            }
        }
        if contentWidth == targetWidth && contentHeight == targetHeight { return cropped }
        guard let provider = CGDataProvider(data: Data(cropped) as CFData),
              let image = CGImage(
                width: contentWidth, height: contentHeight,
                bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: contentWidth,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
              ),
              let context = CGContext(
                data: nil, width: targetWidth, height: targetHeight,
                bitsPerComponent: 8, bytesPerRow: targetWidth,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
              ) else {
            throw VACESubjectSegmentationError.visionFailed("Could not scale the anime mask to the frame.")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        guard let scaled = context.data else {
            throw VACESubjectSegmentationError.visionFailed("Could not read the scaled anime mask.")
        }
        return Array(UnsafeBufferPointer(start: scaled.bindMemory(to: UInt8.self, capacity: targetWidth * targetHeight), count: targetWidth * targetHeight))
    }
}
