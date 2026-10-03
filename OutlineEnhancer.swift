import Foundation
import CoreML
import CoreVideo
import Vision

final class OutlineEnhancer {
    enum OutlineError: LocalizedError {
        case modelMissing
        case unsupportedPixelFormat
        case allocationFailed
        case badModelInterface(String)

        var errorDescription: String? {
            switch self {
            case .modelMissing: return "Anime Sharpie Outline Subtle model is missing from the app bundle."
            case .unsupportedPixelFormat: return "Outline model requires a BGRA/RGBA source frame."
            case .allocationFailed: return "Could not allocate an outline output frame."
            case .badModelInterface(let message): return "Outline model interface error: \(message)"
            }
        }
    }

    private let tile = 256
    private let shrink = 20
    private let inputName = "x"
    private let outputName = "var_244"
    private let modelBlend: Float32 = 0.94
    private let sourceBlend: Float32 = 0.06

    private let model: MLModel
    private static let maxWorkers = 3
    private let inputArrays: [MLMultiArray]
    private var outputPool: CVPixelBufferPool?
    private var outputPoolSize: (Int, Int) = (0, 0)

    // Content-aware outline gate. Vision's person matte is only one cue: strong
    // structural edges are still allowed in the background (tables, chairs,
    // architecture, etc.), while softer painterly texture is attenuated.
    private let segmentationHandler = VNSequenceRequestHandler()
    private let personSegmentationRequest: VNGeneratePersonSegmentationRequest

    init() throws {
        let modelURL = try Self.compiledModelURL()
        let configuration = MLModelConfiguration()
        // .all lets Core ML select ANE/GPU/CPU per operation instead of forbidding GPU.
        // This does not alter weights or model math.
        configuration.computeUnits = ModelComputePreference.current.units
        configuration.allowLowPrecisionAccumulationOnGPU = true
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        var arrays: [MLMultiArray] = []
        for _ in 0..<Self.maxWorkers {
            arrays.append(try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32))
        }
        self.inputArrays = arrays
        let segmentation = VNGeneratePersonSegmentationRequest()
        segmentation.qualityLevel = .fast
        segmentation.outputPixelFormat = kCVPixelFormatType_OneComponent8
        self.personSegmentationRequest = segmentation
    }

    static func compiledModelURL() throws -> URL {
        guard let packageURL = Bundle.main.url(forResource: "AnimeSharpieOutlineSubtle", withExtension: "mlpackage") else {
            throw OutlineError.modelMissing
        }
        let fm = FileManager.default
        let cacheRoot = try fm.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("SharpieCoreML", isDirectory: true)
        try fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let cachedCompiled = cacheRoot.appendingPathComponent("AnimeSharpieOutlineSubtle.mlmodelc", isDirectory: true)
        let modelURL: URL
        if fm.fileExists(atPath: cachedCompiled.path) {
            modelURL = cachedCompiled
        } else {
            let temporaryCompiled = try MLModel.compileModel(at: packageURL)
            do {
                try fm.copyItem(at: temporaryCompiled, to: cachedCompiled)
                modelURL = cachedCompiled
            } catch {
                modelURL = temporaryCompiled
            }
        }
        return modelURL
    }

    private func destinationBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        if outputPool == nil || outputPoolSize != (width, height) {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
            var pool: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess, let pool else {
                throw OutlineError.allocationFailed
            }
            outputPool = pool
            outputPoolSize = (width, height)
        }
        var destination: CVPixelBuffer?
        guard let outputPool,
              CVPixelBufferPoolCreatePixelBuffer(nil, outputPool, &destination) == kCVReturnSuccess,
              let destination else { throw OutlineError.allocationFailed }
        return destination
    }

    func enhance(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let format = CVPixelBufferGetPixelFormatType(source)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA else { throw OutlineError.unsupportedPixelFormat }
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let destination = try destinationBuffer(width: width, height: height)

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source), let destinationBase = CVPixelBufferGetBaseAddress(destination) else { throw OutlineError.allocationFailed }
        guard inputArrays.allSatisfy({ $0.dataType == .float32 }) else { throw OutlineError.badModelInterface("expected Float32 input") }

        let src = sourceBase.assumingMemoryBound(to: UInt8.self)
        let dst = destinationBase.assumingMemoryBound(to: UInt8.self)
        let srcRow = CVPixelBufferGetBytesPerRow(source)
        let dstRow = CVPixelBufferGetBytesPerRow(destination)
        let isBGRA = format == kCVPixelFormatType_32BGRA
        let core = tile - shrink * 2
        let tilesAcross = (width + core - 1) / core
        let tilesDown = (height + core - 1) / core
        let tileCount = tilesAcross * tilesDown

        // Tiles write to disjoint core regions of the destination and only READ the source, so they
        // can run in parallel. Previously ~120 tiles per 4K frame ran strictly one after another,
        // alternating scalar CPU loops and a synchronous Core ML call, leaving the CPU idle while the
        // model ran and the model idle while the CPU ran. Worker count follows the memory governor.
        let tier = currentRenderPerformanceSnapshot().tier
        let desiredWorkers: Int
        switch tier {
        case .performance: desiredWorkers = 3
        case .balanced: desiredWorkers = 2
        case .safe: desiredWorkers = 1
        }
        let cpuLimit = max(ProcessInfo.processInfo.activeProcessorCount - 1, 1)
        let workers = max(1, min(desiredWorkers, inputArrays.count, tileCount, cpuLimit))
        let failure = TileFailure()

        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let input = inputArrays[worker]
            var tileIndex = worker
            while tileIndex < tileCount {
                if failure.hasError { return }
                let coreX = (tileIndex % tilesAcross) * core
                let coreY = (tileIndex / tilesAcross) * core
                do {
                    try processTile(
                        coreX: coreX, coreY: coreY, width: width, height: height,
                        src: src, dst: dst, srcRow: srcRow, dstRow: dstRow,
                        isBGRA: isBGRA, input: input
                    )
                } catch {
                    failure.record(error)
                    return
                }
                tileIndex += workers
            }
        }
        try failure.rethrowIfNeeded()

        // The Sharpie model is intentionally allowed to draw on both characters
        // and hard-edged objects. This final gate only changes how much of those
        // model-generated differences are retained. It uses a soft person matte
        // when Vision can find one, plus a strong source-edge test for background
        // structures. This avoids the old binary "character vs background" idea:
        // a table corner can still receive an outline, while grass/paint texture
        // needs substantially stronger source structure before it survives.
        refineContentAwareOutlines(source: source, destination: destination)
        return destination
    }

    private func refineContentAwareOutlines(source: CVPixelBuffer, destination: CVPixelBuffer) {
        let mask: CVPixelBuffer?
        do {
            try segmentationHandler.perform([personSegmentationRequest], on: source)
            mask = personSegmentationRequest.results?.first?.pixelBuffer
        } catch {
            mask = nil
        }

        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        guard width > 2, height > 2 else { return }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        if let mask { CVPixelBufferLockBaseAddress(mask, .readOnly) }
        defer {
            if let mask { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(source),
              let dstBase = CVPixelBufferGetBaseAddress(destination) else { return }
        let src = srcBase.assumingMemoryBound(to: UInt8.self)
        let dst = dstBase.assumingMemoryBound(to: UInt8.self)
        let srcRow = CVPixelBufferGetBytesPerRow(source)
        let dstRow = CVPixelBufferGetBytesPerRow(destination)
        let srcFormat = CVPixelBufferGetPixelFormatType(source)
        let dstFormat = CVPixelBufferGetPixelFormatType(destination)
        let srcBGRA = srcFormat == kCVPixelFormatType_32BGRA
        let dstBGRA = dstFormat == kCVPixelFormatType_32BGRA

        let maskBase = mask.flatMap { CVPixelBufferGetBaseAddress($0) }
        let maskPtr = maskBase?.assumingMemoryBound(to: UInt8.self)
        let maskRow = mask.map(CVPixelBufferGetBytesPerRow) ?? 0
        let maskWidth = mask.map(CVPixelBufferGetWidth) ?? 0
        let maskHeight = mask.map(CVPixelBufferGetHeight) ?? 0

        @inline(__always) func clamp01(_ v: Double) -> Double { min(max(v, 0), 1) }
        @inline(__always) func smooth(_ a: Double, _ b: Double, _ x: Double) -> Double {
            let t = clamp01((x - a) / max(b - a, 0.000001))
            return t * t * (3 - 2 * t)
        }
        @inline(__always) func lumaAt(_ x: Int, _ y: Int) -> Double {
            let xx = min(max(x, 0), width - 1)
            let yy = min(max(y, 0), height - 1)
            let i = yy * srcRow + xx * 4
            let rOff = srcBGRA ? 2 : 0
            let bOff = srcBGRA ? 0 : 2
            let r = Double(src[i + rOff]) / 255.0
            let g = Double(src[i + 1]) / 255.0
            let b = Double(src[i + bOff]) / 255.0
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        }
        @inline(__always) func personAt(_ x: Int, _ y: Int) -> Double {
            guard let maskPtr, maskWidth > 0, maskHeight > 0 else { return 0 }
            let mx = min(maskWidth - 1, max(0, Int((Double(x) + 0.5) * Double(maskWidth) / Double(width))))
            let my = min(maskHeight - 1, max(0, Int((Double(y) + 0.5) * Double(maskHeight) / Double(height))))
            return Double(maskPtr[my * maskRow + mx]) / 255.0
        }

        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * srcRow + x * 4
                let di = y * dstRow + x * 4
                let rOff = srcBGRA ? 2 : 0
                let bOff = srcBGRA ? 0 : 2
                let drOff = dstBGRA ? 2 : 0
                let dbOff = dstBGRA ? 0 : 2

                let sr = Double(src[i + rOff]) / 255.0
                let sg = Double(src[i + 1]) / 255.0
                let sb = Double(src[i + bOff]) / 255.0
                let er = Double(dst[di + drOff]) / 255.0
                let eg = Double(dst[di + 1]) / 255.0
                let eb = Double(dst[di + dbOff]) / 255.0

                let sourceDelta = abs(er - sr) + abs(eg - sg) + abs(eb - sb)
                let modelDelta = sourceDelta / 3.0
                guard modelDelta > 0.012 else { continue }

                let gx = abs(lumaAt(x + 1, y) - lumaAt(x - 1, y))
                let gy = abs(lumaAt(x, y + 1) - lumaAt(x, y - 1))
                let sourceEdge = max(gx, gy)
                let structural = smooth(0.055, 0.18, sourceEdge)
                let strongStructural = smooth(0.11, 0.24, sourceEdge)
                let person = personAt(x, y)

                // Characters get a lower edge threshold so internal clothing,
                // hair and line-art survive even when their source contrast is
                // modest. Background objects still qualify when their edge is
                // genuinely sharp/structural.
                var keep = max(
                    person * (0.72 + 0.28 * structural),
                    (1.0 - person) * strongStructural
                )

                // Painterly grass/brush texture is usually softer than a hard
                // object boundary. Keep a little of it rather than deleting it
                // completely, but make the model work much harder to add a line.
                if person < 0.35 {
                    keep *= 0.70 + 0.30 * strongStructural
                }

                // Sweat/water/fluids often appear as bright, low-chroma, thin
                // contours. They are visually different from clothing/hair line
                // art, so reduce their added outline without suppressing colorful
                // character brush strokes.
                let sourceLuma = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb
                let chroma = max(sr, max(sg, sb)) - min(sr, min(sg, sb))
                let fluidLike = smooth(0.60, 0.86, sourceLuma)
                    * (1.0 - smooth(0.10, 0.22, chroma))
                    * (1.0 - smooth(0.18, 0.32, sourceEdge))
                    * smooth(0.35, 0.65, person)
                keep *= (1.0 - 0.55 * fluidLike)

                // Don't let very subtle model deltas become a visible painted
                // contour. Stronger generated outlines remain almost untouched.
                let deltaGate = smooth(0.012, 0.055, modelDelta)
                keep = clamp01(keep * deltaGate)

                // The model's enhanced frame is blended back toward the source
                // according to the content-aware confidence. This is intentionally
                // soft, so there is no visible halo at the subject boundary.
                let amount = Float32(clamp01(keep))
                dst[di + drOff] = UInt8((sr + (er - sr) * Double(amount)) * 255.0)
                dst[di + 1] = UInt8((sg + (eg - sg) * Double(amount)) * 255.0)
                dst[di + dbOff] = UInt8((sb + (eb - sb) * Double(amount)) * 255.0)
            }
        }
    }

    private func processTile(
        coreX: Int, coreY: Int, width: Int, height: Int,
        src: UnsafeMutablePointer<UInt8>, dst: UnsafeMutablePointer<UInt8>,
        srcRow: Int, dstRow: Int, isBGRA: Bool, input: MLMultiArray
    ) throws {
        let core = tile - shrink * 2
        let rOff = isBGRA ? 2 : 0
        let bOff = isBGRA ? 0 : 2
        let inv255: Float32 = 1.0 / 255.0

        let inPtr = input.dataPointer.assumingMemoryBound(to: Float32.self)
        let inS1 = input.strides[1].intValue
        let inS2 = input.strides[2].intValue
        let inS3 = input.strides[3].intValue

        for y in 0..<tile {
            let sy = min(max(coreY + y - shrink, 0), height - 1)
            let rowOffset = sy * srcRow
            let planeY = y * inS2
            for x in 0..<tile {
                let sx = min(max(coreX + x - shrink, 0), width - 1)
                let i = rowOffset + sx * 4
                let idx = planeY + x * inS3
                inPtr[idx] = Float32(src[i + rOff]) * inv255
                inPtr[inS1 + idx] = Float32(src[i + 1]) * inv255
                inPtr[2 * inS1 + idx] = Float32(src[i + bOff]) * inv255
            }
        }

        let predicted: MLMultiArray = try autoreleasepool {
            let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
            let output = try model.prediction(from: provider)
            guard let array = output.featureValue(for: outputName)?.multiArrayValue else { throw OutlineError.badModelInterface("missing output \(outputName)") }
            return array
        }
        guard predicted.dataType == .float32, predicted.shape.count == 4 else { throw OutlineError.badModelInterface("expected Float32 NCHW output") }

        let outPtr = predicted.dataPointer.assumingMemoryBound(to: Float32.self)
        let oS1 = predicted.strides[1].intValue
        let oS2 = predicted.strides[2].intValue
        let oS3 = predicted.strides[3].intValue
        let copyWidth = min(core, width - coreX)
        let copyHeight = min(core, height - coreY)
        for y in 0..<copyHeight {
            let dy = coreY + y
            let oy = y + shrink
            let srcRowOffset = dy * srcRow
            let dstRowOffset = dy * dstRow
            let outRow = oy * oS2
            for x in 0..<copyWidth {
                let dx = coreX + x
                let ox = x + shrink
                let si = srcRowOffset + dx * 4
                let di = dstRowOffset + dx * 4
                let oi = outRow + ox * oS3
                let mr = min(max(outPtr[oi], 0), 1)
                let mg = min(max(outPtr[oS1 + oi], 0), 1)
                let mb = min(max(outPtr[2 * oS1 + oi], 0), 1)
                let sr = Float32(src[si + rOff]) * inv255
                let sg = Float32(src[si + 1]) * inv255
                let sb = Float32(src[si + bOff]) * inv255
                let r = mr * modelBlend + sr * sourceBlend
                let g = mg * modelBlend + sg * sourceBlend
                let b = mb * modelBlend + sb * sourceBlend
                dst[di] = UInt8((min(max(b, 0), 1) * 255).rounded())
                dst[di + 1] = UInt8((min(max(g, 0), 1) * 255).rounded())
                dst[di + 2] = UInt8((min(max(r, 0), 1) * 255).rounded())
                dst[di + 3] = 255
            }
        }
    }
}

private final class TileFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Error?

    var hasError: Bool {
        lock.lock(); defer { lock.unlock() }
        return stored != nil
    }

    func record(_ error: Error) {
        lock.lock()
        if stored == nil { stored = error }
        lock.unlock()
    }

    func rethrowIfNeeded() throws {
        lock.lock()
        let error = stored
        lock.unlock()
        if let error { throw error }
    }
}
