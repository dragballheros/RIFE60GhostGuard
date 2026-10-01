import Foundation
import CoreML
import CoreVideo

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

    init() throws {
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
        let configuration = MLModelConfiguration()
        // .all lets Core ML select ANE/GPU/CPU per operation instead of forbidding GPU.
        // This does not alter weights or model math.
        configuration.computeUnits = .all
        configuration.allowLowPrecisionAccumulationOnGPU = true
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        var arrays: [MLMultiArray] = []
        for _ in 0..<Self.maxWorkers {
            arrays.append(try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32))
        }
        self.inputArrays = arrays
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
        return destination
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
