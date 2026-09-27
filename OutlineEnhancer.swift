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
            case .modelMissing:
                return "Anime Sharpie Outline Subtle model is missing from the app bundle."
            case .unsupportedPixelFormat:
                return "Outline model requires a BGRA/RGBA source frame."
            case .allocationFailed:
                return "Could not allocate an outline output frame."
            case .badModelInterface(let message):
                return "Outline model interface error: \(message)"
            }
        }
    }

    private let tile = 256
    private let shrink = 20
    private let inputName = "x"
    private let outputName = "var_244"

    // Final subtle-width correction requested after the original Subtle model:
    // keep almost all of the learned output, but mix a tiny amount of the
    // cleaned source back in so the added ink reads just a little narrower.
    private let modelBlend: Float32 = 0.94
    private let sourceBlend: Float32 = 0.06

    private let model: MLModel
    private let inputArray: MLMultiArray

    init() throws {
        guard let packageURL = Bundle.main.url(
            forResource: "AnimeSharpieOutlineSubtle",
            withExtension: "mlpackage"
        ) else {
            throw OutlineError.modelMissing
        }

        let fm = FileManager.default
        let cacheRoot = try fm.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("SharpieCoreML", isDirectory: true)
        try fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let cachedCompiled = cacheRoot.appendingPathComponent("AnimeSharpieOutlineSubtle.mlmodelc", isDirectory: true)

        let modelURL: URL
        if fm.fileExists(atPath: cachedCompiled.path) {
            modelURL = cachedCompiled
        } else {
            let temporaryCompiled = try MLModel.compileModel(at: packageURL)
            if fm.fileExists(atPath: cachedCompiled.path) {
                try? fm.removeItem(at: cachedCompiled)
            }
            do {
                try fm.copyItem(at: temporaryCompiled, to: cachedCompiled)
                modelURL = cachedCompiled
            } catch {
                modelURL = temporaryCompiled
            }
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.inputArray = try MLMultiArray(
            shape: [1, 3, 256, 256],
            dataType: .float32
        )
    }

    func enhance(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let format = CVPixelBufferGetPixelFormatType(source)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA else {
            throw OutlineError.unsupportedPixelFormat
        }

        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var destination: CVPixelBuffer?
        let result = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &destination
        )
        guard result == kCVReturnSuccess, let destination else {
            throw OutlineError.allocationFailed
        }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }

        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination) else {
            throw OutlineError.allocationFailed
        }

        guard inputArray.dataType == .float32 else {
            throw OutlineError.badModelInterface("expected Float32 input")
        }

        let src = sourceBase.assumingMemoryBound(to: UInt8.self)
        let dst = destinationBase.assumingMemoryBound(to: UInt8.self)
        let srcRow = CVPixelBufferGetBytesPerRow(source)
        let dstRow = CVPixelBufferGetBytesPerRow(destination)
        let core = tile - (shrink * 2)

        let inPtr = inputArray.dataPointer.assumingMemoryBound(to: Float32.self)
        let inS0 = inputArray.strides[0].intValue
        let inS1 = inputArray.strides[1].intValue
        let inS2 = inputArray.strides[2].intValue
        let inS3 = inputArray.strides[3].intValue

        func inputIndex(_ c: Int, _ y: Int, _ x: Int) -> Int {
            0 * inS0 + c * inS1 + y * inS2 + x * inS3
        }

        for coreY in stride(from: 0, to: height, by: core) {
            for coreX in stride(from: 0, to: width, by: core) {
                autoreleasepool {
                    for y in 0..<tile {
                        let sy = min(max(coreY + y - shrink, 0), height - 1)
                        let rowOffset = sy * srcRow
                        for x in 0..<tile {
                            let sx = min(max(coreX + x - shrink, 0), width - 1)
                            let i = rowOffset + sx * 4
                            let r: Float32
                            let g: Float32
                            let b: Float32
                            if format == kCVPixelFormatType_32BGRA {
                                b = Float32(src[i]) / 255.0
                                g = Float32(src[i + 1]) / 255.0
                                r = Float32(src[i + 2]) / 255.0
                            } else {
                                r = Float32(src[i]) / 255.0
                                g = Float32(src[i + 1]) / 255.0
                                b = Float32(src[i + 2]) / 255.0
                            }
                            inPtr[inputIndex(0, y, x)] = r
                            inPtr[inputIndex(1, y, x)] = g
                            inPtr[inputIndex(2, y, x)] = b
                        }
                    }
                }

                let predicted: MLMultiArray = try autoreleasepool {
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        inputName: MLFeatureValue(multiArray: inputArray)
                    ])
                    let output = try model.prediction(from: provider)
                    guard let array = output.featureValue(for: outputName)?.multiArrayValue else {
                        throw OutlineError.badModelInterface("missing output \(outputName)")
                    }
                    return array
                }

                guard predicted.dataType == .float32, predicted.shape.count == 4 else {
                    throw OutlineError.badModelInterface("expected Float32 NCHW output")
                }

                let outPtr = predicted.dataPointer.assumingMemoryBound(to: Float32.self)
                let oS0 = predicted.strides[0].intValue
                let oS1 = predicted.strides[1].intValue
                let oS2 = predicted.strides[2].intValue
                let oS3 = predicted.strides[3].intValue
                func outputIndex(_ c: Int, _ y: Int, _ x: Int) -> Int {
                    0 * oS0 + c * oS1 + y * oS2 + x * oS3
                }

                let copyWidth = min(core, width - coreX)
                let copyHeight = min(core, height - coreY)
                for y in 0..<copyHeight {
                    let dy = coreY + y
                    let oy = y + shrink
                    let srcRowOffset = dy * srcRow
                    let dstRowOffset = dy * dstRow
                    for x in 0..<copyWidth {
                        let dx = coreX + x
                        let ox = x + shrink
                        let si = srcRowOffset + dx * 4
                        let di = dstRowOffset + dx * 4

                        let modelR = min(max(outPtr[outputIndex(0, oy, ox)], 0), 1)
                        let modelG = min(max(outPtr[outputIndex(1, oy, ox)], 0), 1)
                        let modelB = min(max(outPtr[outputIndex(2, oy, ox)], 0), 1)

                        let srcR: Float32
                        let srcG: Float32
                        let srcB: Float32
                        if format == kCVPixelFormatType_32BGRA {
                            srcB = Float32(src[si]) / 255.0
                            srcG = Float32(src[si + 1]) / 255.0
                            srcR = Float32(src[si + 2]) / 255.0
                        } else {
                            srcR = Float32(src[si]) / 255.0
                            srcG = Float32(src[si + 1]) / 255.0
                            srcB = Float32(src[si + 2]) / 255.0
                        }

                        let r = modelR * modelBlend + srcR * sourceBlend
                        let g = modelG * modelBlend + srcG * sourceBlend
                        let b = modelB * modelBlend + srcB * sourceBlend

                        dst[di] = UInt8((min(max(b, 0), 1) * 255.0).rounded())
                        dst[di + 1] = UInt8((min(max(g, 0), 1) * 255.0).rounded())
                        dst[di + 2] = UInt8((min(max(r, 0), 1) * 255.0).rounded())
                        dst[di + 3] = 255
                    }
                }
            }
        }

        return destination
    }
}
