import Accelerate
import CoreML
import CoreVideo
import Foundation

extension AnimeRestoreRegistry {
    static func installDefault() {
        guard factory == nil else { return }
        factory = { try AnimeRestoreTiledModel(variant: .apisr) }
        displayName = "APISR / SPAN"
    }
}

enum AnimeRestoreFiles {
    static let remoteRoot = "https://raw.githubusercontent.com/dragballheros/RIFE60GhostGuard/cursor/rife60-quality-speed-h3-1273/Models/AnimeRestore"

    static var cacheDirectoryOverride: URL?
    static func cacheDirectory() throws -> URL {
        if let cacheDirectoryOverride { return cacheDirectoryOverride }
        let base = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let directory = base.appendingPathComponent("AnimeRestore", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func packageURL(_ variant: AnimeRestoreVariant) throws -> URL {
        try cacheDirectory().appendingPathComponent(variant.packageDirectoryName, isDirectory: true)
    }

    static func ensurePackage(_ variant: AnimeRestoreVariant) throws -> URL {
        let package = try packageURL(variant)
        if FileManager.default.fileExists(atPath: package.appendingPathComponent("Manifest.json").path) {
            return package
        }
        let zip = try cacheDirectory().appendingPathComponent(variant.zipName)
        if !FileManager.default.fileExists(atPath: zip.path) {
            guard let remote = URL(string: "\(remoteRoot)/\(variant.zipName)") else {
                throw ProcessorError.conversionFailed("Anime Restore download URL is invalid")
            }
            DiagnosticsLogger.shared.log("Anime Restore downloading \(variant.title)")
            CrashCapture.step("Anime Restore downloading \(variant.title)")
            let semaphore = DispatchSemaphore(value: 0)
            var failure: Error?
            let task = URLSession.shared.downloadTask(with: remote) { location, response, error in
                defer { semaphore.signal() }
                if let error { failure = error; return }
                guard let location, let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    failure = ProcessorError.conversionFailed("Anime Restore download failed for \(variant.title)")
                    return
                }
                try? FileManager.default.removeItem(at: zip)
                do { try FileManager.default.moveItem(at: location, to: zip) }
                catch { failure = error }
            }
            task.resume()
            semaphore.wait()
            if let failure { throw failure }
        }
        try StoredZip.extract(zip, to: try cacheDirectory())
        guard FileManager.default.fileExists(atPath: package.appendingPathComponent("Manifest.json").path) else {
            throw ProcessorError.conversionFailed("Anime Restore package \(variant.title) did not unpack")
        }
        return package
    }
}

enum StoredZip {
    static func extract(_ zip: URL, to directory: URL) throws {
        let data = try Data(contentsOf: zip)
        var offset = 0
        while offset + 30 <= data.count {
            let signature = readU32(data, offset)
            if signature != 0x04034b50 { break }
            let method = readU16(data, offset + 8)
            let compressed = Int(readU32(data, offset + 18))
            let nameLength = Int(readU16(data, offset + 26))
            let extraLength = Int(readU16(data, offset + 28))
            let nameStart = offset + 30
            let dataStart = nameStart + nameLength + extraLength
            guard dataStart + compressed <= data.count, method == 0 else {
                throw ProcessorError.conversionFailed("Anime Restore package zip is not a stored archive")
            }
            let name = String(data: data.subdata(in: nameStart..<(nameStart + nameLength)), encoding: .utf8) ?? ""
            let payload = data.subdata(in: dataStart..<(dataStart + compressed))
            if !name.isEmpty && !name.hasSuffix("/") {
                let destination = directory.appendingPathComponent(name)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try payload.write(to: destination, options: .atomic)
            }
            offset = dataStart + compressed
        }
    }

    private static func readU16(_ data: Data, _ offset: Int) -> UInt16 {
        UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }
    private static func readU32(_ data: Data, _ offset: Int) -> UInt32 {
        UInt32(data[offset]) | (UInt32(data[offset + 1]) << 8) | (UInt32(data[offset + 2]) << 16) | (UInt32(data[offset + 3]) << 24)
    }
}

final class AnimeRestoreTiledModel: AnimeRestoreModel {
    static let tile = 256
    static let overlap = 16
    private let model: MLModel
    private let inputArray: MLMultiArray
    let variant: AnimeRestoreVariant

    init(variant: AnimeRestoreVariant, units: MLComputeUnits = ModelComputePreference.current.units) throws {
        self.variant = variant
        let package = try AnimeRestoreFiles.ensurePackage(variant)
        let compiled = try MLModel.compileModel(at: package)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        configuration.allowLowPrecisionAccumulationOnGPU = true
        model = try MLModel(contentsOf: compiled, configuration: configuration)
        inputArray = try MLMultiArray(shape: [1, 3, NSNumber(value: Self.tile), NSNumber(value: Self.tile)], dataType: .float32)
    }

    private(set) var lastStep = "idle"
    private(set) var tilesFinished = 0
    private(set) var tileCount = 0

    func release() {}

    func restore(_ frame: CVPixelBuffer) throws -> CVPixelBuffer {
        try restore(frame, deadline: CFAbsoluteTimeGetCurrent() + 15, onTile: nil)
    }

    /// Each 256 source tile is upscaled to 512, scaled back to 256, then stitched.
    /// The full 2× frame is never allocated.
    func restore(_ frame: CVPixelBuffer, deadline: CFAbsoluteTime, onTile: ((Int, Int) -> Void)? = nil) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(frame)
        let height = CVPixelBufferGetHeight(frame)
        let xs = Self.origins(width)
        let ys = Self.origins(height)
        tileCount = xs.count * ys.count
        tilesFinished = 0
        lastStep = "tile-plan \(tileCount)"
        var canvas = [Float](repeating: 0, count: width * height * 3)
        var weights = [Float](repeating: 0, count: width * height)
        for originY in ys {
            for originX in xs {
                if CFAbsoluteTimeGetCurrent() >= deadline {
                    throw AnimeRestoreStall(variant: variant.title, tilesFinished: tilesFinished, tileCount: tileCount, lastStep: lastStep)
                }
                let tile = try predictSourceTile(frame: frame, originX: originX, originY: originY, width: width, height: height, deadline: deadline)
                Self.blend(tile: tile, into: &canvas, weights: &weights, originX: originX, originY: originY, width: width, height: height)
                tilesFinished += 1
                lastStep = "stitched \(tilesFinished)/\(tileCount)"
                onTile?(tilesFinished, tileCount)
            }
        }
        lastStep = "pack"
        return try Self.pack(canvas: canvas, weights: weights, width: width, height: height)
    }

    func milliseconds(forTileCount count: Int, frame: CVPixelBuffer) throws -> Double {
        let xs = Self.origins(CVPixelBufferGetWidth(frame))
        let ys = Self.origins(CVPixelBufferGetHeight(frame))
        guard let originX = xs.first, let originY = ys.first else { return 0 }
        let samples = min(max(count, 1), 4)
        let started = CFAbsoluteTimeGetCurrent()
        let deadline = CFAbsoluteTimeGetCurrent() + 15
        for _ in 0..<samples {
            _ = try predictSourceTile(frame: frame, originX: originX, originY: originY, width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame), deadline: deadline)
        }
        return (CFAbsoluteTimeGetCurrent() - started) * 1000 / Double(samples)
    }

    private func predictSourceTile(frame: CVPixelBuffer, originX: Int, originY: Int, width: Int, height: Int, deadline: CFAbsoluteTime) throws -> [UInt8] {
        if CFAbsoluteTimeGetCurrent() >= deadline {
            throw AnimeRestoreStall(variant: variant.title, tilesFinished: tilesFinished, tileCount: tileCount, lastStep: lastStep)
        }
        let tile = Self.tile
        let pointer = inputArray.dataPointer.bindMemory(to: Float.self, capacity: 3 * tile * tile)
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        guard let base = CVPixelBufferGetBaseAddress(frame) else {
            CVPixelBufferUnlockBaseAddress(frame, .readOnly)
            throw ProcessorError.conversionFailed("Anime Restore source frame is unreadable")
        }
        let stride = CVPixelBufferGetBytesPerRow(frame)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<tile {
            let sy = Self.reflect(originY + y, height)
            let row = pixels.advanced(by: sy * stride)
            for x in 0..<tile {
                let sx = Self.reflect(originX + x, width)
                let p = row.advanced(by: sx * 4)
                let i = y * tile + x
                pointer[i] = Float(p[2]) / 255
                pointer[tile * tile + i] = Float(p[1]) / 255
                pointer[2 * tile * tile + i] = Float(p[0]) / 255
            }
        }
        CVPixelBufferUnlockBaseAddress(frame, .readOnly)
        lastStep = "predict"
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
        let result = try model.prediction(from: provider)
        guard let output = result.featureValue(for: "output")?.multiArrayValue else {
            throw ProcessorError.conversionFailed("Anime Restore model returned no output")
        }
        lastStep = "downscale-tile"
        return try Self.downscaleTile(output)
    }

    private static func sample(_ raw: UnsafeMutableRawPointer, floats: Bool, strides: [Int], channel: Int, y: Int, x: Int) -> Float {
        let offset = channel * strides[1] + y * strides[2] + x * strides[3]
        if floats {
            return raw.assumingMemoryBound(to: Float.self)[offset]
        }
        return Float(raw.assumingMemoryBound(to: Float16.self)[offset])
    }

    static func origins(_ length: Int) -> [Int] {
        if length <= tile { return [0] }
        let step = tile - overlap
        var values: [Int] = []
        var origin = 0
        while origin + tile < length {
            values.append(origin)
            origin += step
        }
        values.append(length - tile)
        return values
    }

    static func reflect(_ index: Int, _ limit: Int) -> Int {
        guard limit > 1 else { return 0 }
        var value = index
        while value < 0 || value >= limit {
            value = value < 0 ? -value : (2 * limit - value - 2)
        }
        return value
    }

    private static func ramp(_ index: Int, length: Int, overlap: Int) -> Float {
        let edge = min(index, length - 1 - index)
        if overlap <= 0 || edge >= overlap { return 1 }
        return Float(edge + 1) / Float(overlap + 1)
    }

    private static func downscaleTile(_ output: MLMultiArray) throws -> [UInt8] {
        let strides = output.strides.map(\.intValue)
        let floats = output.dataType != .float16
        let raw = output.dataPointer
        var big = [UInt8](repeating: 255, count: 512 * 512 * 4)
        for y in 0..<512 {
            for x in 0..<512 {
                let index = (y * 512 + x) * 4
                big[index] = byte(sample(raw, floats: floats, strides: strides, channel: 2, y: y, x: x))
                big[index + 1] = byte(sample(raw, floats: floats, strides: strides, channel: 1, y: y, x: x))
                big[index + 2] = byte(sample(raw, floats: floats, strides: strides, channel: 0, y: y, x: x))
            }
        }
        var small = [UInt8](repeating: 0, count: 256 * 256 * 4)
        let error: vImage_Error = big.withUnsafeMutableBytes { sourceRaw in
            small.withUnsafeMutableBytes { destinationRaw in
                var source = vImage_Buffer(data: sourceRaw.baseAddress, height: 512, width: 512, rowBytes: 512 * 4)
                var destination = vImage_Buffer(data: destinationRaw.baseAddress, height: 256, width: 256, rowBytes: 256 * 4)
                return vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        guard error == kvImageNoError else { throw ProcessorError.conversionFailed("Anime Restore tile downscale failed") }
        return small
    }

    private static func blend(tile: [UInt8], into canvas: inout [Float], weights: inout [Float], originX: Int, originY: Int, width: Int, height: Int) {
        for y in 0..<tile where originY + y < height {
            let weightY = ramp(y, length: tile, overlap: overlap)
            for x in 0..<tile where originX + x < width {
                let weight = weightY * ramp(x, length: tile, overlap: overlap)
                let index = (originY + y) * width + originX + x
                let pixel = (y * tile + x) * 4
                canvas[index * 3] += Float(tile[pixel + 2]) / 255 * weight
                canvas[index * 3 + 1] += Float(tile[pixel + 1]) / 255 * weight
                canvas[index * 3 + 2] += Float(tile[pixel]) / 255 * weight
                weights[index] += weight
            }
        }
    }

    private static func pack(canvas: [Float], weights: [Float], width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &buffer) == kCVReturnSuccess, let buffer else {
            throw ProcessorError.conversionFailed("Anime Restore output buffer failed")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let destination = CVPixelBufferGetBaseAddress(buffer) {
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            let pixels = destination.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                let row = pixels.advanced(by: y * rowBytes)
                for x in 0..<width {
                    let index = y * width + x
                    let weight = max(weights[index], 1e-4)
                    let pixel = x * 4
                    row[pixel] = byte(canvas[index * 3 + 2] / weight)
                    row[pixel + 1] = byte(canvas[index * 3 + 1] / weight)
                    row[pixel + 2] = byte(canvas[index * 3] / weight)
                    row[pixel + 3] = 255
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    private static func byte(_ value: Float) -> UInt8 {
        UInt8(min(255, max(0, value * 255 + 0.5)))
    }

}

struct AnimeRestoreStall: Error, CustomStringConvertible {
    let variant: String
    let tilesFinished: Int
    let tileCount: Int
    let lastStep: String
    var description: String { "Anime Restore stall • \(variant) • tiles \(tilesFinished)/\(tileCount) • last=\(lastStep)" }
}


enum AnimeRestoreProbe {
    static func resolve(quality: AnimeRestoreQuality, width: Int, height: Int, frame: CVPixelBuffer) throws -> AnimeRestoreVariant {
        if quality == .span { return .span }
        if quality == .apisr { return .apisr }
        if let saved = AnimeRestoreChoice.savedVariant() {
            DiagnosticsLogger.shared.log("Anime Restore auto pick reused • \(saved.title)")
            return saved
        }
        let tiles = AnimeRestoreChoice.tileCount(length: width) * AnimeRestoreChoice.tileCount(length: height)
        let model = try AnimeRestoreTiledModel(variant: .apisr, units: .cpuAndNeuralEngine)
        let perTile = try model.milliseconds(forTileCount: 4, frame: frame)
        let frameCost = perTile * Double(max(tiles, 1))
        let decision = AnimeRestoreChoice.pick(quality: .auto, apisrMillisecondsPerFrame: frameCost)
        AnimeRestoreChoice.save(variant: decision.0, apisrMillisecondsPerFrame: frameCost, effective: decision.1)
        let line = String(format: "Anime Restore auto pick • APISR %.0f ms/tile • %.0f ms/unique frame • %.0f tiles • effective %.0f ms after 2s reuse • chose %@", perTile, frameCost, Double(tiles), decision.1, decision.0.title)
        DiagnosticsLogger.shared.log(line)
        CrashCapture.step(line)
        return decision.0
    }
}
