import Foundation
import AVFoundation
import CoreVideo

struct AnimeGenerativeEditConfiguration: Equatable, Sendable {
    var prompt = ""
    var negativePrompt = "flicker, temporal inconsistency, warped anatomy, extra limbs, changed face, changed hair, blurry details, distorted background, low quality"
    var strength = 0.82
    var steps = 8
    var seed: Int64 = -1
}

enum AnimeGenerativeEditError: LocalizedError {
    case unsupportedDevice, modelsMissing, invalidVideo, backendUnavailable(String)
    var errorDescription: String? {
        switch self {
        case .unsupportedDevice: return "Anime Generative Edit requires an 8 GB+ device. It is blocked on lower-memory devices to prevent OOM/thermal crashes."
        case .modelsMissing: return "The VACE model bundle is not installed."
        case .invalidVideo: return "The selected video could not be decoded."
        case .backendUnavailable(let s): return s
        }
    }
}

#if canImport(VACECPP)
import VACECPP
#endif

final class AnimeVACEVideoEngine {
    private var handle: OpaquePointer?

    func load(diffusion: URL, vae: URL, t5: URL) throws {
        guard ProcessInfo.processInfo.physicalMemory >= 8 * 1024 * 1024 * 1024 else {
            throw AnimeGenerativeEditError.unsupportedDevice
        }
        #if canImport(VACECPP)
        handle = vace_create(diffusion.path, vae.path, t5.path)
        guard handle != nil else {
            throw AnimeGenerativeEditError.backendUnavailable(String(cString: vace_last_error()))
        }
        #else
        throw AnimeGenerativeEditError.backendUnavailable("Native VACE Metal backend was not linked into this build.")
        #endif
    }

    func cancel() {
        #if canImport(VACECPP)
        if let handle { vace_cancel(handle) }
        #endif
    }

    func unload() {
        #if canImport(VACECPP)
        if let handle { vace_destroy(handle) }
        #endif
        handle = nil
    }

    deinit { unload() }

    func edit(sourceURL: URL, configuration: AnimeGenerativeEditConfiguration,
              diffusion: URL, vae: URL, t5: URL,
              progress: @escaping @Sendable (Double, String) -> Void) async throws -> URL {
        try load(diffusion: diffusion, vae: vae, t5: t5)
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw AnimeGenerativeEditError.invalidVideo
        }
        let size = try await track.load(.naturalSize)
        let duration = try await asset.load(.duration)
        let fps = max(1, Int((try? await track.load(.nominalFrameRate)) ?? 24))
        let count = min(33, max(5, Int((duration.seconds * Double(fps)).rounded())))
        let longSide = max(abs(size.width), abs(size.height))
        let w = min(832, max(256, Int(abs(size.width) * 832 / longSide / 16) * 16))
        let h = min(480, max(256, Int(abs(size.height) * 480 / longSide / 16) * 16))
        let frames = try decode(asset, count: count, width: w, height: h)
        var packed = Data(capacity: frames.count * w * h * 4)
        for frame in frames { packed.append(frame) }
        progress(0.08, "VACE: rebuilding \(frames.count) source frames at \(w)x\(h)")
        let seed = configuration.seed >= 0 ? configuration.seed : Int64.random(in: 0...Int64.max)
        let generated = try await native(packed, frames.count, w, h, fps, configuration, seed, progress)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("AnimeVACE-\(UUID().uuidString).mov")
        try encode(generated.frames, width: w, height: h, fps: generated.fps, to: url)
        progress(1, "Anime Generative Edit complete")
        return url
    }

    private func native(_ data: Data, _ count: Int, _ w: Int, _ h: Int, _ fps: Int,
                        _ configuration: AnimeGenerativeEditConfiguration, _ seed: Int64,
                        _ progress: @escaping @Sendable (Double, String) -> Void) async throws -> ([Data], Int) {
        try await Task.detached(priority: .userInitiated) {
            #if canImport(VACECPP)
            var out: UnsafeMutablePointer<UInt8>?
            var n = 0
            var outFPS = fps
            let ok = data.withUnsafeBytes { raw in
                configuration.prompt.withCString { prompt in
                    configuration.negativePrompt.withCString { negative in
                        vace_generate(self.handle, raw.bindMemory(to: UInt8.self).baseAddress,
                                      Int32(count), Int32(w), Int32(h), Int32(fps),
                                      prompt, negative, Int32(configuration.steps),
                                      Float(configuration.strength), seed, &out, &n, &outFPS)
                    }
                }
            }
            guard ok != 0, let out else {
                throw AnimeGenerativeEditError.backendUnavailable(String(cString: vace_last_error()))
            }
            let bytes = w * h * 4
            var frames = [Data]()
            frames.reserveCapacity(n)
            for i in 0..<n {
                frames.append(Data(bytes: out.advanced(by: i * bytes), count: bytes))
            }
            vace_free_frames(out)
            progress(0.9, "VACE: generated \(n) consistent frames")
            return (frames, outFPS)
            #else
            throw AnimeGenerativeEditError.backendUnavailable("Native VACE Metal backend unavailable.")
            #endif
        }.value
    }

    private func decode(_ asset: AVAsset, count: Int, width: Int, height: Int) throws -> [Data] {
        let reader = try AVAssetReader(asset: asset)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw AnimeGenerativeEditError.invalidVideo
        }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height
        ])
        reader.add(output)
        reader.startReading()
        var result = [Data]()
        var index = 0
        let total = max(1, Int((asset.duration.seconds * Double(max(1, Int(track.nominalFrameRate)))).rounded()))
        let step = Double(total) / Double(count)
        while let sample = output.copyNextSampleBuffer(), result.count < count {
            if Double(index) >= Double(result.count) * step,
               let pb = CMSampleBufferGetImageBuffer(sample) {
                CVPixelBufferLockBaseAddress(pb, .readOnly)
                let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
                let row = CVPixelBufferGetBytesPerRow(pb)
                var data = Data(count: width * height * 4)
                data.withUnsafeMutableBytes { dst in
                    for y in 0..<height {
                        memcpy(dst.baseAddress!.advanced(by: y * width * 4),
                               base.advanced(by: y * row), width * 4)
                    }
                }
                CVPixelBufferUnlockBaseAddress(pb, .readOnly)
                result.append(data)
            }
            index += 1
        }
        guard result.count >= 5 else { throw AnimeGenerativeEditError.invalidVideo }
        return result
    }

    private func encode(_ frames: [Data], width: Int, height: Int, fps: Int, to url: URL) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ])
        for (index, data) in frames.enumerated() {
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pb)
            CVPixelBufferLockBaseAddress(pb!, [])
            let base = CVPixelBufferGetBaseAddress(pb!)!.assumingMemoryBound(to: UInt8.self)
            let row = CVPixelBufferGetBytesPerRow(pb!)
            data.withUnsafeBytes { src in
                for y in 0..<height {
                    memcpy(base.advanced(by: y * row), src.baseAddress!.advanced(by: y * width * 4), width * 4)
                }
            }
            CVPixelBufferUnlockBaseAddress(pb!, [])
            adaptor.append(pb!, withPresentationTime: CMTime(value: Int64(index), timescale: Int32(max(1, fps))))
        }
        input.markAsFinished()
        writer.finishWriting {}
        if writer.status != .completed {
            throw writer.error ?? AnimeGenerativeEditError.backendUnavailable("VACE output encoding failed")
        }
    }
}
