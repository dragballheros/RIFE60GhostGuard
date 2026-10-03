import Foundation
import AVFoundation
import CoreVideo
import Darwin

struct GenerativeEditConfiguration: Equatable, Sendable {
    var prompt = ""
    var negativePrompt = "flicker, temporal inconsistency, warped anatomy, extra limbs, changed face, changed hair, inconsistent clothing, distorted background, blurry details"
    var strength = 0.82
    var steps = 8
    var seed: Int64 = -1
}

enum GenerativeEditError: LocalizedError {
    case modelsMissing, invalidVideo, backendUnavailable(String), memoryUnavailable
    var errorDescription: String? {
        switch self {
        case .modelsMissing: return "Install the VACE diffusion GGUF, UMT5-XXL encoder, and Wan VAE first."
        case .invalidVideo: return "The selected video could not be decoded into a temporal edit window."
        case .backendUnavailable(let message): return message
        case .memoryUnavailable: return "iOS has under 700 MB free. Close other apps and run the edit again. A 14 Pro Max can load Wan 2.1 VACE 1.3B."
        }
    }
}

final class GenerativeVideoEngine {
    private var handle: OpaquePointer?

    func load() throws {
        guard GenerativeModelStore.installed else { throw GenerativeEditError.modelsMissing }
        // 14 Pro Max is 6 GB marketing RAM but reports about 5.5 GiB. Block only when iOS has almost nothing free.
        let available = os_proc_available_memory()
        guard available >= 700 * 1024 * 1024 else { throw GenerativeEditError.memoryUnavailable }
        handle = ge_create(GenerativeModelStore.diffusion.path, GenerativeModelStore.vae.path, GenerativeModelStore.textEncoder.path)
        guard handle != nil else { throw GenerativeEditError.backendUnavailable(String(cString: ge_last_error())) }
    }

    func cancel() {
        if let handle { ge_cancel(handle) }
    }

    func unload() {
        if let handle { ge_destroy(handle) }
        handle = nil
    }

    deinit { unload() }

    func edit(sourceURL: URL, configuration: GenerativeEditConfiguration,
              progress: @escaping @Sendable (Double, String) -> Void) async throws -> URL {
        try load()
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw GenerativeEditError.invalidVideo }
        let natural = try await track.load(.naturalSize)
        let duration = try await asset.load(.duration)
        let sourceFPS = max(1, Int((try? await track.load(.nominalFrameRate)) ?? 24))
        let requested = max(5, Int((duration.seconds * Double(sourceFPS)).rounded()))
        let count = min(13, requested)
        let longSide = max(abs(natural.width), abs(natural.height))
        let width = max(256, min(480, Int(abs(natural.width) * 480 / longSide / 16) * 16))
        let height = max(256, min(320, Int(abs(natural.height) * 320 / longSide / 16) * 16))
        let frames = try decode(asset, count: count, width: width, height: height)
        var packed = Data(capacity: frames.count * width * height * 4)
        frames.forEach { packed.append($0) }
        progress(0.05, "Generative Edit • preparing \(frames.count)-frame window at \(width)x\(height)")
        let seed = configuration.seed >= 0 ? configuration.seed : Int64.random(in: 0...Int64.max)
        let result = try await generate(packed, count: frames.count, width: width, height: height, fps: sourceFPS, configuration: configuration, seed: seed, progress: progress)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("AnimeGenerativeEdit-(UUID().uuidString).mov")
        try encode(result.0, width: width, height: height, fps: result.1, to: url)
        progress(1, "Generative Edit complete")
        return url
    }

    private func generate(_ data: Data, count: Int, width: Int, height: Int, fps: Int,
                          configuration: GenerativeEditConfiguration, seed: Int64,
                          progress: @escaping @Sendable (Double, String) -> Void) async throws -> ([Data], Int) {
        try await Task.detached(priority: .userInitiated) {
            var output: UnsafeMutablePointer<UInt8>?
            var outputCount: Int32 = 0
            var outputFPS: Int32 = Int32(fps)
            let ok = data.withUnsafeBytes { raw in
                configuration.prompt.withCString { prompt in
                    configuration.negativePrompt.withCString { negative in
                        ge_generate(self.handle, raw.bindMemory(to: UInt8.self).baseAddress,
                                    Int32(count), Int32(width), Int32(height), Int32(fps),
                                    prompt, negative, Int32(configuration.steps), Float(configuration.strength),
                                    seed, &output, &outputCount, &outputFPS)
                    }
                }
            }
            guard ok != 0, let output else { throw GenerativeEditError.backendUnavailable(String(cString: ge_last_error())) }
            let bytes = width * height * 4
            let generatedCount = Int(outputCount)
            var frames = [Data]()
            frames.reserveCapacity(generatedCount)
            for i in 0..<generatedCount { frames.append(Data(bytes: output.advanced(by: i * bytes), count: bytes)) }
            ge_free_frames(output)
            progress(0.9, "Generative Edit • reconstructed (outputCount) frames")
            return (frames, Int(outputFPS))

        }.value
    }

    private func decode(_ asset: AVAsset, count: Int, width: Int, height: Int) throws -> [Data] {
        let reader = try AVAssetReader(asset: asset)
        guard let track = asset.tracks(withMediaType: .video).first else { throw GenerativeEditError.invalidVideo }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height
        ])
        reader.add(output); reader.startReading()
        var frames = [Data](); var index = 0
        let total = max(1, Int((asset.duration.seconds * Double(max(1, Int(track.nominalFrameRate)))).rounded()))
        let stride = Double(total) / Double(count)
        while let sample = output.copyNextSampleBuffer(), frames.count < count {
            if Double(index) >= Double(frames.count) * stride, let pb = CMSampleBufferGetImageBuffer(sample) {
                CVPixelBufferLockBaseAddress(pb, .readOnly)
                let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
                let row = CVPixelBufferGetBytesPerRow(pb)
                var data = Data(count: width * height * 4)
                data.withUnsafeMutableBytes { dst in
                    for y in 0..<height { memcpy(dst.baseAddress!.advanced(by: y * width * 4), base.advanced(by: y * row), width * 4) }
                }
                CVPixelBufferUnlockBaseAddress(pb, .readOnly)
                frames.append(data)
            }
            index += 1
        }
        guard frames.count >= 5 else { throw GenerativeEditError.invalidVideo }
        return frames
    }

    private func encode(_ frames: [Data], width: Int, height: Int, fps: Int, to url: URL) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: width, AVVideoHeightKey: height])
        writer.add(input); writer.startWriting(); writer.startSession(atSourceTime: .zero)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height
        ])
        for (index, data) in frames.enumerated() {
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pb)
            CVPixelBufferLockBaseAddress(pb!, [])
            let base = CVPixelBufferGetBaseAddress(pb!)!.assumingMemoryBound(to: UInt8.self)
            let row = CVPixelBufferGetBytesPerRow(pb!)
            data.withUnsafeBytes { src in
                for y in 0..<height { memcpy(base.advanced(by: y * row), src.baseAddress!.advanced(by: y * width * 4), width * 4) }
            }
            CVPixelBufferUnlockBaseAddress(pb!, [])
            adaptor.append(pb!, withPresentationTime: CMTime(value: Int64(index), timescale: Int32(max(1, fps))))
        }
        input.markAsFinished(); writer.finishWriting {}
        if writer.status != .completed { throw writer.error ?? GenerativeEditError.backendUnavailable("Generative Edit output encoding failed.") }
    }
}
