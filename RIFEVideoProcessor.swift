import Foundation
import AVFoundation
import RifeMetal

struct ProcessorConfiguration {
    let quality: RifeQualityTier
    let ghostProtection: Bool
    let sceneCutProtection: Bool
    let ghostSensitivity: Double
    let codec: OutputCodec
    let preserveAudio: Bool
    let targetFPS: Double
}

enum ProcessorError: LocalizedError {
    case missingVideoTrack
    case reader(String)
    case writer(String)
    case conversionFailed(String)
    case noOutput

    var errorDescription: String? {
        switch self {
        case .missingVideoTrack: return "The selected file has no readable video track."
        case .reader(let s): return "Video reader failed: \(s)"
        case .writer(let s): return "Video writer failed: \(s)"
        case .conversionFailed(let s): return "Frame conversion failed: \(s)"
        case .noOutput: return "The output video could not be created."
        }
    }
}

final class RIFEVideoProcessor {
    private let config: ProcessorConfiguration

    init(configuration: ProcessorConfiguration) { self.config = configuration }

    func process(sourceURL: URL,
                 progress: @escaping @Sendable (Double, String) -> Void) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ProcessorError.missingVideoTrack }
        let duration = try await asset.load(.duration)
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())

        let silentURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rife60-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: silentURL)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot attach video output") }
        reader.add(output)

        let writer = try AVAssetWriter(outputURL: silentURL, fileType: .mov)
        let codec: AVVideoCodecType = config.codec == .hevc ? .hevc : .h264
        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: max(12_000_000, width * height * 10),
            AVVideoExpectedSourceFrameRateKey: Int(config.targetFPS),
            AVVideoMaxKeyFrameIntervalKey: Int(config.targetFPS * 2)
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: codec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression
        ])
        input.expectsMediaDataInRealTime = false
        input.transform = transform
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ])
        guard writer.canAdd(input) else { throw ProcessorError.writer("cannot attach video input") }
        writer.add(input)

        let interpolator = try RifeInterpolator(configuration: .bundled(qualityTier: config.quality))
        let guarder = GhostGuard(sensitivity: config.ghostSensitivity, enableSceneCuts: config.sceneCutProtection)

        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "unknown error") }
        guard writer.startWriting() else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "unknown error") }
        writer.startSession(atSourceTime: .zero)
        guard let pool = adaptor.pixelBufferPool else { throw ProcessorError.writer("pixel buffer pool unavailable") }

        let frameStep = CMTime(value: 1, timescale: CMTimeScale(config.targetFPS.rounded()))
        var nextOutputTime = CMTime.zero
        var previousSample: CMSampleBuffer?
        var rejected = 0
        var generated = 0

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
            let currentTime = CMSampleBufferGetPresentationTimeStamp(sample)
            if previousSample == nil {
                try await append(pb, at: .zero, input: input, adaptor: adaptor)
                nextOutputTime = frameStep
                previousSample = sample
                continue
            }

            guard let prevSample = previousSample, let prevPB = CMSampleBufferGetImageBuffer(prevSample) else { continue }
            let prevTime = CMSampleBufferGetPresentationTimeStamp(prevSample)
            let span = CMTimeSubtract(currentTime, prevTime)
            let spanSeconds = max(CMTimeGetSeconds(span), 1.0/240.0)

            while CMTimeCompare(nextOutputTime, currentTime) < 0 {
                try Task.checkCancellation()
                let rel = CMTimeGetSeconds(CMTimeSubtract(nextOutputTime, prevTime)) / spanSeconds
                if rel <= 0.001 {
                    try await append(prevPB, at: nextOutputTime, input: input, adaptor: adaptor)
                } else {
                    let t = min(max(rel, 0.001), 0.999)
                    let synth = try interpolator.interpolate(previous: prevPB, current: pb, timesteps: [Float(t)])[0]
                    generated += 1
                    var chosen: CVPixelBuffer = synth
                    if config.ghostProtection {
                        let check = guarder.inspect(previous: prevPB, generated: synth, current: pb)
                        if check.reject {
                            rejected += 1
                            chosen = t < 0.5 ? prevPB : pb
                        }
                    }
                    try await append(chosen, at: nextOutputTime, input: input, adaptor: adaptor)
                }
                nextOutputTime = CMTimeAdd(nextOutputTime, frameStep)
            }

            previousSample = sample
            let frac = min(max(CMTimeGetSeconds(currentTime) / max(CMTimeGetSeconds(duration), 0.001), 0), 1)
            progress(frac * 0.92, "Interpolating • \(generated) generated • \(rejected) rejected")
        }

        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "finish failed") }

        if config.preserveAudio {
            progress(0.95, "Restoring original audio…")
            return try await addOriginalAudio(videoURL: silentURL, sourceAsset: asset)
        }
        progress(1.0, "Finished")
        return silentURL
    }

    private func append(_ pixelBuffer: CVPixelBuffer,
                        at time: CMTime,
                        input: AVAssetWriterInput,
                        adaptor: AVAssetWriterInputPixelBufferAdaptor) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 2_000_000)
        }
        guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
            throw ProcessorError.writer("failed appending frame at \(CMTimeGetSeconds(time)) s")
        }
    }

    private func addOriginalAudio(videoURL: URL, sourceAsset: AVAsset) async throws -> URL {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: outputURL)

        let composition = AVMutableComposition()
        let processed = AVURLAsset(url: videoURL)
        guard let pv = try await processed.loadTracks(withMediaType: .video).first,
              let cv = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw ProcessorError.noOutput
        }
        let pDuration = try await processed.load(.duration)
        try cv.insertTimeRange(CMTimeRange(start: .zero, duration: pDuration), of: pv, at: .zero)
        cv.preferredTransform = try await pv.load(.preferredTransform)

        if let audio = try await sourceAsset.loadTracks(withMediaType: .audio).first,
           let ca = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
            let aDuration = try await sourceAsset.load(.duration)
            let d = CMTimeMinimum(pDuration, aDuration)
            try ca.insertTimeRange(CMTimeRange(start: .zero, duration: d), of: audio, at: .zero)
        }

        guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
            throw ProcessorError.noOutput
        }
        exporter.outputURL = outputURL
        exporter.outputFileType = .mp4
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            exporter.exportAsynchronously { continuation.resume() }
        }
        guard exporter.status == .completed else {
            throw ProcessorError.writer(exporter.error?.localizedDescription ?? "audio mux failed")
        }
        try? FileManager.default.removeItem(at: videoURL)
        return outputURL
    }
}
