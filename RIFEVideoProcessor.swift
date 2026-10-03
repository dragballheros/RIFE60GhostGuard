import Foundation
import AVFoundation
import VideoToolbox
import RifeMetal

struct ProcessorConfiguration {
    let quality: RifeQualityTier
    let ghostProtection: Bool
    let sceneCutProtection: Bool
    let compressionProtection: Bool
    let outlineProtection: Bool
    let ghostSensitivity: Double
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
    private var transferSession: VTPixelTransferSession?

    private let colorPopGrade: ColorPopGrade?
    private let guardUpscaleFirstMemory: Bool

    init(configuration: ProcessorConfiguration, colorPopStrength: Double = 0, guardUpscaleFirstMemory: Bool = false) {
        self.guardUpscaleFirstMemory = guardUpscaleFirstMemory
        self.colorPopGrade = colorPopStrength > 0 ? ColorPopGrade(strength: colorPopStrength) : nil
        self.config = configuration
    }

    func process(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void = { _ in }
    ) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ProcessorError.missingVideoTrack
        }

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
        guard reader.canAdd(output) else {
            throw ProcessorError.reader("cannot attach video output")
        }
        reader.add(output)

        let writer = try AVAssetWriter(outputURL: silentURL, fileType: .mov)
        let calculatedBitrate = width * height * Int(config.targetFPS) / 2
        let bitrate = min(max(60_000_000, calculatedBitrate), 800_000_000)

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: bitrate,
            AVVideoQualityKey: 1.0,
            AVVideoExpectedSourceFrameRateKey: Int(config.targetFPS),
            AVVideoMaxKeyFrameIntervalKey: Int(config.targetFPS * 2),
            AVVideoAllowFrameReorderingKey: true,
            AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ])
        input.expectsMediaDataInRealTime = false
        input.transform = transform

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
        )

        guard writer.canAdd(input) else {
            throw ProcessorError.writer("cannot attach HEVC Main10 video input")
        }
        writer.add(input)

        let guarder = GhostGuard(
            sensitivity: config.ghostSensitivity,
            enableSceneCuts: config.sceneCutProtection
        )
        let fastMotionGuard = config.ghostProtection
            ? FastMotionGhostGuard(sensitivity: config.ghostSensitivity)
            : nil
        let compressionGuard = config.compressionProtection ? CompressionGuard() : nil

        let outlineEnhancer: OutlineEnhancer?
        if config.outlineProtection {
            progress(0.001, "Loading Sharpie Outline Subtle…")
            outlineEnhancer = try autoreleasepool { try OutlineEnhancer() }
        } else {
            outlineEnhancer = nil
        }

        progress(0.002, "Loading streaming tiled RIFE 4.26 HQ…")
        let interpolator = try autoreleasepool {
            try RifeInterpolator(configuration: .bundled(qualityTier: config.quality))
        }
        let tiledHQ = try TiledHQInterpolator(
            interpolator: interpolator,
            width: width,
            height: height,
            bandCount: 3,
            overlap: 64,
            memoryAdaptive: guardUpscaleFirstMemory
        )

        guard reader.startReading() else {
            throw ProcessorError.reader(reader.error?.localizedDescription ?? "unknown error")
        }
        guard writer.startWriting() else {
            throw ProcessorError.writer(writer.error?.localizedDescription ?? "unknown error")
        }
        writer.startSession(atSourceTime: .zero)

        guard let pool = adaptor.pixelBufferPool else {
            throw ProcessorError.writer("10-bit pixel buffer pool unavailable")
        }

        let frameStep = CMTime(value: 1, timescale: CMTimeScale(config.targetFPS.rounded()))
        var nextOutputTime = CMTime.zero
        var previousPB: CVPixelBuffer?
        var previousTime = CMTime.zero
        var rejected = 0
        var generated = 0
        var cleaned = 0
        var outlined = 0
        var sourceFrames = 0
        var outputFrames = 0

        var compressionSeconds = 0.0
        var outlineSeconds = 0.0
        var rifeSeconds = 0.0
        var ghostSeconds = 0.0
        var encodeSeconds = 0.0
        let benchmarkStart = CFAbsoluteTimeGetCurrent()

        func emitTelemetry() {
            let elapsed = max(CFAbsoluteTimeGetCurrent() - benchmarkStart, 0.001)
            var t = PerformanceTelemetry()
            t.sourceFrames = sourceFrames
            t.generatedFrames = generated
            t.rejectedFrames = rejected
            t.compressionMsPerFrame = cleaned > 0 ? compressionSeconds * 1000.0 / Double(cleaned) : 0
            t.outlineMsPerFrame = outlined > 0 ? outlineSeconds * 1000.0 / Double(outlined) : 0
            t.rifeMsPerGeneratedFrame = generated > 0 ? rifeSeconds * 1000.0 / Double(generated) : 0
            t.ghostMsPerGeneratedFrame = generated > 0 ? ghostSeconds * 1000.0 / Double(generated) : 0
            t.encodeMsPerOutputFrame = outputFrames > 0 ? encodeSeconds * 1000.0 / Double(outputFrames) : 0
            t.generatedFPS = Double(generated) / elapsed
            t.thermalState = currentThermalStateName()
            telemetry(t)
        }

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let decodedPB = CMSampleBufferGetImageBuffer(sample) else { continue }
            sourceFrames += 1

            let currentTime = CMSampleBufferGetPresentationTimeStamp(sample)
            var currentPB = decodedPB

            if let compressionGuard {
                let started = CFAbsoluteTimeGetCurrent()
                if let cleanedPB = try? compressionGuard.clean(currentPB) {
                    currentPB = cleanedPB
                    cleaned += 1
                    compressionSeconds += CFAbsoluteTimeGetCurrent() - started
                }
            }

            if let outlineEnhancer {
                let started = CFAbsoluteTimeGetCurrent()
                currentPB = try outlineEnhancer.enhance(currentPB)
                outlineSeconds += CFAbsoluteTimeGetCurrent() - started
                outlined += 1
            }

            if previousPB == nil {
                let started = CFAbsoluteTimeGetCurrent()
                try await append10Bit(currentPB, at: .zero, input: input, adaptor: adaptor, pool: pool)
                encodeSeconds += CFAbsoluteTimeGetCurrent() - started
                outputFrames += 1
                try autoreleasepool { try tiledHQ.seed(currentPB) }
                nextOutputTime = frameStep
                previousPB = currentPB
                previousTime = currentTime
                emitTelemetry()
                continue
            }

            guard let prevPB = previousPB else { continue }
            let span = CMTimeSubtract(currentTime, previousTime)
            let spanSeconds = max(CMTimeGetSeconds(span), 1.0 / 240.0)
            var requestedTimes: [CMTime] = []
            var requestedTimesteps: [Float] = []

            while CMTimeCompare(nextOutputTime, currentTime) < 0 {
                try Task.checkCancellation()
                let rel = CMTimeGetSeconds(CMTimeSubtract(nextOutputTime, previousTime)) / spanSeconds
                if rel <= 0.001 {
                    let started = CFAbsoluteTimeGetCurrent()
                    try await append10Bit(prevPB, at: nextOutputTime, input: input, adaptor: adaptor, pool: pool)
                    encodeSeconds += CFAbsoluteTimeGetCurrent() - started
                    outputFrames += 1
                } else {
                    requestedTimes.append(nextOutputTime)
                    requestedTimesteps.append(Float(min(max(rel, 0.001), 0.999)))
                }
                nextOutputTime = CMTimeAdd(nextOutputTime, frameStep)
            }

            let rifeStarted = CFAbsoluteTimeGetCurrent()
            let synthesized = try autoreleasepool {
                try tiledHQ.interpolate(current: currentPB, timesteps: requestedTimesteps)
            }
            rifeSeconds += CFAbsoluteTimeGetCurrent() - rifeStarted

            guard synthesized.count == requestedTimesteps.count else {
                throw ProcessorError.conversionFailed("Streaming tiled HQ RIFE returned an unexpected frame count")
            }

            // Fast-motion recovery is deliberately per-timestamp. We keep a
            // cached midpoint for this source span so multiple difficult 60 FPS
            // timestamps do not repeat the first half of the recovery work.
            var retryMidpoint: CVPixelBuffer?
            var retryMidpointAttempted = false

            func motionAwareRetry(t: Double) throws -> CVPixelBuffer? {
                guard let fastMotionGuard else { return nil }

                if !retryMidpointAttempted {
                    retryMidpointAttempted = true
                    let midpointStarted = CFAbsoluteTimeGetCurrent()
                    let midpoint = try autoreleasepool {
                        let frames = try interpolator.interpolate(
                            previous: prevPB,
                            current: currentPB,
                            timesteps: [0.5]
                        )
                        return frames.first
                    }
                    rifeSeconds += CFAbsoluteTimeGetCurrent() - midpointStarted
                    retryMidpoint = midpoint
                }

                guard let midpoint = retryMidpoint else { return nil }

                // Replace one large A→B jump with two smaller temporal
                // problems. This follows RIFE's recursive interpolation idea:
                // difficult motion is easier when the target is reconstructed
                // from a nearer temporal neighbor.
                if abs(t - 0.5) <= 0.001 {
                    return midpoint
                }

                let retryPrevious: CVPixelBuffer
                let retryCurrent: CVPixelBuffer
                let retryT: Float

                if t < 0.5 {
                    retryPrevious = prevPB
                    retryCurrent = midpoint
                    retryT = Float(min(max(t * 2.0, 0.001), 0.999))
                } else {
                    retryPrevious = midpoint
                    retryCurrent = currentPB
                    retryT = Float(min(max((t - 0.5) * 2.0, 0.001), 0.999))
                }

                let retryStarted = CFAbsoluteTimeGetCurrent()
                let result = try autoreleasepool {
                    let frames = try interpolator.interpolate(
                        previous: retryPrevious,
                        current: retryCurrent,
                        timesteps: [retryT]
                    )
                    return frames.first
                }
                rifeSeconds += CFAbsoluteTimeGetCurrent() - retryStarted
                return result
            }

            for index in synthesized.indices {
                try Task.checkCancellation()
                let synth = synthesized[index]
                let t = Double(requestedTimesteps[index])
                var chosen: CVPixelBuffer = synth
                var existingGuardRejected = false
                var fastCheck: FastMotionGhostGuard.Result?

                if config.ghostProtection {
                    let started = CFAbsoluteTimeGetCurrent()
                    let check = autoreleasepool {
                        guarder.inspect(previous: prevPB, generated: synth, current: currentPB)
                    }
                    ghostSeconds += CFAbsoluteTimeGetCurrent() - started
                    existingGuardRejected = check.reject

                    if let fastMotionGuard {
                        let fastStarted = CFAbsoluteTimeGetCurrent()
                        let result = autoreleasepool {
                            fastMotionGuard.inspect(previous: prevPB, generated: synth, current: currentPB)
                        }
                        ghostSeconds += CFAbsoluteTimeGetCurrent() - fastStarted
                        fastCheck = result
                    }

                    // A fast-motion failure gets a real interpolation retry,
                    // not an immediate duplicate/source-frame fallback.
                    if let fastCheck,
                       fastCheck.fastMotion,
                       (existingGuardRejected || fastCheck.reject),
                       let retry = try motionAwareRetry(t: t) {
                        let retryGuard = autoreleasepool {
                            guarder.inspect(previous: prevPB, generated: retry, current: currentPB)
                        }
                        let retryFastGuard = fastMotionGuard.inspect(
                            previous: prevPB,
                            generated: retry,
                            current: currentPB
                        )

                        if !retryGuard.reject && !retryFastGuard.reject {
                            chosen = retry
                            DiagnosticsLogger.shared.log(
                                "Fast-motion Ghost Guard recovered • motion=\(String(format: "%.3f", fastCheck.motionScore)) • originalArtifact=\(String(format: "%.3f", fastCheck.artifactScore)) • t=\(String(format: "%.3f", t))"
                            )
                        } else {
                            rejected += 1
                            chosen = t < 0.5 ? prevPB : currentPB
                            DiagnosticsLogger.shared.log(
                                "Fast-motion Ghost Guard final fallback • motion=\(String(format: "%.3f", fastCheck.motionScore)) • artifact=\(String(format: "%.3f", fastCheck.artifactScore)) • t=\(String(format: "%.3f", t))"
                            )
                        }
                    } else if existingGuardRejected || (fastCheck?.reject ?? false) {
                        rejected += 1
                        chosen = t < 0.5 ? prevPB : currentPB
                    }
                }

                let encodeStarted = CFAbsoluteTimeGetCurrent()
                try await append10Bit(chosen, at: requestedTimes[index], input: input, adaptor: adaptor, pool: pool)
                encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStarted
                outputFrames += 1
                generated += 1
            }

            previousPB = currentPB
            previousTime = currentTime

            if sourceFrames % 3 == 0 {
                emitTelemetry()
            }

            if sourceFrames % 6 == 0 {
                let frac = min(max(CMTimeGetSeconds(currentTime) / max(CMTimeGetSeconds(duration), 0.001), 0), 1)
                var stages: [String] = []
                if config.compressionProtection { stages.append("\(cleaned) cleaned") }
                if config.outlineProtection { stages.append("\(outlined) outlined") }
                let stageText = stages.isEmpty ? "" : " • " + stages.joined(separator: " • ")
                progress(frac * 0.92, "Streaming tiled HQ • \(generated) generated • \(rejected) rejected\(stageText)")
                await Task.yield()
            }

            if sourceFrames % 12 == 0 {
                CVPixelBufferPoolFlush(pool, .excessBuffers)
            }
        }

        emitTelemetry()

        if reader.status == .failed {
            throw ProcessorError.reader(reader.error?.localizedDescription ?? "decode failed")
        }

        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }

        guard writer.status == .completed else {
            throw ProcessorError.writer(writer.error?.localizedDescription ?? "finish failed")
        }

        if config.preserveAudio {
            progress(0.95, "Restoring original audio…")
            return try await addOriginalAudio(videoURL: silentURL, sourceAsset: asset)
        }

        progress(1.0, "Finished • Streaming Tiled HQ • HEVC Main10")
        return silentURL
    }

    private func append10Bit(_ source: CVPixelBuffer,
                             at time: CMTime,
                             input: AVAssetWriterInput,
                             adaptor: AVAssetWriterInputPixelBufferAdaptor,
                             pool: CVPixelBufferPool) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 2_000_000)
        }

        var destination: CVPixelBuffer?
        let poolStatus = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination)
        guard poolStatus == kCVReturnSuccess, let destination else {
            throw ProcessorError.conversionFailed("could not allocate 10-bit P010 output frame")
        }

        if transferSession == nil {
            var session: VTPixelTransferSession?
            let status = VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session)
            guard status == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create VideoToolbox pixel transfer session")
            }
            transferSession = session
        }

        guard let transferSession else {
            throw ProcessorError.conversionFailed("pixel transfer session unavailable")
        }

        let finalFrame: CVPixelBuffer
        if let colorPopGrade { finalFrame = try colorPopGrade.apply(source) } else { finalFrame = source }
        let transferStatus = VTPixelTransferSessionTransferImage(transferSession, from: finalFrame, to: destination)
        guard transferStatus == noErr else {
            throw ProcessorError.conversionFailed("BGRA→P010 conversion failed (\(transferStatus))")
        }

        guard adaptor.append(destination, withPresentationTime: time) else {
            throw ProcessorError.writer("failed appending 10-bit frame at \(CMTimeGetSeconds(time)) s")
        }
    }

    private func addOriginalAudio(videoURL: URL, sourceAsset: AVAsset) async throws -> URL {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-Main10-\(UUID().uuidString).mp4")
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
