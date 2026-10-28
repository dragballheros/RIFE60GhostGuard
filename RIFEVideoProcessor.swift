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

    init(configuration: ProcessorConfiguration, colorPopStrength: Double = 0, guardUpscaleFirstMemory: Bool = false, applyWarmthCorrection: Bool? = nil) {
        self.guardUpscaleFirstMemory = guardUpscaleFirstMemory
        let shouldApplyWarmth = applyWarmthCorrection ?? (colorPopStrength > 0)
        self.colorPopGrade = (colorPopStrength > 0 || shouldApplyWarmth)
            ? ColorPopGrade(strength: colorPopStrength, warmthMode: applyWarmthCorrection == false ? .off : nil)
            : nil
        self.config = configuration
    }

    func process(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void = { _ in },
        recoveryDirectory: URL? = nil
    ) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ProcessorError.missingVideoTrack
        }

        // Off is a true model bypass: preserve source cadence and audio; downstream enhancement can continue.
        if config.targetFPS <= 0 {
            progress(0.01, "RIFE Off • preserving source frame cadence…")
            let output = try await exportWithoutRIFE(sourceURL: sourceURL)
            progress(1, "RIFE Off • source cadence preserved")
            telemetry(PerformanceTelemetry())
            return output
        }

        let duration = try await asset.load(.duration)
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())
        let detectedSourceFPS = Double(track.nominalFrameRate)
        let sourceFPSLabel: String = {
            guard detectedSourceFPS.isFinite, detectedSourceFPS > 0.5 else { return "source FPS unknown" }
            return String(format: "%.2f FPS", detectedSourceFPS)
        }()
        DiagnosticsLogger.shared.log(
            "RIFE cadence • source=\(sourceFPSLabel) • target=\(String(format: "%.2f", config.targetFPS)) FPS • timestamp-driven interpolation"
        )

        let sourceFrameStep = CMTime(seconds: 1.0 / max(detectedSourceFPS, 1.0), preferredTimescale: 600)
        let resumeTimeHint: CMTime = {
            if let recoveryDirectory,
               let data = try? Data(contentsOf: recoveryDirectory.appendingPathComponent("incremental-rife/manifest.json")),
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let seconds = object["lastPTSSeconds"] as? Double,
               seconds >= 0 {
                return CMTime(seconds: seconds, preferredTimescale: 600)
            }
            return .zero
        }()

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
        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ]
        let checkpoint = try IncrementalVideoCheckpointWriter(
            recoveryDirectory: recoveryDirectory,
            stageID: "rife",
            outputSettings: outputSettings,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ],
            transform: transform,
            segmentDuration: 5.0
        )
        let resumeTime = checkpoint.resumeTime > .zero ? checkpoint.resumeTime : resumeTimeHint
        if resumeTime > .zero {
            let start = CMTimeMaximum(.zero, CMTimeSubtract(resumeTime, sourceFrameStep))
            reader.timeRange = CMTimeRange(start: start, duration: CMTimeMaximum(.zero, CMTimeSubtract(duration, start)))
            DiagnosticsLogger.shared.log("RIFE incremental resume • from output \(String(format: "%.3f", CMTimeGetSeconds(resumeTime)))s • reader overlap \(String(format: "%.3f", CMTimeGetSeconds(start)))s")
        }
        let guarder = GhostGuard(
            sensitivity: config.ghostSensitivity,
            enableSceneCuts: config.sceneCutProtection
        )
        let fastMotionGuard = config.ghostProtection
            ? FastMotionGhostGuard(sensitivity: config.ghostSensitivity)
            : nil
        let compressionGuard = config.compressionProtection ? CompressionGuard() : nil
        let colorArtifactGuard = LocalizedColorArtifactGuard()

        let outlineEnhancer: OutlineEnhancer?
        if config.outlineProtection {
            progress(0.001, "Loading Sharpie Outline Subtle…")
            outlineEnhancer = try autoreleasepool { try OutlineEnhancer() }
        } else {
            outlineEnhancer = nil
        }

        progress(0.002, "Loading streaming tiled RIFE 4.26 HQ • \(sourceFPSLabel) → \(String(format: "%.0f", config.targetFPS)) FPS…")
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

        let frameStep = CMTime(value: 1, timescale: CMTimeScale(config.targetFPS.rounded()))
        var nextOutputTime = resumeTime > .zero ? CMTimeAdd(resumeTime, frameStep) : CMTime.zero
        var previousPB: CVPixelBuffer?
        var previousPreviousPB: CVPixelBuffer?
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
            try await checkpoint.checkCancellation()
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
                if resumeTime > .zero {
                    // Decode one overlap source frame so the first post-checkpoint
                    // interpolation still has a temporal predecessor. Its output
                    // was already saved in the previous segment, so it is not
                    // emitted again.
                    previousPB = currentPB
                    previousTime = currentTime
                    try autoreleasepool { try tiledHQ.seed(currentPB) }
                    continue
                }

                let started = CFAbsoluteTimeGetCurrent()
                let firstOutputFrame = try colorPopGrade?.apply(currentPB) ?? currentPB
                try await checkpoint.append10Bit(firstOutputFrame, at: .zero)
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
                try await checkpoint.checkCancellation()
                let rel = CMTimeGetSeconds(CMTimeSubtract(nextOutputTime, previousTime)) / spanSeconds
                if rel <= 0.001 {
                    let started = CFAbsoluteTimeGetCurrent()
                    let repeatedOutputFrame = try colorPopGrade?.apply(prevPB) ?? prevPB
                    try await checkpoint.append10Bit(repeatedOutputFrame, at: nextOutputTime)
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

            // Cache recursive temporal midpoints across this source span. A difficult
            // 24→60 FPS transition can require a much smaller temporal interval than
            // one midpoint split, so flagged frames may descend up to four levels.
            var recoveryMidpoints: [String: CVPixelBuffer] = [:]

            func cachedRecoveryMidpoint(previous: CVPixelBuffer,
                                        current: CVPixelBuffer,
                                        path: String) throws -> CVPixelBuffer? {
                if let cached = recoveryMidpoints[path] { return cached }
                let started = CFAbsoluteTimeGetCurrent()
                let frames = try autoreleasepool {
                    try interpolator.interpolate(previous: previous, current: current, timesteps: [0.5])
                }
                rifeSeconds += CFAbsoluteTimeGetCurrent() - started
                guard let midpoint = frames.first else { return nil }
                recoveryMidpoints[path] = midpoint
                return midpoint
            }

            func motionAwareRetry(t: Double) throws -> CVPixelBuffer? {
                guard fastMotionGuard != nil else { return nil }

                var lower = 0.0
                var upper = 1.0
                var leftFrame = prevPB
                var rightFrame = currentPB
                var path = ""

                for _ in 0..<4 {
                    let midpointTime = (lower + upper) * 0.5
                    guard let midpoint = try cachedRecoveryMidpoint(
                        previous: leftFrame,
                        current: rightFrame,
                        path: path
                    ) else { return nil }

                    if abs(t - midpointTime) <= 0.001 { return midpoint }
                    if t < midpointTime {
                        upper = midpointTime
                        rightFrame = midpoint
                        path += "L"
                    } else {
                        lower = midpointTime
                        leftFrame = midpoint
                        path += "R"
                    }

                    // After narrowing to 1/8 of the original span, a final RIFE
                    // inference is sufficiently local for the requested timestamp.
                    if upper - lower <= 0.125 { break }
                }

                let localT = (t - lower) / max(upper - lower, 0.001)
                if localT <= 0.001 { return leftFrame }
                if localT >= 0.999 { return rightFrame }

                let started = CFAbsoluteTimeGetCurrent()
                let frames = try autoreleasepool {
                    try interpolator.interpolate(
                        previous: leftFrame,
                        current: rightFrame,
                        timesteps: [Float(min(max(localT, 0.001), 0.999))]
                    )
                }
                rifeSeconds += CFAbsoluteTimeGetCurrent() - started
                return frames.first
            }

            for index in synthesized.indices {
                try await checkpoint.checkCancellation()
                let synth = synthesized[index]
                let t = Double(requestedTimesteps[index])
                var chosen: CVPixelBuffer = synth
                var existingGuardRejected = false
                var unresolvedColorAnomaly = false
                var fastCheck: FastMotionGhostGuard.Result?

                if config.ghostProtection {
                    let started = CFAbsoluteTimeGetCurrent()
                    let colorInspection = autoreleasepool {
                        colorArtifactGuard.inspect(previous: prevPB, candidate: chosen, next: currentPB, progress: t)
                    }
                    if colorInspection.isAnomaly {
                        if let repaired = colorArtifactGuard.repair(
                            previous: prevPB,
                            candidate: chosen,
                            next: currentPB,
                            progress: t,
                            inspection: colorInspection
                        ) {
                            chosen = repaired
                            let verification = colorArtifactGuard.inspect(
                                previous: prevPB, candidate: chosen, next: currentPB, progress: t
                            )
                            unresolvedColorAnomaly = verification.isAnomaly
                            DiagnosticsLogger.shared.log(
                                "Localized chroma repair • regions=\(colorInspection.regions.count) • score=\(String(format: "%.3f", colorInspection.score)) • t=\(String(format: "%.3f", t))"
                            )
                        } else {
                            unresolvedColorAnomaly = true
                        }
                    }

                    let check = autoreleasepool {
                        guarder.inspect(previous: prevPB, generated: chosen, current: currentPB, previousContext: previousPreviousPB)
                    }
                    ghostSeconds += CFAbsoluteTimeGetCurrent() - started
                    existingGuardRejected = check.reject || unresolvedColorAnomaly

                    if let fastMotionGuard {
                        let fastStarted = CFAbsoluteTimeGetCurrent()
                        let result = autoreleasepool {
                            fastMotionGuard.inspect(previous: prevPB, generated: chosen, current: currentPB)
                        }
                        ghostSeconds += CFAbsoluteTimeGetCurrent() - fastStarted
                        fastCheck = result
                    }

                    // Never recursively interpolate across a detected cut or an
                    // isolated scene-end source frame. Midpoint recovery can recreate
                    // the exact A/B warp that the scene guard rejected.
                    let sceneBoundaryRejected = check.reason == "scene cut / transition discontinuity"
                        || check.reason == "isolated scene-end frame"

                    // A fast-motion failure gets a real interpolation retry only
                    // when the pair is not a scene boundary.
                    if !sceneBoundaryRejected,
                       let fastMotionGuard,
                       let fastCheck,
                       fastCheck.fastMotion,
                       (existingGuardRejected || fastCheck.reject || fastCheck.recoveryRecommended),
                       let retry = try motionAwareRetry(t: t) {
                        var retryChosen = retry
                        let retryColorInspection = colorArtifactGuard.inspect(
                            previous: prevPB, candidate: retryChosen, next: currentPB, progress: t
                        )
                        if retryColorInspection.isAnomaly,
                           let repairedRetry = colorArtifactGuard.repair(
                                previous: prevPB,
                                candidate: retryChosen,
                                next: currentPB,
                                progress: t,
                                inspection: retryColorInspection
                           ) {
                            retryChosen = repairedRetry
                        }
                        let retryColorStillInvalid = colorArtifactGuard.inspect(
                            previous: prevPB, candidate: retryChosen, next: currentPB, progress: t
                        ).isAnomaly
                        let retryGuard = autoreleasepool {
                            guarder.inspect(previous: prevPB, generated: retryChosen, current: currentPB, previousContext: previousPreviousPB)
                        }
                        let retryFastGuard = fastMotionGuard.inspect(
                            previous: prevPB,
                            generated: retryChosen,
                            current: currentPB
                        )

                        if !retryGuard.reject && !retryFastGuard.reject && !retryColorStillInvalid {
                            chosen = retryChosen
                            DiagnosticsLogger.shared.log(
                                "Fast-motion Ghost Guard recovered • motion=\(String(format: "%.3f", fastCheck.motionScore)) • originalArtifact=\(String(format: "%.3f", fastCheck.artifactScore)) • t=\(String(format: "%.3f", t))"
                            )
                        } else {
                            rejected += 1
                            chosen = t < 0.5 ? prevPB : currentPB
                            DiagnosticsLogger.shared.log(
                                "Fast-motion Ghost Guard final fallback • motion=\(String(format: "%.3f", fastCheck.motionScore)) • artifact=\(String(format: "%.3f", fastCheck.artifactScore)) • t=\(String(format: "%.3f", t))"
                            )
                            PipelineStageTimer.recordRejection(
                                seconds: CMTimeGetSeconds(requestedTimes[index]),
                                reason: "Fast-motion Ghost Guard final fallback • motion=\(String(format: "%.3f", fastCheck.motionScore)) • artifact=\(String(format: "%.3f", fastCheck.artifactScore))",
                                disposition: "replaced"
                            )
                        }
                    } else if existingGuardRejected || (fastCheck?.reject ?? false) {
                        // Hard failures preserve the timestamp without blending.
                        // If the previous source frame is an isolated endpoint, use
                        // the current frame, which matched the prior scene context.
                        rejected += 1
                        let rejectionReason = check.reason == "ok" ? (fastCheck?.reject == true ? "fast-motion artifact" : "guard rejected") : check.reason
                        if check.reason == "isolated scene-end frame" {
                            chosen = currentPB
                            DiagnosticsLogger.shared.log("Ghost Guard fallback • isolated scene-end source frame replaced • t=\(String(format: "%.3f", t))")
                        } else {
                            chosen = t < 0.5 ? prevPB : currentPB
                            if sceneBoundaryRejected {
                                DiagnosticsLogger.shared.log("Ghost Guard fallback • scene boundary protected, recursive retry skipped • t=\(String(format: "%.3f", t))")
                            }
                        }
                        PipelineStageTimer.recordRejection(
                            seconds: CMTimeGetSeconds(requestedTimes[index]),
                            reason: rejectionReason,
                            disposition: "replaced"
                        )
                    }
                }

                let encodeStarted = CFAbsoluteTimeGetCurrent()
                let finalOutputFrame = try colorPopGrade?.apply(chosen) ?? chosen
                try await checkpoint.append10Bit(finalOutputFrame, at: requestedTimes[index])
                encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStarted
                outputFrames += 1
                generated += 1
            }

            previousPreviousPB = prevPB
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
                progress(frac * 0.92, "Streaming tiled HQ • \(sourceFPSLabel) → \(String(format: "%.0f", config.targetFPS)) FPS • \(generated) generated • \(rejected) rejected\(stageText)")
                await Task.yield()
            }

        }

        emitTelemetry()

        if reader.status == .failed {
            throw ProcessorError.reader(reader.error?.localizedDescription ?? "decode failed")
        }

        let completedURL = try await checkpoint.finish()

        if config.preserveAudio {
            progress(0.95, "Restoring original audio…")
            return try await addOriginalAudio(videoURL: completedURL, sourceAsset: asset)
        }

        progress(1.0, "Finished • Streaming Tiled HQ • HEVC Main10")
        return completedURL
    }

    // Compatibility helper retained for the existing CI performance patch. It is
    // intentionally unused by the active pause-checkpoint writer path.
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
            throw ProcessorError.conversionFailed("BGRA→P010 conversion failed (\\(transferStatus))")
        }
        guard adaptor.append(destination, withPresentationTime: time) else {
            throw ProcessorError.writer("failed appending 10-bit frame at \\(CMTimeGetSeconds(time)) s")
        }
    }

    private func exportWithoutRIFE(sourceURL: URL) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("rife-off-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: outputURL)
        guard let session = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
            throw ProcessorError.writer("Could not create a passthrough export session for RIFE Off.")
        }
        session.outputURL = outputURL
        session.outputFileType = .mov
        session.shouldOptimizeForNetworkUse = true
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            session.exportAsynchronously {
                switch session.status {
                case .completed: continuation.resume(returning: ())
                case .cancelled: continuation.resume(throwing: CancellationError())
                case .failed: continuation.resume(throwing: ProcessorError.writer(session.error?.localizedDescription ?? "RIFE-off passthrough export failed."))
                default: continuation.resume(throwing: ProcessorError.writer("RIFE-off passthrough export ended unexpectedly (\(session.status.rawValue))."))
                }
            }
        }
        return outputURL
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