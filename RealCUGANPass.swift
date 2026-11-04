import Foundation
import AVFoundation
import CoreML
import CoreImage
import VideoToolbox
import CoreVideo

private final class FusedPolishBox: @unchecked Sendable {
    var buffer: CVPixelBuffer?
    var pts: CMTime = .zero
    var error: Error?
    var outlineSeconds = 0.0
    var predictWallSeconds = 0.0
    var shadowSeconds = 0.0
    var compressionSeconds = 0.0
    var packSeconds = 0.0
    var refineSeconds = 0.0
    func note(_ split: FinalOutlinePass.PolishSplit) {
        outlineSeconds = split.outlineSeconds
        predictWallSeconds = split.predictWallSeconds
        shadowSeconds = split.shadowSeconds
        compressionSeconds = split.compressionSeconds
        packSeconds = split.packSeconds
        refineSeconds = split.refineSeconds
    }
}

/// Keeps at most two encodes in flight. `submit` returns once a slot is free,
/// while the transfer and append keep running on the encode queue.
private final class BoundedEncodeQueue: @unchecked Sendable {
    private let slots = DispatchSemaphore(value: 2)
    private let queue = DispatchQueue(label: "rife60.encode-append", qos: .userInitiated)
    private let lock = NSLock()
    private var failure: Error?

    func submit(_ body: @escaping () async throws -> Void) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                self.slots.wait()
                self.queue.async {
                    let done = DispatchSemaphore(value: 0)
                    var thrown: Error?
                    Task {
                        do { try await body() } catch { thrown = error }
                        done.signal()
                    }
                    done.wait()
                    if let thrown {
                        self.lock.lock()
                        if self.failure == nil { self.failure = thrown }
                        self.lock.unlock()
                    }
                    self.slots.signal()
                }
                continuation.resume()
            }
        }
    }

    func drain() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                self.slots.wait()
                self.slots.wait()
                self.slots.signal()
                self.slots.signal()
                self.lock.lock()
                let failure = self.failure
                self.lock.unlock()
                if let failure { continuation.resume(throwing: failure) } else { continuation.resume() }
            }
        }
    }
}

final class RealCUGANPass {
    private let intensity: Double
    private var outputTransferSession: VTPixelTransferSession?

    private struct TileProfile {
        let width: Int
        let height: Int
        let overlap: Int
        let modelName: String
        var stepX: Int { width - overlap * 2 }
        var stepY: Int { height - overlap * 2 }
    }

    // Quality is unchanged: same Real-CUGAN weights, Noise 3, intensity and 32px
    // overlap.  The 1080p profile simply uses a wider inference surface so a
    // 1920x1080 frame needs 2x2 = 4 predictions instead of 3x2 = 6.
    private let fallbackProfile = TileProfile(width: 512, height: 512, overlap: 32, modelName: "RealCUGAN2xNoise3_Tile512")
    private let fast1080Profile = TileProfile(width: 704, height: 608, overlap: 32, modelName: "RealCUGAN2xNoise3_Tile704x608")

    private var tileInputPool: CVPixelBufferPool?
    private var tileOutputPool: CVPixelBufferPool?
    private var stitchedFramePool: CVPixelBufferPool?
    private var fixedBatchModels: [String: MLModel] = [:]
    private var loggedStrategyKey = ""
    private var forceCopyForRestOfPass = false
    private var staySequentialForRestOfPass = false
    private var didBenchmarkStrategy = false
    private var fusedPolishActive = false
    private var lastFusedOutlineMs = 0.0
    // Keep the compiled 704x608 profile and reuse IOSurface-backed pixel-buffer pools.
    // Do not flush Core Image or pixel-buffer caches during normal CUGAN processing:
    // repeated cache teardown/rebuild adds latency and makes long runs climb from the
    // initial ~800ms/frame toward ~1200ms/frame as the device heats up.
    // The CVMetalTextureCache CIContextOption is unavailable in the Xcode 16.4 SDK used
    // by the unsigned build, so retain the IOSurface/Metal-compatible pixel-buffer pools
    // as the reusable pixel-buffer boundary without using a version-fragile option key.
    private lazy var ciContext: CIContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    private let colorPopGrade: ColorPopGrade?
    private var renderInSeconds = 0.0
    private var predictionSeconds = 0.0
    private var waitInSeconds = 0.0
    private var computeWaitSeconds = 0.0
    private var activeStrategyName = "sequentialBacked"
    private var activeStrategyLoaded = false
    private var activeStrategySlots = 1
    private var activeStrategyFallback = ""
    private var stitchSeconds = 0.0

    init(intensity: Double = 1.30, colorPopStrength: Double = 0, applyWarmthCorrection: Bool? = nil) {
        self.intensity = intensity
        let shouldApplyWarmth = applyWarmthCorrection ?? (colorPopStrength > 0)
        self.colorPopGrade = (colorPopStrength > 0 || shouldApplyWarmth)
            ? ColorPopGrade(strength: colorPopStrength, warmthMode: applyWarmthCorrection == false ? .off : nil)
            : nil
    }

    func run(
        sourceURL: URL,
        finalAudioBitrate: Double,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void,
        recoveryDirectory: URL? = nil,
        fusedPolish: FinalOutlinePass? = nil
    ) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ProcessorError.missingVideoTrack }
        let duration = try await asset.load(.duration)
        let seconds = max(CMTimeGetSeconds(duration), 0.001)
        let sourceFPS = Double(try await track.load(.nominalFrameRate))
        let estimatedFrameRate = sourceFPS.isFinite && sourceFPS > 0 ? sourceFPS : 60
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())
        let targetWidth = width * 2
        let targetHeight = height * 2

        let profile = (width >= 1600 || height >= 1600) ? fast1080Profile : fallbackProfile
        let tilesAcross = Int(ceil(Double(width) / Double(profile.stepX)))
        let tilesDown = Int(ceil(Double(height) / Double(profile.stepY)))
        let tilesPerFrame = tilesAcross * tilesDown

        let strategyExplanation = RealCUGANStrategyStore.lookupExplanation(modelName: profile.modelName, tileWidth: profile.width, tileHeight: profile.height)
        PipelineStageTimer.current?.recordText("cuganStrategyLookup", strategyExplanation)
        DiagnosticsLogger.shared.log("Real-CUGAN entered • native source=\(width)x\(height) • native 2x target=\(targetWidth)x\(targetHeight) • tile=\(profile.width)x\(profile.height) • core=\(profile.stepX)x\(profile.stepY) • tiles/frame=\(tilesPerFrame) • strategy=\(strategyExplanation) • thermal=\(currentThermalStateName()) • benchmark tiles are timed on a cool phone; this pass follows RIFE so per-tile time can be higher")
        progress(0.001, "Pass 3/3 • Real-CUGAN native 2× • \(width)×\(height) → \(targetWidth)×\(targetHeight)…")

        try preparePools(targetWidth: targetWidth, targetHeight: targetHeight, profile: profile)
        DiagnosticsLogger.shared.log("Real-CUGAN tile pools ready • Metal compatible • tile=\(profile.width)x\(profile.height) • overlap=\(profile.overlap) • core=\(profile.stepX)x\(profile.stepY)")

        progress(0.002, "Pass 3/3 • Loading Real-CUGAN Anime 2x Noise 3…")
        DiagnosticsLogger.shared.log("Real-CUGAN tiled model load begin • \(profile.modelName)")
        let preferNeuralEngine = profile.modelName == fast1080Profile.modelName && ModelComputePreference.current == .auto
        let model = try loadModel(named: profile.modelName, preferNeuralEngine: preferNeuralEngine)
        DiagnosticsLogger.shared.log("Real-CUGAN tiled model load complete")

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot attach Real-CUGAN reader") }
        reader.add(output)

        let checkpointRoot = recoveryDirectory ?? FileManager.default.temporaryDirectory
        let freeAtStart = availableDiskSpaceBytes(at: checkpointRoot)
        if let freeAtStart, freeAtStart < 1_500_000_000 {
            throw ProcessorError.writer("Not enough free storage for the 4K Real-CUGAN pass • \(formatStorageMB(freeAtStart)) MB available • at least 1500 MB required")
        }
        DiagnosticsLogger.shared.log("Real-CUGAN storage preflight • free=\(freeAtStart.map(formatStorageMB) ?? "unknown") MB")
        let targetTotalBytes = 950_000_000.0
        let containerReserveBytes = 16_000_000.0
        let usableBits = max((targetTotalBytes - containerReserveBytes) * 8.0, 8_000_000.0)
        let audioBits = max(finalAudioBitrate, 0) * seconds
        let sizeBudgetBitrate = max((usableBits - audioBits) / seconds * 0.96, 500_000.0)
        let qualityBitrate = fusedPolish == nil
            ? Double(targetWidth * targetHeight) * 60.0 * 0.30
            : Double(targetWidth * targetHeight) * 60.0 * 1.80
        let codecSafetyCeiling = fusedPolish == nil ? 160_000_000.0 : Double.greatestFiniteMagnitude
        let fusedSizeBudget = fusedPolish == nil
            ? sizeBudgetBitrate
            : max(((1_080_000_000.0 - 16_000_000.0) * 8.0 - audioBits) / seconds * 0.96, 500_000.0)
        let videoBitrate = Int(max(500_000.0, min(qualityBitrate, fusedSizeBudget, codecSafetyCeiling)))
        let estimatedTotalMB = ((Double(videoBitrate) + max(finalAudioBitrate, 0)) * seconds / 8.0) / 1_000_000.0
        DiagnosticsLogger.shared.log("Real-CUGAN bitrate policy • qualityTarget=\(Int(qualityBitrate)) • sizeCeiling=\(Int(sizeBudgetBitrate)) • selected=\(videoBitrate) • estimated=\(String(format: "%.1f", estimatedTotalMB)) MB")

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: videoBitrate,
            AVVideoExpectedSourceFrameRateKey: 60,
            AVVideoMaxKeyFrameIntervalKey: 120,
            AVVideoAllowFrameReorderingKey: true,
            AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String
        ]
        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: targetWidth,
            AVVideoHeightKey: targetHeight,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ]
        fusedPolishActive = fusedPolish != nil
        if fusedPolish != nil {
            try fusedPolish?.prepareStream(width: targetWidth, height: targetHeight)
            DiagnosticsLogger.shared.log("Fused schedule • CUGAN concurrent slots capped at 2 while Sharpie runs • compute units unchanged so the pixels stay on the saved path")
            DiagnosticsLogger.shared.log("Fused Real-CUGAN + final polish • Sharpie runs on the upscaled frame while the next frame is upscaled • no intermediate 4K file")
        }
        let checkpoint = try IncrementalVideoCheckpointWriter(
            recoveryDirectory: recoveryDirectory,
            stageID: fusedPolish == nil ? "cugan" : "final-outline",
            outputSettings: outputSettings,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                kCVPixelBufferWidthKey as String: targetWidth,
                kCVPixelBufferHeightKey as String: targetHeight,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ],
            transform: transform,
            segmentDuration: 5.0
        )
        if checkpoint.resumeTime > .zero {
            let start = checkpoint.resumeTime
            reader.timeRange = CMTimeRange(start: start, duration: CMTimeMaximum(.zero, CMTimeSubtract(duration, start)))
            DiagnosticsLogger.shared.log("Real-CUGAN incremental resume • from \(String(format: "%.3f", CMTimeGetSeconds(start)))s")
        }
        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN reader failed") }
        DiagnosticsLogger.shared.log("Real-CUGAN reader/writer started • target=\(targetWidth)x\(targetHeight) HEVC Main10 • bitrate=\(videoBitrate)")

        let alphaValue = Float(1.0 / max(intensity, 0.01))
        let alpha = try MLMultiArray(shape: [1], dataType: .float16)
        alpha[0] = NSNumber(value: alphaValue)
        var frames = 0
        var inferenceSeconds = 0.0
        var handoffSeconds = 0.0
        var polishWaitSeconds = 0.0
        let passStart = CFAbsoluteTimeGetCurrent()
        var encodeTask: Task<Void, Error>?
        let ahead = FrameAheadReader(output: output, depth: 2)
        var iterationStart = CFAbsoluteTimeGetCurrent()
        var notedFrames = 0
        let polishQueue = DispatchQueue(label: "rife60.fused-polish", qos: .userInitiated)
        let polishBox = FusedPolishBox()
        var polishGroup: DispatchGroup?
        let encodePipeline = BoundedEncodeQueue()
        var polishOutlineTotal = 0.0
        var polishPredictTotal = 0.0
        var polishShadowTotal = 0.0
        var polishCompressionTotal = 0.0
        var polishPackTotal = 0.0
        var polishRefineTotal = 0.0

        while let item = ahead.next() {
            let waitIn = ahead.lastWaitSeconds
            try await checkpoint.checkCancellation()
            let decodedFrame = item.buffer
            let pts = item.time
            if CMTimeCompare(pts, checkpoint.resumeTime) <= 0 {
                iterationStart = CFAbsoluteTimeGetCurrent()
                continue
            }
            waitInSeconds += waitIn
            let frameNumber = frames + 1
            let predictBefore = predictionSeconds
            let copyBefore = renderInSeconds
            let stitchBefore = stitchSeconds

            if frameNumber == 1 {
                DiagnosticsLogger.shared.log("Real-CUGAN first frame decoded • pixel=\(CVPixelBufferGetWidth(decodedFrame))x\(CVPixelBufferGetHeight(decodedFrame)) • tiled prediction begin")
            }

            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let upscaled = try await Self.offTask {
                try self.upscaleNative2x(decodedFrame, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile)
            }
            inferenceSeconds += CFAbsoluteTimeGetCurrent() - inferenceStart

            let handoffStart = CFAbsoluteTimeGetCurrent()
            var computeWait = 0.0
            var sharpieWait = 0.0
            var encodeSlotWait = 0.0
            if let fusedPolish {
                if let polishGroup {
                    let sharpieStart = CFAbsoluteTimeGetCurrent()
                    await Self.wait(polishGroup)
                    sharpieWait = CFAbsoluteTimeGetCurrent() - sharpieStart
                    computeWait += sharpieWait
                    polishWaitSeconds += sharpieWait
                    if let error = polishBox.error { throw error }
                    guard let ready = polishBox.buffer else {
                        throw ProcessorError.conversionFailed("Fused polish produced no frame")
                    }
                    let readyPTS = polishBox.pts
                    let outline = polishBox.outlineSeconds
                    let predict = polishBox.predictWallSeconds
                    polishOutlineTotal += outline
                    polishPredictTotal += predict
                    polishShadowTotal += polishBox.shadowSeconds
                    polishCompressionTotal += polishBox.compressionSeconds
                    polishPackTotal += polishBox.packSeconds
                    polishRefineTotal += polishBox.refineSeconds
                    let submitStart = CFAbsoluteTimeGetCurrent()
                    await encodePipeline.submit {
                        try await checkpoint.append10Bit(ready, at: readyPTS)
                    }
                    encodeSlotWait = CFAbsoluteTimeGetCurrent() - submitStart
                    computeWait += encodeSlotWait
                    frames += 1
                    if outline > 0 {
                        lastFusedOutlineMs = outline * 1000
                        PipelineStageTimer.current?.record("finalOutlineMs", milliseconds: lastFusedOutlineMs)
                        PipelineStageTimer.current?.record("sharpiePredictMs", milliseconds: predict * 1000)
                    }
                }
                let group = DispatchGroup()
                group.enter()
                polishBox.error = nil
                polishBox.buffer = nil
                polishBox.pts = pts
                let frameForPolish = upscaled
                polishQueue.async {
                    do {
                        let split = try fusedPolish.polishPrepared(frameForPolish)
                        polishBox.buffer = split.buffer
                        polishBox.note(split)
                    } catch {
                        polishBox.error = error
                    }
                    group.leave()
                }
                polishGroup = group
            } else {
            let previousEncode = encodeTask
            let encodeWaitStart = CFAbsoluteTimeGetCurrent()
            encodeTask = Task {
                try await checkpoint.append10Bit(upscaled, at: pts)
            }
            if let previousEncode { try await previousEncode.value }
            encodeSlotWait = CFAbsoluteTimeGetCurrent() - encodeWaitStart
            computeWait += encodeSlotWait
            frames += 1
            }
            computeWaitSeconds += computeWait
            handoffSeconds += CFAbsoluteTimeGetCurrent() - handoffStart
            let predictMs = (predictionSeconds - predictBefore) * 1000
            let stitchMs = (stitchSeconds - stitchBefore + renderInSeconds - copyBefore) * 1000
            let wallMs = (CFAbsoluteTimeGetCurrent() - iterationStart) * 1000
            let frameLine = String(
                format: "CUGAN breakdown: wait_in %.1f ms, wait_compute %.1f ms, predict %.1f ms (%d tiles, %d concurrent), stitch %.1f ms, wall %.1f ms",
                waitIn * 1000,
                computeWait * 1000,
                predictMs,
                tilesPerFrame,
                activeStrategySlots,
                stitchMs,
                wallMs
            )
            DiagnosticsLogger.shared.log(frameLine)
            DiagnosticsLogger.shared.log(String(format: "CUGAN wait_compute split: sharpie %.1f ms, encode %.1f ms", sharpieWait * 1000, encodeSlotWait * 1000))
            notedFrames += 1
            let done = max(frames, 1)
            let rolling = String(
                format: "CUGAN breakdown: wait_in %.1f ms, wait_compute %.1f ms, predict %.1f ms (%d tiles, %d concurrent), stitch %.1f ms, wall %.1f ms",
                waitInSeconds * 1000 / Double(done),
                computeWaitSeconds * 1000 / Double(done),
                predictionSeconds * 1000 / Double(done),
                tilesPerFrame,
                activeStrategySlots,
                (stitchSeconds + renderInSeconds) * 1000 / Double(done),
                (CFAbsoluteTimeGetCurrent() - passStart) * 1000 / Double(done)
            )
            if notedFrames == 1 || notedFrames % 12 == 0 {
                DiagnosticsLogger.shared.log("CUGAN breakdown rolling • \(rolling)")
                CrashCapture.step(rolling)
            }
            iterationStart = CFAbsoluteTimeGetCurrent()

            if frames > 0 && (frames == 1 || frames % 20 == 0) {
                let perTile = predictionSeconds * 1000 / Double(frames * max(tilesPerFrame, 1))
                DiagnosticsLogger.shared.log(String(format: "Real-CUGAN breakdown (running averages) • frame=%d • tiles/frame=%d • prediction=%.1fms/frame • %.1fms/tile • render-in=%.1fms • stitch=%.1fms • sharpieWait+submit=%.1fms • thermal=%@", frames, tilesPerFrame, predictionSeconds * 1000 / Double(frames), perTile, renderInSeconds * 1000 / Double(frames), stitchSeconds * 1000 / Double(frames), handoffSeconds * 1000 / Double(frames), currentThermalStateName()))
                let memory = currentRenderPerformanceSnapshot()
                DiagnosticsLogger.shared.log("Real-CUGAN frame \(frames) complete • \(tilesPerFrame) tiles/frame • inference=\(String(format: "%.1f", inferenceSeconds * 1000.0 / Double(frames)))ms/frame • headroom=\(Int(memory.availableMemoryMB.rounded())) MB • thermal=\(memory.thermalAndMode)")
            }

            // UI/diagnostic publishing is deliberately less frequent during CUGAN.
            // It does not change inference output and avoids waking the main thread every 3 frames.
            if notedFrames > 0 {
                let elapsed = max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)
                var t = PerformanceTelemetry()
                t.upscaledFrames = frames
                t.cuganMsPerFrame = inferenceSeconds * 1000.0 / Double(max(frames, 1))
                t.outlineMsPerFrame = lastFusedOutlineMs
                t.encodeMsPerOutputFrame = checkpoint.appendCallSeconds * 1000.0 / Double(max(frames, 1))
                t.upscaleFPS = frames > 0 ? Double(frames) / elapsed : 0
                t.cuganWaitInMs = waitIn * 1000
                t.cuganWaitComputeMs = computeWait * 1000
                t.cuganPredictMs = predictMs
                t.cuganTiles = tilesPerFrame
                t.cuganConcurrency = activeStrategySlots
                t.cuganStitchMs = stitchMs
                t.cuganWallMs = wallMs
                t.thermalState = currentThermalStateName()
                telemetry(t)
            }
            if frames > 0 && frames % 12 == 0 {
                let elapsed = max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)

                let frac = min(max(CMTimeGetSeconds(pts) / seconds, 0), 1)
                let fps = Double(frames) / elapsed
                let remaining = fps > 0 ? (seconds * estimatedFrameRate - Double(frames)) / fps : 0
                progress(frac, "Pass 3/3 • Real-CUGAN native 2× • \(frames) frames • ETA \(formatDuration(remaining))")
                await Task.yield()
            }
        }

        if let polishGroup {
            polishGroup.wait()
            if let error = polishBox.error { throw error }
            guard let ready = polishBox.buffer else {
                throw ProcessorError.conversionFailed("Fused polish produced no frame")
            }
            let readyPTS = polishBox.pts
            polishOutlineTotal += polishBox.outlineSeconds
            polishPredictTotal += polishBox.predictWallSeconds
            await encodePipeline.submit {
                try await checkpoint.append10Bit(ready, at: readyPTS)
            }
            frames += 1
            if polishBox.outlineSeconds > 0 {
                lastFusedOutlineMs = polishBox.outlineSeconds * 1000
                PipelineStageTimer.current?.record("finalOutlineMs", milliseconds: lastFusedOutlineMs)
                PipelineStageTimer.current?.record("sharpiePredictMs", milliseconds: polishBox.predictWallSeconds * 1000)
            }
        }
        if fusedPolish != nil {
            try await encodePipeline.drain()
        }
        if let encodeTask { try await encodeTask.value }
        if reader.status == .failed { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN decode failed") }
        guard frames > 0 else { throw ProcessorError.conversionFailed("Real-CUGAN received zero decoded frames") }
        let completedURL = try await checkpoint.finish()
        let completedBytes = (try? FileManager.default.attributesOfItem(atPath: completedURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        let count = Double(frames)
        PipelineStageTimer.current?.record("cuganDecodeMs", milliseconds: ahead.averageDecodeMilliseconds)
        PipelineStageTimer.current?.record("cuganPackMs", milliseconds: renderInSeconds * 1000 / count)
        PipelineStageTimer.current?.record("cuganPredictMs", milliseconds: predictionSeconds * 1000 / count)
        PipelineStageTimer.current?.record("cuganStitchMs", milliseconds: stitchSeconds * 1000 / count)
        PipelineStageTimer.current?.record("cuganSlots", milliseconds: Double(activeStrategySlots))
        PipelineStageTimer.current?.record("cuganStrategyLoaded", milliseconds: activeStrategyLoaded ? 1 : 0)
        PipelineStageTimer.current?.recordText("cuganStrategy", activeStrategyName)
        PipelineStageTimer.current?.recordText("cuganFallback", activeStrategyFallback.isEmpty ? "none" : activeStrategyFallback)
        let passWall = max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)
        let predictMs = predictionSeconds * 1000 / count
        let sharpieMs = polishOutlineTotal * 1000 / count
        let predictWallMs = polishPredictTotal * 1000 / count
        let convertMs = checkpoint.transferSeconds * 1000 / count
        let readyMs = checkpoint.readyWaitSeconds * 1000 / count
        let appendMs = checkpoint.appendCallSeconds * 1000 / count
        let passMs = passWall * 1000 / count
        let waitMs = polishWaitSeconds * 1000 / count
        let busy = max(predictMs, sharpieMs)
        let idleMs = max(0, passMs - busy - waitMs - convertMs - readyMs)
        let overlapMs = max(0, predictMs + sharpieMs - busy)
        let stageSum = predictMs + sharpieMs + convertMs + readyMs
        let summary = String(
            format: "Stage wall • frames=%.0f • pass=%.1fs • %.0fms/frame • animeRestore=%.0fms • cuganPredict=%.0fms • sharpie=%.0fms (predict %.0f • pack %.0f • refine %.0f • shadow %.0f • compression %.0f) • convert=%.0fms • encoderReady=%.0fms • append=%.0fms • sharpieWait=%.0fms • idle=%.0fms • stageSum=%.0fms • overlap=%.0fms",
            count,
            passWall,
            passMs,
            PipelineStageTimer.current?.metric("animeRestoreMs") ?? 0,
            predictMs,
            sharpieMs,
            predictWallMs,
            polishPackTotal * 1000 / count,
            polishRefineTotal * 1000 / count,
            polishShadowTotal * 1000 / count,
            polishCompressionTotal * 1000 / count,
            convertMs,
            readyMs,
            appendMs,
            waitMs,
            idleMs,
            stageSum,
            overlapMs
        )
        DiagnosticsLogger.shared.log(summary)
        CrashCapture.step(summary)
        PipelineStageTimer.current?.record("convertMs", milliseconds: convertMs)
        PipelineStageTimer.current?.record("encoderReadyMs", milliseconds: readyMs)
        PipelineStageTimer.current?.record("appendMs", milliseconds: appendMs)
        DiagnosticsLogger.shared.log("Real-CUGAN writer completed • native 2x=\(targetWidth)x\(targetHeight) • frames=\(frames) • strategy=\(activeStrategyName) • loaded=\(activeStrategyLoaded) • slots=\(activeStrategySlots) • file=\(formatStorageMB(completedBytes)) MB")
        return completedURL
    }

    /// Single-image native 2x upscale. Uses the exact same model, noise level, intensity,
    /// overlap and tiled stitching as the video path, so stills match video quality.
    func upscaleStill(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let stillWidth = CVPixelBufferGetWidth(source)
        let stillHeight = CVPixelBufferGetHeight(source)
        let useWideTiles = (stillWidth >= 1600 || stillHeight >= 1600) && currentRenderPerformanceSnapshot().allowsWideCUGANTiles
        let stillProfile = useWideTiles ? fast1080Profile : fallbackProfile
        try preparePools(targetWidth: stillWidth * 2, targetHeight: stillHeight * 2, profile: stillProfile)
        defer {
            tileInputPool = nil
            tileOutputPool = nil
            stitchedFramePool = nil
            ReclaimableMemoryRegistry.shared.remove(name: "cugan-tile-input")
            ReclaimableMemoryRegistry.shared.remove(name: "cugan-tile-output")
            ReclaimableMemoryRegistry.shared.remove(name: "cugan-stitched")
        }
        let stillPreferNeuralEngine = stillProfile.modelName == fast1080Profile.modelName && ModelComputePreference.current == .auto
        let stillModel = try loadModel(named: stillProfile.modelName, preferNeuralEngine: stillPreferNeuralEngine)
        let stillAlpha = try MLMultiArray(shape: [1], dataType: .float16)
        stillAlpha[0] = NSNumber(value: Float(1.0 / max(intensity, 0.01)))
        DiagnosticsLogger.shared.log("Real-CUGAN still image • \(stillWidth)x\(stillHeight) -> \(stillWidth * 2)x\(stillHeight * 2) • tile=\(stillProfile.width)x\(stillProfile.height)")
        let started = CFAbsoluteTimeGetCurrent()
        let result = try upscaleNative2x(source, model: stillModel, alpha: stillAlpha, frameNumber: 1, profile: stillProfile)
        let imageMs = (CFAbsoluteTimeGetCurrent() - started) * 1000
        PipelineStageTimer.current?.record("imageUpscaleMs", milliseconds: imageMs)
        DiagnosticsLogger.shared.log(String(format: "Frame stage imageUpscaleMs %.2f ms/frame", imageMs))
        return result
    }

    private func loadModel(named name: String, preferNeuralEngine: Bool = false) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw ProcessorError.conversionFailed("Bundled tiled Real-CUGAN model is missing: \(name)")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = preferNeuralEngine ? .cpuAndNeuralEngine : ModelComputePreference.current.units
        configuration.allowLowPrecisionAccumulationOnGPU = true
        DiagnosticsLogger.shared.log("Real-CUGAN compute policy • model=\(name) • requested=\(configuration.computeUnits.rawValue) • autoWideTileANE=\(preferNeuralEngine)")
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private func preparePools(targetWidth: Int, targetHeight: Int, profile: TileProfile) throws {
        tileInputPool = nil
        tileOutputPool = nil
        stitchedFramePool = nil

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: profile.width,
            kCVPixelBufferHeightKey as String: profile.height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferPoolMinimumBufferCountKey as String: 3
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess, let pool else {
            throw ProcessorError.conversionFailed("could not create Real-CUGAN input tile pool")
        }
        tileInputPool = pool
        ReclaimableMemoryRegistry.shared.trackPixelPool(pool, name: "cugan-tile-input", width: profile.width, height: profile.height)

        let outputAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: profile.width * 2,
            kCVPixelBufferHeightKey as String: profile.height * 2,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferPoolMinimumBufferCountKey as String: 3
        ]
        var outputPool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, outputAttrs as CFDictionary, &outputPool) == kCVReturnSuccess, let outputPool else {
            throw ProcessorError.conversionFailed("could not create Real-CUGAN output tile pool")
        }
        tileOutputPool = outputPool
        ReclaimableMemoryRegistry.shared.trackPixelPool(outputPool, name: "cugan-tile-output", width: profile.width * 2, height: profile.height * 2)

        let stitchedAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: targetWidth,
            kCVPixelBufferHeightKey as String: targetHeight,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferPoolMinimumBufferCountKey as String: 3
        ]
        var stitched: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, stitchedAttrs as CFDictionary, &stitched) == kCVReturnSuccess, let stitched else {
            throw ProcessorError.conversionFailed("could not create native 2x Real-CUGAN frame pool")
        }
        stitchedFramePool = stitched
        ReclaimableMemoryRegistry.shared.trackPixelPool(stitched, name: "cugan-stitched", width: targetWidth, height: targetHeight)
    }

    private struct TileSpec {
        let x: Int
        let y: Int
        let coreWidth: Int
        let coreHeight: Int
    }

    private func tileSpecs(sourceWidth: Int, sourceHeight: Int, profile: TileProfile) -> [TileSpec] {
        var specs: [TileSpec] = []
        var y = 0
        while y < sourceHeight {
            let coreHeight = min(profile.stepY, sourceHeight - y)
            var x = 0
            while x < sourceWidth {
                specs.append(TileSpec(x: x, y: y, coreWidth: min(profile.stepX, sourceWidth - x), coreHeight: coreHeight))
                x += profile.stepX
            }
            y += profile.stepY
        }
        return specs
    }

    /// Times the first real tiles of this frame. Those predictions are discarded.
    /// The frame is then rendered entirely with the saved winner, so the output
    /// matches that one strategy. A new iOS version or app build misses the record
    /// and this runs again. Benchmark Models is not required.
    private func benchmarkAndSave(source: CVPixelBuffer, model: MLModel, alpha: MLMultiArray, profile: TileProfile) {
        let existing = RealCUGANStrategyStore.plan(modelName: profile.modelName, tileWidth: profile.width, tileHeight: profile.height)
        if existing.medianMilliseconds != nil { return }
        ReclaimableMemoryRegistry.shared.flushAll()
        let snap = currentRenderPerformanceSnapshot()
        let thermalBlocks = snap.thermalState == .serious || snap.thermalState == .critical
        let head = RealCUGANStrategyStore.ProbeHeadroom(
            rawMB: snap.availableMemoryMB,
            reclaimableMB: snap.reclaimableMemoryMB,
            thermalSerious: thermalBlocks
        )
        if !head.canStart {
            let line = "Real-CUGAN automatic probe deferred • \(head.summary) • thermal=\(snap.thermalAndMode) • this pass stays sequential until headroom recovers"
            DiagnosticsLogger.shared.log(line)
            CrashCapture.step(line)
            return
        }
        guard let tileInputPool, let tileOutputPool else { return }
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let specs = Array(tileSpecs(sourceWidth: sourceWidth, sourceHeight: sourceHeight, profile: profile).prefix(2))
        guard !specs.isEmpty else { return }
        func croppedInput(_ spec: TileSpec) throws -> CVPixelBuffer {
            var input: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &input) == kCVReturnSuccess, let input else {
                throw ProcessorError.conversionFailed("automatic probe tile allocation failed")
            }
            CVPixelBufferLockBaseAddress(source, .readOnly)
            CVPixelBufferLockBaseAddress(input, [])
            defer {
                CVPixelBufferUnlockBaseAddress(input, [])
                CVPixelBufferUnlockBaseAddress(source, .readOnly)
            }
            guard let sourceBase = CVPixelBufferGetBaseAddress(source), let tileBase = CVPixelBufferGetBaseAddress(input) else {
                throw ProcessorError.conversionFailed("automatic probe tile address is unavailable")
            }
            RealCUGANTileCopy.crop(
                source: sourceBase.assumingMemoryBound(to: UInt8.self),
                sourceStride: CVPixelBufferGetBytesPerRow(source),
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                destination: tileBase.assumingMemoryBound(to: UInt8.self),
                destinationStride: CVPixelBufferGetBytesPerRow(input),
                ciOriginX: spec.x - profile.overlap,
                ciOriginY: spec.y - profile.overlap,
                tileWidth: profile.width,
                tileHeight: profile.height
            )
            return input
        }
        do {
            let first = try croppedInput(specs[0])
            var firstOutput: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &firstOutput) == kCVReturnSuccess, let firstOutput else { return }
            let firstProvider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: first),
                "alpha": MLFeatureValue(multiArray: alpha)
            ])
            let firstOptions = try Self.probeBackedOptions(model: model, output: firstOutput)
            _ = try model.prediction(from: firstProvider, options: firstOptions)
            let sequentialStart = CFAbsoluteTimeGetCurrent()
            _ = try model.prediction(from: firstProvider, options: firstOptions)
            var sequentialMs = (CFAbsoluteTimeGetCurrent() - sequentialStart) * 1000
            if specs.count > 1 {
                let second = try croppedInput(specs[1])
                var secondOutput: CVPixelBuffer?
                guard CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &secondOutput) == kCVReturnSuccess, let secondOutput else { return }
                let secondProvider = try MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(pixelBuffer: second),
                    "alpha": MLFeatureValue(multiArray: alpha)
                ])
                let secondOptions = try Self.probeBackedOptions(model: model, output: secondOutput)
                let extraStart = CFAbsoluteTimeGetCurrent()
                _ = try model.prediction(from: secondProvider, options: secondOptions)
                sequentialMs += (CFAbsoluteTimeGetCurrent() - extraStart) * 1000
            }
            var winner = RealCUGANTileStrategy.sequentialBacked
            var slots = 1
            var best = sequentialMs / Double(specs.count)
            var notes = ["sequential \(String(format: "%.1f", sequentialMs / Double(specs.count))) ms/tile"]
            if snap.effectiveMemoryMB >= 600 {
                let inputs = try specs.map(croppedInput)
                let providers = try inputs.map { input in
                    try MLDictionaryFeatureProvider(dictionary: [
                        "image": MLFeatureValue(pixelBuffer: input),
                        "alpha": MLFeatureValue(multiArray: alpha)
                    ]) as MLFeatureProvider
                }
                let batchStart = CFAbsoluteTimeGetCurrent()
                try Self.probeUnbackedBatch(model: model, providers: providers)
                let batchPerTile = (CFAbsoluteTimeGetCurrent() - batchStart) * 1000 / Double(specs.count)
                notes.append("batched \(String(format: "%.1f", batchPerTile)) ms/tile")
                if batchPerTile < best {
                    winner = .batchedUnbacked
                    slots = specs.count
                    best = batchPerTile
                }
            } else {
                notes.append("batched skipped • effective \(Int(snap.effectiveMemoryMB)) MB")
            }
            if let candidate = RealCUGANStrategyStore.concurrentCandidate(availableMB: snap.effectiveMemoryMB, tierSafe: thermalBlocks), specs.count > 1 {
                let inputs = try specs.map(croppedInput)
                var outputs: [CVPixelBuffer] = []
                var pairs: [(MLFeatureProvider, MLPredictionOptions)] = []
                for input in inputs {
                    var output: CVPixelBuffer?
                    guard CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &output) == kCVReturnSuccess, let output else { throw ProcessorError.conversionFailed("concurrent probe output failed") }
                    outputs.append(output)
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "image": MLFeatureValue(pixelBuffer: input),
                        "alpha": MLFeatureValue(multiArray: alpha)
                    ])
                    pairs.append((provider, try Self.probeBackedOptions(model: model, output: output)))
                }
                let group = DispatchGroup()
                let failureLock = NSLock()
                var failed = false
                let concurrentStart = CFAbsoluteTimeGetCurrent()
                for pair in pairs {
                    group.enter()
                    DispatchQueue.global(qos: .userInitiated).async {
                        defer { group.leave() }
                        do { _ = try model.prediction(from: pair.0, options: pair.1) } catch {
                            failureLock.lock(); failed = true; failureLock.unlock()
                        }
                    }
                }
                group.wait()
                let concurrentPerTile = (CFAbsoluteTimeGetCurrent() - concurrentStart) * 1000 / Double(specs.count)
                notes.append("concurrent \(String(format: "%.1f", concurrentPerTile)) ms/tile\(failed ? " failed" : "")")
                if !failed && concurrentPerTile < best {
                    winner = .concurrentBacked
                    slots = candidate.slots
                    best = concurrentPerTile
                }
            } else {
                notes.append("concurrent skipped • effective \(Int(snap.effectiveMemoryMB)) MB")
            }
            RealCUGANStrategyStore.save(RealCUGANStrategyRecord(
                strategy: winner.rawValue,
                slots: slots,
                medianMilliseconds: best,
                peakExtraMB: 0,
                modelName: profile.modelName,
                tileWidth: profile.width,
                tileHeight: profile.height,
                deviceModel: RealCUGANStrategyStore.deviceModel(),
                systemVersion: RealCUGANStrategyStore.systemVersion(),
                buildNumber: RealCUGANStrategyStore.buildNumber(),
                needsRebenchmark: false
            ))
            let line = "Real-CUGAN automatic probe • \(head.summary) • \(notes.joined(separator: " • ")) • winner=\(winner.rawValue) • slots=\(slots) • \(String(format: "%.1f", best))ms/tile • saved for this device, iOS, and app build"
            DiagnosticsLogger.shared.log(line)
            CrashCapture.step(line)
        } catch {
            let line = "Real-CUGAN automatic probe skipped • \(error.localizedDescription)"
            DiagnosticsLogger.shared.log(line)
            CrashCapture.step(line)
        }
    }

    /// Runs synchronous Core ML off the Swift task so an encoder continuation can run.
    private static func offTask<T>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do { continuation.resume(returning: try body()) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }

    private static func wait(_ group: DispatchGroup) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            group.notify(queue: .global(qos: .userInitiated)) {
                continuation.resume()
            }
        }
    }

    /// Single-tile probe only. Output backings are valid on prediction(from:options:).
    private static func probeBackedOptions(model: MLModel, output: CVPixelBuffer) throws -> MLPredictionOptions {
        let options = MLPredictionOptions()
        let backing = try RealCUGANOutputBacking.imageBacking(named: "output", model: model, reusing: output)
        options.outputBackings = backing.dictionary
        return options
    }

    /// Batch probe only. This function must not set output backings.
    private static func probeUnbackedBatch(model: MLModel, providers: [MLFeatureProvider]) throws {
        _ = try model.predictions(from: MLArrayBatchProvider(array: providers), options: MLPredictionOptions())
    }

    private func upscaleNative2x(
        _ source: CVPixelBuffer,
        model: MLModel,
        alpha: MLMultiArray,
        frameNumber: Int,
        profile: TileProfile
    ) throws -> CVPixelBuffer {
        if forceCopyForRestOfPass {
            return try renderFrame(source, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile, strategy: .sequentialBacked, slots: 1, forceCopy: true)
        }
        if !didBenchmarkStrategy {
            let snap = currentRenderPerformanceSnapshot()
            let thermalBlocks = snap.thermalState == .serious || snap.thermalState == .critical
            if !thermalBlocks {
                didBenchmarkStrategy = true
                benchmarkAndSave(source: source, model: model, alpha: alpha, profile: profile)
            }
        }
        let plan = RealCUGANStrategyStore.plan(modelName: profile.modelName, tileWidth: profile.width, tileHeight: profile.height)
        let memory = currentRenderPerformanceSnapshot()
        let allowed = RealCUGANStrategyStore.allowedConcurrentSlots(availableMB: memory.effectiveMemoryMB)
        // The saved slot count comes from a two-tile probe, so it stays at 2 even when
        // the current headroom allows 3 or 4. Use the headroom count for this frame.
        let strategy: RealCUGANTileStrategy
        let slots: Int
        if !staySequentialForRestOfPass && allowed >= 2 && plan.strategy != .fixedBatch2 && plan.strategy != .fixedBatch4 {
            strategy = .concurrentBacked
            slots = allowed
        } else {
            strategy = plan.strategy
            slots = plan.slots
        }
        activeStrategyName = strategy.rawValue
        activeStrategySlots = slots
        activeStrategyLoaded = plan.medianMilliseconds != nil
        let logKey = "\(profile.modelName)-\(strategy.rawValue)-\(slots)-\(Int(memory.effectiveMemoryMB))"
        if loggedStrategyKey != logKey {
            loggedStrategyKey = logKey
            let timing = plan.medianMilliseconds.map { String(format: "%.1fms/tile", $0) } ?? "no benchmark yet"
            let line = "Real-CUGAN strategy • \(strategy.rawValue) • slots=\(slots) • saved=\(plan.strategy.rawValue) slots=\(plan.slots) • effective=\(Int(memory.effectiveMemoryMB)) MB • raw=\(Int(memory.availableMemoryMB)) MB • \(timing) • model=\(profile.modelName) • tile=\(profile.width)x\(profile.height)"
            DiagnosticsLogger.shared.log(line)
            CrashCapture.step(line)
        }
        do {
            return try renderFrame(source, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile, strategy: strategy, slots: slots)
        } catch {
            let message = "Real-CUGAN strategy \(strategy.rawValue) failed (\(error.localizedDescription)); falling back to sequential backed"
            activeStrategyFallback = message
            activeStrategyName = "sequentialBacked"
            activeStrategySlots = 1
            activeStrategyLoaded = false
            staySequentialForRestOfPass = true
            DiagnosticsLogger.shared.log(message)
            PipelineStageTimer.note(message)
            RealCUGANStrategyStore.markNeedsRebenchmark(modelName: profile.modelName, tileWidth: profile.width, tileHeight: profile.height, failed: strategy)
            if strategy != .sequentialBacked {
                do {
                    return try renderFrame(source, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile, strategy: .sequentialBacked, slots: 1)
                } catch {
                    let copyMessage = "Real-CUGAN sequential backed failed (\(error.localizedDescription)); using per-tile prediction copy"
                    DiagnosticsLogger.shared.log(copyMessage)
                    PipelineStageTimer.note(copyMessage)
                }
            }
            forceCopyForRestOfPass = true
            return try renderFrame(source, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile, strategy: .sequentialBacked, slots: 1, forceCopy: true)
        }
    }

    private func renderFrame(
        _ source: CVPixelBuffer,
        model: MLModel,
        alpha: MLMultiArray,
        frameNumber: Int,
        profile: TileProfile,
        strategy: RealCUGANTileStrategy,
        slots: Int,
        forceCopy: Bool = false
    ) throws -> CVPixelBuffer {
        switch strategy {
        case .sequentialBacked:
            return try renderSequential(source, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile, allowBacking: !forceCopy)
        case .batchedUnbacked:
            return try renderBatched(source, model: model, alpha: alpha, profile: profile, batchSize: min(4, max(2, slots == 1 ? 2 : slots)))
        case .concurrentBacked:
            let memory = currentRenderPerformanceSnapshot()
            let allowed = RealCUGANStrategyStore.allowedConcurrentSlots(availableMB: memory.effectiveMemoryMB)
            let use = min(slots, allowed)
            guard use >= 2 else { throw ProcessorError.conversionFailed("not enough memory headroom for concurrent Real-CUGAN tiles") }
            activeStrategySlots = use
            return try renderConcurrent(source, model: model, alpha: alpha, profile: profile, slots: use)
        case .fixedBatch2, .fixedBatch4:
            return try renderFixedBatch(source, modelName: profile.modelName, alpha: alpha, profile: profile, strategy: strategy)
        }
    }

    private func renderSequential(
        _ source: CVPixelBuffer,
        model: MLModel,
        alpha: MLMultiArray,
        frameNumber: Int,
        profile: TileProfile,
        allowBacking: Bool
    ) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let targetWidth = sourceWidth * 2
        let targetHeight = sourceHeight * 2
        guard let tileInputPool, let tileOutputPool, let stitchedFramePool else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile pools are unavailable")
        }
        var stitchedBuffer: CVPixelBuffer?
        var reusableTile: CVPixelBuffer?
        var reusableOutput: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, stitchedFramePool, &stitchedBuffer) == kCVReturnSuccess, let stitched = stitchedBuffer,
              CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &reusableTile) == kCVReturnSuccess, let tileBuffer = reusableTile,
              CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &reusableOutput) == kCVReturnSuccess, let outputBuffer = reusableOutput else {
            throw ProcessorError.conversionFailed("could not allocate reusable Real-CUGAN tiles")
        }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(stitched, [])
        defer {
            CVPixelBufferUnlockBaseAddress(stitched, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let stitchedBase = CVPixelBufferGetBaseAddress(stitched) else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile addresses are unavailable")
        }
        let specs = tileSpecs(sourceWidth: sourceWidth, sourceHeight: sourceHeight, profile: profile)
        var secondTile: CVPixelBuffer?
        if CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &secondTile) != kCVReturnSuccess {
            secondTile = nil
        }
        let tileBuffers = [tileBuffer, secondTile ?? tileBuffer]
        var packedNext = false
        for (tileIndex, spec) in specs.enumerated() {
            let buffer = tileBuffers[tileIndex % tileBuffers.count]
            if !(tileIndex > 0 && packedNext && secondTile != nil) {
                let renderStart = CFAbsoluteTimeGetCurrent()
                CVPixelBufferLockBaseAddress(buffer, [])
                guard let base = CVPixelBufferGetBaseAddress(buffer) else {
                    throw ProcessorError.conversionFailed("Real-CUGAN tile addresses are unavailable")
                }
                RealCUGANTileCopy.crop(
                    source: sourceBase.assumingMemoryBound(to: UInt8.self),
                    sourceStride: CVPixelBufferGetBytesPerRow(source),
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    destination: base.assumingMemoryBound(to: UInt8.self),
                    destinationStride: CVPixelBufferGetBytesPerRow(buffer),
                    ciOriginX: spec.x - profile.overlap,
                    ciOriginY: spec.y - profile.overlap,
                    tileWidth: profile.width,
                    tileHeight: profile.height
                )
                CVPixelBufferUnlockBaseAddress(buffer, [])
                renderInSeconds += CFAbsoluteTimeGetCurrent() - renderStart
            }
            packedNext = false
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffer),
                "alpha": MLFeatureValue(multiArray: alpha)
            ])
            let predictionStart = CFAbsoluteTimeGetCurrent()
            if let next = specs.dropFirst(tileIndex + 1).first, let secondTile, buffer !== secondTile || tileBuffers[0] !== tileBuffers[1] {
                let nextBuffer = tileBuffers[(tileIndex + 1) % tileBuffers.count]
                let cropGroup = DispatchGroup()
                cropGroup.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    CVPixelBufferLockBaseAddress(nextBuffer, [])
                    if let nextBase = CVPixelBufferGetBaseAddress(nextBuffer) {
                        RealCUGANTileCopy.crop(
                            source: sourceBase.assumingMemoryBound(to: UInt8.self),
                            sourceStride: CVPixelBufferGetBytesPerRow(source),
                            sourceWidth: sourceWidth,
                            sourceHeight: sourceHeight,
                            destination: nextBase.assumingMemoryBound(to: UInt8.self),
                            destinationStride: CVPixelBufferGetBytesPerRow(nextBuffer),
                            ciOriginX: next.x - profile.overlap,
                            ciOriginY: next.y - profile.overlap,
                            tileWidth: profile.width,
                            tileHeight: profile.height
                        )
                    }
                    CVPixelBufferUnlockBaseAddress(nextBuffer, [])
                    cropGroup.leave()
                }
                try predictSingle(model: model, provider: provider, outputBuffer: outputBuffer, allowBacking: allowBacking)
                cropGroup.wait()
                packedNext = true
            } else {
                try predictSingle(model: model, provider: provider, outputBuffer: outputBuffer, allowBacking: allowBacking)
            }
            predictionSeconds += CFAbsoluteTimeGetCurrent() - predictionStart
            try stitchOutput(outputBuffer, into: stitchedBase, stitchedStride: CVPixelBufferGetBytesPerRow(stitched), targetHeight: targetHeight, spec: spec, profile: profile)
            if frameNumber == 1 && (tileIndex == 0 || tileIndex == specs.count - 1) {
                DiagnosticsLogger.shared.log("Real-CUGAN first-frame tile \(tileIndex + 1)/\(specs.count) complete • core=\(spec.coreWidth)x\(spec.coreHeight)")
            }
        }
        guard CVPixelBufferGetWidth(stitched) == targetWidth, CVPixelBufferGetHeight(stitched) == targetHeight else {
            throw ProcessorError.conversionFailed("Real-CUGAN stitched frame size mismatch")
        }
        return stitched
    }

    private func renderBatched(
        _ source: CVPixelBuffer,
        model: MLModel,
        alpha: MLMultiArray,
        profile: TileProfile,
        batchSize: Int
    ) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let targetHeight = sourceHeight * 2
        guard let tileInputPool, let tileOutputPool, let stitchedFramePool else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile pools are unavailable")
        }
        let specs = tileSpecs(sourceWidth: sourceWidth, sourceHeight: sourceHeight, profile: profile)
        var stitchedBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, stitchedFramePool, &stitchedBuffer) == kCVReturnSuccess, let stitched = stitchedBuffer else {
            throw ProcessorError.conversionFailed("could not allocate native 2x stitched frame")
        }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(stitched, [])
        defer {
            CVPixelBufferUnlockBaseAddress(stitched, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let stitchedBase = CVPixelBufferGetBaseAddress(stitched) else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile addresses are unavailable")
        }
        var cursor = 0
        while cursor < specs.count {
            let count = min(batchSize, specs.count - cursor)
            var inputs: [CVPixelBuffer] = []
            var outputs: [CVPixelBuffer] = []
            var providers: [MLFeatureProvider] = []
            for _ in 0..<count {
                var input: CVPixelBuffer?
                var output: CVPixelBuffer?
                guard CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &input) == kCVReturnSuccess, let input,
                      CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &output) == kCVReturnSuccess, let output else {
                    throw ProcessorError.conversionFailed("could not allocate a Real-CUGAN batch tile")
                }
                inputs.append(input)
                outputs.append(output)
            }
            let predictionStart = CFAbsoluteTimeGetCurrent()
            for index in 0..<count {
                let spec = specs[cursor + index]
                CVPixelBufferLockBaseAddress(inputs[index], [])
                guard let tileBase = CVPixelBufferGetBaseAddress(inputs[index]) else {
                    throw ProcessorError.conversionFailed("Real-CUGAN batch tile address is unavailable")
                }
                RealCUGANTileCopy.crop(
                    source: sourceBase.assumingMemoryBound(to: UInt8.self),
                    sourceStride: CVPixelBufferGetBytesPerRow(source),
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    destination: tileBase.assumingMemoryBound(to: UInt8.self),
                    destinationStride: CVPixelBufferGetBytesPerRow(inputs[index]),
                    ciOriginX: spec.x - profile.overlap,
                    ciOriginY: spec.y - profile.overlap,
                    tileWidth: profile.width,
                    tileHeight: profile.height
                )
                CVPixelBufferUnlockBaseAddress(inputs[index], [])
                providers.append(try MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(pixelBuffer: inputs[index]),
                    "alpha": MLFeatureValue(multiArray: alpha)
                ]))
            }
            let batch = MLArrayBatchProvider(array: providers)
            let results = try model.predictions(from: batch, options: MLPredictionOptions())
            predictionSeconds += CFAbsoluteTimeGetCurrent() - predictionStart
            for index in 0..<count {
                guard let image = results.features(at: index).featureValue(for: "output")?.imageBufferValue else {
                    throw ProcessorError.conversionFailed("Real-CUGAN batch item did not return an output image")
                }
                try copyPixelBuffer(image, into: outputs[index])
                try stitchOutput(outputs[index], into: stitchedBase, stitchedStride: CVPixelBufferGetBytesPerRow(stitched), targetHeight: targetHeight, spec: specs[cursor + index], profile: profile)
            }
            cursor += count
        }
        return stitched
    }

    private func renderConcurrent(
        _ source: CVPixelBuffer,
        model: MLModel,
        alpha: MLMultiArray,
        profile: TileProfile,
        slots: Int
    ) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let targetHeight = sourceHeight * 2
        guard let tileInputPool, let tileOutputPool, let stitchedFramePool else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile pools are unavailable")
        }
        let specs = tileSpecs(sourceWidth: sourceWidth, sourceHeight: sourceHeight, profile: profile)
        var stitchedBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, stitchedFramePool, &stitchedBuffer) == kCVReturnSuccess, let stitched = stitchedBuffer else {
            throw ProcessorError.conversionFailed("could not allocate native 2x stitched frame")
        }
        var inputs: [CVPixelBuffer] = []
        var outputs: [CVPixelBuffer] = []
        var providers: [MLFeatureProvider] = []
        var predictionOptions: [MLPredictionOptions] = []
        for _ in 0..<slots {
            var input: CVPixelBuffer?
            var output: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &input) == kCVReturnSuccess, let input,
                  CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &output) == kCVReturnSuccess, let output else {
                throw ProcessorError.conversionFailed("could not allocate concurrent Real-CUGAN tiles")
            }
            inputs.append(input)
            outputs.append(output)
            providers.append(try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: input),
                "alpha": MLFeatureValue(multiArray: alpha)
            ]))
            let options = MLPredictionOptions()
            let backing = try RealCUGANOutputBacking.imageBacking(named: "output", model: model, reusing: output)
            options.outputBackings = backing.dictionary
            predictionOptions.append(options)
        }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(stitched, [])
        defer {
            CVPixelBufferUnlockBaseAddress(stitched, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let stitchedBase = CVPixelBufferGetBaseAddress(stitched) else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile addresses are unavailable")
        }
        var cursor = 0
        while cursor < specs.count {
            let count = min(slots, specs.count - cursor)
            let cropStart = CFAbsoluteTimeGetCurrent()
            for index in 0..<count {
                let spec = specs[cursor + index]
                CVPixelBufferLockBaseAddress(inputs[index], [])
                guard let tileBase = CVPixelBufferGetBaseAddress(inputs[index]) else {
                    throw ProcessorError.conversionFailed("Real-CUGAN concurrent tile address is unavailable")
                }
                RealCUGANTileCopy.crop(
                    source: sourceBase.assumingMemoryBound(to: UInt8.self),
                    sourceStride: CVPixelBufferGetBytesPerRow(source),
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    destination: tileBase.assumingMemoryBound(to: UInt8.self),
                    destinationStride: CVPixelBufferGetBytesPerRow(inputs[index]),
                    ciOriginX: spec.x - profile.overlap,
                    ciOriginY: spec.y - profile.overlap,
                    tileWidth: profile.width,
                    tileHeight: profile.height
                )
                CVPixelBufferUnlockBaseAddress(inputs[index], [])
            }
            renderInSeconds += CFAbsoluteTimeGetCurrent() - cropStart
            let predictionStart = CFAbsoluteTimeGetCurrent()
            let group = DispatchGroup()
            let errorLock = NSLock()
            var firstError: Error?
            for index in 0..<count {
                group.enter()
                let provider = providers[index]
                let options = predictionOptions[index]
                DispatchQueue.global(qos: .userInitiated).async {
                    defer { group.leave() }
                    do {
                        _ = try model.prediction(from: provider, options: options)
                    } catch {
                        errorLock.lock()
                        if firstError == nil { firstError = error }
                        errorLock.unlock()
                    }
                }
            }
            group.wait()
            predictionSeconds += CFAbsoluteTimeGetCurrent() - predictionStart
            if let firstError { throw firstError }
            for index in 0..<count {
                try stitchOutput(outputs[index], into: stitchedBase, stitchedStride: CVPixelBufferGetBytesPerRow(stitched), targetHeight: targetHeight, spec: specs[cursor + index], profile: profile)
            }
            cursor += count
        }
        return stitched
    }

    private func renderFixedBatch(
        _ source: CVPixelBuffer,
        modelName: String,
        alpha: MLMultiArray,
        profile: TileProfile,
        strategy: RealCUGANTileStrategy
    ) throws -> CVPixelBuffer {
        guard let count = strategy.fixedBatchCount else {
            throw ProcessorError.conversionFailed("fixed-batch strategy is missing a batch count")
        }
        let resource = strategy.modelResourceName(base: modelName)
        let model: MLModel
        if let cached = fixedBatchModels[resource] {
            model = cached
        } else {
            guard let url = Bundle.main.url(forResource: resource, withExtension: "mlmodelc") else {
                throw ProcessorError.conversionFailed("fixed-batch Real-CUGAN model is not installed: \(resource)")
            }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = ModelComputePreference.current == .auto ? .cpuAndNeuralEngine : ModelComputePreference.current.units
            configuration.allowLowPrecisionAccumulationOnGPU = true
            model = try MLModel(contentsOf: url, configuration: configuration)
            fixedBatchModels[resource] = model
        }
        guard let imageInput = model.modelDescription.inputDescriptionsByName["image"]?.multiArrayConstraint,
              let outputDescription = model.modelDescription.outputDescriptionsByName["output"]?.multiArrayConstraint,
              imageInput.dataType == .float32,
              outputDescription.dataType == .float32 else {
            throw ProcessorError.conversionFailed("fixed-batch Real-CUGAN model must use float32 tensor image input and tensor output")
        }
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let targetHeight = sourceHeight * 2
        guard let tileInputPool, let tileOutputPool, let stitchedFramePool else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile pools are unavailable")
        }
        let specs = tileSpecs(sourceWidth: sourceWidth, sourceHeight: sourceHeight, profile: profile)
        var stitchedBuffer: CVPixelBuffer?
        var tileBuffer: CVPixelBuffer?
        var modelOutputBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, stitchedFramePool, &stitchedBuffer) == kCVReturnSuccess, let stitched = stitchedBuffer,
              CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &tileBuffer) == kCVReturnSuccess, let tile = tileBuffer,
              CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &modelOutputBuffer) == kCVReturnSuccess, let modelOutput = modelOutputBuffer else {
            throw ProcessorError.conversionFailed("could not allocate fixed-batch Real-CUGAN buffers")
        }
        let inputArray = try MLMultiArray(shape: imageInput.shape, dataType: imageInput.dataType)
        let outputArray = try MLMultiArray(shape: outputDescription.shape, dataType: outputDescription.dataType)
        let plane = profile.width * profile.height
        let outputPlane = profile.width * 2 * profile.height * 2
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(tile, [])
        CVPixelBufferLockBaseAddress(stitched, [])
        defer {
            CVPixelBufferUnlockBaseAddress(stitched, [])
            CVPixelBufferUnlockBaseAddress(tile, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let tileBase = CVPixelBufferGetBaseAddress(tile),
              let stitchedBase = CVPixelBufferGetBaseAddress(stitched) else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile addresses are unavailable")
        }
        var cursor = 0
        while cursor < specs.count {
            let live = min(count, specs.count - cursor)
            for index in 0..<count {
                let spec = specs[min(cursor + index, specs.count - 1)]
                RealCUGANTileCopy.crop(
                    source: sourceBase.assumingMemoryBound(to: UInt8.self),
                    sourceStride: CVPixelBufferGetBytesPerRow(source),
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    destination: tileBase.assumingMemoryBound(to: UInt8.self),
                    destinationStride: CVPixelBufferGetBytesPerRow(tile),
                    ciOriginX: spec.x - profile.overlap,
                    ciOriginY: spec.y - profile.overlap,
                    tileWidth: profile.width,
                    tileHeight: profile.height
                )
                let destination = inputArray.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: index * plane * 3)
                RealCUGANBatchTensor.pack(
                    bgra: tileBase.assumingMemoryBound(to: UInt8.self),
                    stride: CVPixelBufferGetBytesPerRow(tile),
                    width: profile.width,
                    height: profile.height,
                    destination: destination,
                    planeFloats: plane
                )
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(multiArray: inputArray),
                "alpha": MLFeatureValue(multiArray: alpha)
            ])
            let options = MLPredictionOptions()
            guard let outputName = model.modelDescription.outputDescriptionsByName.keys.first else {
                throw ProcessorError.conversionFailed("fixed-batch Real-CUGAN model has no output")
            }
            options.outputBackings = [outputName: outputArray]
            let predictionStart = CFAbsoluteTimeGetCurrent()
            _ = try model.prediction(from: provider, options: options)
            predictionSeconds += CFAbsoluteTimeGetCurrent() - predictionStart
            for index in 0..<live {
                CVPixelBufferLockBaseAddress(modelOutput, [])
                guard let modelOutputBase = CVPixelBufferGetBaseAddress(modelOutput) else {
                    CVPixelBufferUnlockBaseAddress(modelOutput, [])
                    throw ProcessorError.conversionFailed("fixed-batch output address is unavailable")
                }
                RealCUGANBatchTensor.unpack(
                    source: outputArray.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: index * outputPlane * 3),
                    planeFloats: outputPlane,
                    width: profile.width * 2,
                    height: profile.height * 2,
                    bgra: modelOutputBase.assumingMemoryBound(to: UInt8.self),
                    stride: CVPixelBufferGetBytesPerRow(modelOutput)
                )
                CVPixelBufferUnlockBaseAddress(modelOutput, [])
                try stitchOutput(modelOutput, into: stitchedBase, stitchedStride: CVPixelBufferGetBytesPerRow(stitched), targetHeight: targetHeight, spec: specs[cursor + index], profile: profile)
            }
            cursor += live
        }
        return stitched
    }

    private func stitchOutput(
        _ outputBuffer: CVPixelBuffer,
        into stitchedBase: UnsafeMutableRawPointer,
        stitchedStride: Int,
        targetHeight: Int,
        spec: TileSpec,
        profile: TileProfile,
        outputIsModelSized: Bool = true
    ) throws {
        CVPixelBufferLockBaseAddress(outputBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(outputBuffer, []) }
        guard let freshOutput = CVPixelBufferGetBaseAddress(outputBuffer) else {
            throw ProcessorError.conversionFailed("Real-CUGAN output tile address is unavailable")
        }
        let stitchStart = CFAbsoluteTimeGetCurrent()
        let outputHeight = outputIsModelSized ? profile.height * 2 : CVPixelBufferGetHeight(outputBuffer)
        RealCUGANTileCopy.stitch(
            modelOutput: freshOutput.assumingMemoryBound(to: UInt8.self),
            outputStride: CVPixelBufferGetBytesPerRow(outputBuffer),
            outputHeight: outputHeight,
            destination: stitchedBase.assumingMemoryBound(to: UInt8.self),
            destinationStride: stitchedStride,
            destinationHeight: targetHeight,
            coreX: spec.x,
            coreY: spec.y,
            coreWidth: spec.coreWidth,
            coreHeight: spec.coreHeight,
            overlap: profile.overlap
        )
        stitchSeconds += CFAbsoluteTimeGetCurrent() - stitchStart
    }

    /// Single-tile prediction. Output backings stay on this call. The unbacked
    /// batch path is a different function and never sets them.
    private func predictSingle(model: MLModel, provider: MLFeatureProvider, outputBuffer: CVPixelBuffer, allowBacking: Bool) throws {
        if allowBacking {
            let options = MLPredictionOptions()
            let backing = try RealCUGANOutputBacking.imageBacking(named: "output", model: model, reusing: outputBuffer)
            options.outputBackings = backing.dictionary
            _ = try model.prediction(from: provider, options: options)
            if let separate = backing.copyIntoPool {
                try copyPixelBuffer(separate, into: outputBuffer)
            }
            return
        }
        let result = try model.prediction(from: provider)
        guard let image = result.featureValue(for: "output")?.imageBufferValue else {
            throw ProcessorError.conversionFailed("Real-CUGAN prediction did not return an output image")
        }
        try copyPixelBuffer(image, into: outputBuffer)
    }

    private func copyPixelBuffer(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination) else {
            throw ProcessorError.conversionFailed("Real-CUGAN fallback tile addresses are unavailable")
        }
        let height = min(CVPixelBufferGetHeight(source), CVPixelBufferGetHeight(destination))
        let sourceStride = CVPixelBufferGetBytesPerRow(source)
        let destinationStride = CVPixelBufferGetBytesPerRow(destination)
        let rowBytes = min(sourceStride, destinationStride)
        let src = sourceBase.assumingMemoryBound(to: UInt8.self)
        let dst = destinationBase.assumingMemoryBound(to: UInt8.self)
        for row in 0..<height {
            dst.advanced(by: row * destinationStride).update(from: src.advanced(by: row * sourceStride), count: rowBytes)
        }
    }

    private func append10Bit(_ source: CVPixelBuffer, at time: CMTime, input: AVAssetWriterInput, adaptor: AVAssetWriterInputPixelBufferAdaptor, pool: CVPixelBufferPool, writer: AVAssetWriter) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess, let destination else {
            throw ProcessorError.conversionFailed("could not allocate native 2x P010 frame")
        }
        if outputTransferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create native 2x pixel transfer session")
            }
            outputTransferSession = session
        }
        let finalFrame: CVPixelBuffer
        if let colorPopGrade { finalFrame = try colorPopGrade.apply(source) } else { finalFrame = source }
        guard let outputTransferSession,
              VTPixelTransferSessionTransferImage(outputTransferSession, from: finalFrame, to: destination) == noErr else {
            throw ProcessorError.conversionFailed("native 2x BGRA→P010 conversion failed")
        }
        guard adaptor.append(destination, withPresentationTime: time) else {
            let detail = writer.error?.localizedDescription ?? "writer status \(writer.status.rawValue)"
            let free = availableDiskSpaceBytes(at: writer.outputURL)
            DiagnosticsLogger.shared.log("Real-CUGAN append rejected • \(detail) • free=\(free.map(formatStorageMB) ?? "unknown") MB")
            throw ProcessorError.writer("failed appending native 2x frame • \(detail) • free storage \(free.map(formatStorageMB) ?? "unknown") MB")
        }
    }
}

func formatDuration(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds > 0 else { return "--:--" }
    let total = Int(seconds.rounded())
    let h = total / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    return h > 0 ? String(format: "%d:%02d:%02d", h, m, s) : String(format: "%02d:%02d", m, s)
}

func availableDiskSpaceBytes(at url: URL) -> Int64? {
    try? url.deletingLastPathComponent()
        .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        .volumeAvailableCapacityForImportantUsage
}

func formatStorageMB(_ bytes: Int64) -> String {
    String(format: "%.0f", Double(bytes) / 1_000_000.0)
}