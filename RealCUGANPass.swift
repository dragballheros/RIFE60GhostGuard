import Foundation
import AVFoundation
import CoreML
import CoreImage
import VideoToolbox
import CoreVideo

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
        recoveryDirectory: URL? = nil
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
        let qualityBitrate = Double(targetWidth * targetHeight) * 60.0 * 0.30
        let codecSafetyCeiling = 160_000_000.0
        let videoBitrate = Int(max(500_000.0, min(qualityBitrate, sizeBudgetBitrate, codecSafetyCeiling)))
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
        let checkpoint = try IncrementalVideoCheckpointWriter(
            recoveryDirectory: recoveryDirectory,
            stageID: "cugan",
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
        var encodeSeconds = 0.0
        let passStart = CFAbsoluteTimeGetCurrent()
        var encodeTask: Task<Void, Error>?

        while let sample = output.copyNextSampleBuffer() {
            try await checkpoint.checkCancellation()
            guard let decodedFrame = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            if CMTimeCompare(pts, checkpoint.resumeTime) <= 0 { continue }
            let frameNumber = frames + 1

            if frameNumber == 1 {
                DiagnosticsLogger.shared.log("Real-CUGAN first frame decoded • pixel=\(CVPixelBufferGetWidth(decodedFrame))x\(CVPixelBufferGetHeight(decodedFrame)) • tiled prediction begin")
            }

            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let upscaled = try upscaleNative2x(decodedFrame, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile)
            inferenceSeconds += CFAbsoluteTimeGetCurrent() - inferenceStart

            if let encodeTask { try await encodeTask.value }
            let encodeStart = CFAbsoluteTimeGetCurrent()
            encodeTask = Task {
                try await checkpoint.append10Bit(upscaled, at: pts)
            }
            frames += 1
            encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStart

            if frames == 1 || frames % 20 == 0 {
                let perTile = predictionSeconds * 1000 / Double(frames * max(tilesPerFrame, 1))
                DiagnosticsLogger.shared.log(String(format: "Real-CUGAN breakdown (running averages) • frame=%d • tiles/frame=%d • prediction=%.1fms/frame • %.1fms/tile • render-in=%.1fms • stitch=%.1fms • convert+append/backpressure=%.1fms • thermal=%@", frames, tilesPerFrame, predictionSeconds * 1000 / Double(frames), perTile, renderInSeconds * 1000 / Double(frames), stitchSeconds * 1000 / Double(frames), encodeSeconds * 1000 / Double(frames), currentThermalStateName()))
                let memory = currentRenderPerformanceSnapshot()
                DiagnosticsLogger.shared.log("Real-CUGAN frame \(frames) complete • \(tilesPerFrame) tiles/frame • inference=\(String(format: "%.1f", inferenceSeconds * 1000.0 / Double(frames)))ms/frame • headroom=\(Int(memory.availableMemoryMB.rounded())) MB • thermal=\(memory.thermalAndMode)")
            }

            // UI/diagnostic publishing is deliberately less frequent during CUGAN.
            // It does not change inference output and avoids waking the main thread every 3 frames.
            if frames % 12 == 0 {
                let elapsed = max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)
                var t = PerformanceTelemetry()
                t.upscaledFrames = frames
                t.cuganMsPerFrame = inferenceSeconds * 1000.0 / Double(frames)
                t.encodeMsPerOutputFrame = encodeSeconds * 1000.0 / Double(frames)
                t.upscaleFPS = Double(frames) / elapsed
                t.thermalState = currentThermalStateName()
                telemetry(t)

                let frac = min(max(CMTimeGetSeconds(pts) / seconds, 0), 1)
                let fps = Double(frames) / elapsed
                let remaining = fps > 0 ? (seconds * estimatedFrameRate - Double(frames)) / fps : 0
                progress(frac, "Pass 3/3 • Real-CUGAN native 2× • \(frames) frames • ETA \(formatDuration(remaining))")
                await Task.yield()
            }
        }

        if let encodeTask { try await encodeTask.value }
        if reader.status == .failed { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN decode failed") }
        guard frames > 0 else { throw ProcessorError.conversionFailed("Real-CUGAN received zero decoded frames") }
        let completedURL = try await checkpoint.finish()
        let completedBytes = (try? FileManager.default.attributesOfItem(atPath: completedURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        DiagnosticsLogger.shared.log("Real-CUGAN writer completed • native 2x=\(targetWidth)x\(targetHeight) • frames=\(frames) • file=\(formatStorageMB(completedBytes)) MB")
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
        }
        let stillPreferNeuralEngine = stillProfile.modelName == fast1080Profile.modelName && ModelComputePreference.current == .auto
        let stillModel = try loadModel(named: stillProfile.modelName, preferNeuralEngine: stillPreferNeuralEngine)
        let stillAlpha = try MLMultiArray(shape: [1], dataType: .float16)
        stillAlpha[0] = NSNumber(value: Float(1.0 / max(intensity, 0.01)))
        DiagnosticsLogger.shared.log("Real-CUGAN still image • \(stillWidth)x\(stillHeight) -> \(stillWidth * 2)x\(stillHeight * 2) • tile=\(stillProfile.width)x\(stillProfile.height)")
        return try upscaleNative2x(source, model: stillModel, alpha: stillAlpha, frameNumber: 1, profile: stillProfile)
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
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess, let pool else {
            throw ProcessorError.conversionFailed("could not create Real-CUGAN input tile pool")
        }
        tileInputPool = pool

        let outputAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: profile.width * 2,
            kCVPixelBufferHeightKey as String: profile.height * 2,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var outputPool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, outputAttrs as CFDictionary, &outputPool) == kCVReturnSuccess, let outputPool else {
            throw ProcessorError.conversionFailed("could not create Real-CUGAN output tile pool")
        }
        tileOutputPool = outputPool

        let stitchedAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: targetWidth,
            kCVPixelBufferHeightKey as String: targetHeight,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var stitched: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, stitchedAttrs as CFDictionary, &stitched) == kCVReturnSuccess, let stitched else {
            throw ProcessorError.conversionFailed("could not create native 2x Real-CUGAN frame pool")
        }
        stitchedFramePool = stitched
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
        let plan = RealCUGANStrategyStore.plan(modelName: profile.modelName, tileWidth: profile.width, tileHeight: profile.height)
        let logKey = "\(profile.modelName)-\(plan.strategy.rawValue)-\(plan.slots)"
        if loggedStrategyKey != logKey {
            loggedStrategyKey = logKey
            let timing = plan.medianMilliseconds.map { String(format: "%.1fms/tile", $0) } ?? "no benchmark yet"
            DiagnosticsLogger.shared.log("Real-CUGAN strategy • \(plan.strategy.title) • slots=\(plan.slots) • \(timing) • model=\(profile.modelName) • tile=\(profile.width)x\(profile.height)")
        }
        do {
            return try renderFrame(source, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile, strategy: plan.strategy, slots: plan.slots)
        } catch {
            let message = "Real-CUGAN strategy \(plan.strategy.rawValue) failed (\(error.localizedDescription)); falling back to sequential backed"
            DiagnosticsLogger.shared.log(message)
            PipelineStageTimer.note(message)
            RealCUGANStrategyStore.markNeedsRebenchmark(modelName: profile.modelName, tileWidth: profile.width, tileHeight: profile.height, failed: plan.strategy)
            if plan.strategy != .sequentialBacked {
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
            let available = currentRenderPerformanceSnapshot().availableMemoryMB
            let allowed = RealCUGANStrategyStore.allowedConcurrentSlots(availableMB: available)
            guard allowed >= 2 else { throw ProcessorError.conversionFailed("not enough memory headroom for concurrent Real-CUGAN tiles") }
            return try renderConcurrent(source, model: model, alpha: alpha, profile: profile, slots: min(slots, allowed))
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
        CVPixelBufferLockBaseAddress(tileBuffer, [])
        CVPixelBufferLockBaseAddress(stitched, [])
        defer {
            CVPixelBufferUnlockBaseAddress(stitched, [])
            CVPixelBufferUnlockBaseAddress(tileBuffer, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let tileBase = CVPixelBufferGetBaseAddress(tileBuffer),
              let stitchedBase = CVPixelBufferGetBaseAddress(stitched) else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile addresses are unavailable")
        }
        let specs = tileSpecs(sourceWidth: sourceWidth, sourceHeight: sourceHeight, profile: profile)
        for (tileIndex, spec) in specs.enumerated() {
            let renderStart = CFAbsoluteTimeGetCurrent()
            RealCUGANTileCopy.crop(
                source: sourceBase.assumingMemoryBound(to: UInt8.self),
                sourceStride: CVPixelBufferGetBytesPerRow(source),
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                destination: tileBase.assumingMemoryBound(to: UInt8.self),
                destinationStride: CVPixelBufferGetBytesPerRow(tileBuffer),
                ciOriginX: spec.x - profile.overlap,
                ciOriginY: spec.y - profile.overlap,
                tileWidth: profile.width,
                tileHeight: profile.height
            )
            renderInSeconds += CFAbsoluteTimeGetCurrent() - renderStart
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: tileBuffer),
                "alpha": MLFeatureValue(multiArray: alpha)
            ])
            let predictionStart = CFAbsoluteTimeGetCurrent()
            try predictSingle(model: model, provider: provider, outputBuffer: outputBuffer, allowBacking: allowBacking)
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
        for _ in 0..<slots {
            var input: CVPixelBuffer?
            var output: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &input) == kCVReturnSuccess, let input,
                  CVPixelBufferPoolCreatePixelBuffer(nil, tileOutputPool, &output) == kCVReturnSuccess, let output else {
                throw ProcessorError.conversionFailed("could not allocate concurrent Real-CUGAN tiles")
            }
            inputs.append(input)
            outputs.append(output)
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
            let predictionStart = CFAbsoluteTimeGetCurrent()
            let group = DispatchGroup()
            let errorLock = NSLock()
            var firstError: Error?
            for index in 0..<count {
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    defer { group.leave() }
                    do {
                        let provider = try MLDictionaryFeatureProvider(dictionary: [
                            "image": MLFeatureValue(pixelBuffer: inputs[index]),
                            "alpha": MLFeatureValue(multiArray: alpha)
                        ])
                        let options = MLPredictionOptions()
                        options.outputBackings = try RealCUGANOutputBacking.imageBacking(named: "output", model: model, reusing: outputs[index]).dictionary
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