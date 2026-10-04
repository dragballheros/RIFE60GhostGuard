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
    private var stitchedFramePool: CVPixelBufferPool?
    // Core Image can reuse the Metal texture bindings for IOSurface-backed pixel buffers.
    // Real-CUGAN renders one tile in and one tile out per prediction, so avoiding repeated
    // CVPixelBuffer -> Metal texture setup is particularly valuable on long 4K runs.
    // Keep Core Image's intermediate cache disabled to reduce per-tile memory churn.
    // The CVMetalTextureCache CIContextOption is unavailable in the Xcode 16.4 SDK used
    // by the unsigned build, so retain the IOSurface/Metal-compatible pixel-buffer pools
    // as the reusable pixel-buffer boundary without using a version-fragile option key.
    private lazy var ciContext: CIContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    private let colorPopGrade: ColorPopGrade?
    private var renderInSeconds = 0.0
    private var predictionSeconds = 0.0
    private var stitchSeconds = 0.0

    init(intensity: Double = 1.30, colorPopStrength: Double = 0) {
        self.intensity = intensity
        self.colorPopGrade = colorPopStrength > 0 ? ColorPopGrade(strength: colorPopStrength) : nil
    }

    func run(
        sourceURL: URL,
        finalAudioBitrate: Double,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void
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

        DiagnosticsLogger.shared.log("Real-CUGAN entered • native source=\(width)x\(height) • native 2x target=\(targetWidth)x\(targetHeight) • tile=\(profile.width)x\(profile.height) • core=\(profile.stepX)x\(profile.stepY) • tiles/frame=\(tilesPerFrame) • bufferReuse=true • textureCacheBoundary=IOSurface • duration=\(String(format: "%.3f", seconds))s • thermal=\(currentThermalStateName())")
        progress(0.001, "Pass 3/3 • Real-CUGAN native 2× • \(width)×\(height) → \(targetWidth)×\(targetHeight)…")

        try preparePools(targetWidth: targetWidth, targetHeight: targetHeight, profile: profile)
        DiagnosticsLogger.shared.log("Real-CUGAN tile pools ready • Metal compatible • tile=\(profile.width)x\(profile.height) • overlap=\(profile.overlap) • core=\(profile.stepX)x\(profile.stepY)")

        progress(0.002, "Pass 3/3 • Loading Real-CUGAN Anime 2x Noise 3…")
        DiagnosticsLogger.shared.log("Real-CUGAN tiled model load begin • \(profile.modelName)")
        let preferNeuralEngine = false
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

        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-CUGAN-2X-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: outURL)
        var keepOutput = false
        defer { if !keepOutput { try? FileManager.default.removeItem(at: outURL) } }
        let freeAtStart = availableDiskSpaceBytes(at: outURL)
        if let freeAtStart, freeAtStart < 1_500_000_000 {
            throw ProcessorError.writer("Not enough free storage for the 4K Real-CUGAN pass • \(formatStorageMB(freeAtStart)) MB available • at least 1500 MB required")
        }
        DiagnosticsLogger.shared.log("Real-CUGAN storage preflight • free=\(freeAtStart.map(formatStorageMB) ?? "unknown") MB")
        let writer = try AVAssetWriter(outputURL: outURL, fileType: .mov)
        writer.movieFragmentInterval = CMTime(seconds: 2, preferredTimescale: 600)

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
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: targetWidth,
            AVVideoHeightKey: targetHeight,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ])
        input.expectsMediaDataInRealTime = false
        input.transform = transform
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelBufferWidthKey as String: targetWidth,
            kCVPixelBufferHeightKey as String: targetHeight,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ])
        guard writer.canAdd(input) else { throw ProcessorError.writer("cannot attach native 2x HEVC Main10 writer") }
        writer.add(input)
        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN reader failed") }
        guard writer.startWriting() else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "native 2x writer failed") }
        writer.startSession(atSourceTime: .zero)
        guard let writerPool = adaptor.pixelBufferPool else { throw ProcessorError.writer("native 2x P010 pixel buffer pool unavailable") }
        DiagnosticsLogger.shared.log("Real-CUGAN reader/writer started • target=\(targetWidth)x\(targetHeight) HEVC Main10 • bitrate=\(videoBitrate)")

        let alphaValue = Float(1.0 / max(intensity, 0.01))
        let alpha = try MLMultiArray(shape: [1], dataType: .float16)
        alpha[0] = NSNumber(value: alphaValue)
        var frames = 0
        var inferenceSeconds = 0.0
        var encodeSeconds = 0.0
        let passStart = CFAbsoluteTimeGetCurrent()

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let decodedFrame = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            let frameNumber = frames + 1

            if frameNumber == 1 {
                DiagnosticsLogger.shared.log("Real-CUGAN first frame decoded • pixel=\(CVPixelBufferGetWidth(decodedFrame))x\(CVPixelBufferGetHeight(decodedFrame)) • tiled prediction begin")
            }

            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let upscaled = try upscaleNative2x(decodedFrame, model: model, alpha: alpha, frameNumber: frameNumber, profile: profile)
            inferenceSeconds += CFAbsoluteTimeGetCurrent() - inferenceStart

            let encodeStart = CFAbsoluteTimeGetCurrent()
            try await append10Bit(upscaled, at: pts, input: input, adaptor: adaptor, pool: writerPool, writer: writer)
            encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStart
            frames += 1

            if frames == 1 || frames % 20 == 0 {
                DiagnosticsLogger.shared.log(String(format: "Real-CUGAN breakdown (running averages) • frame=%d • render-in=%.1fms • prediction=%.1fms • stitch=%.1fms • convert+append/backpressure=%.1fms", frames, renderInSeconds * 1000 / Double(frames), predictionSeconds * 1000 / Double(frames), stitchSeconds * 1000 / Double(frames), encodeSeconds * 1000 / Double(frames)))
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

        if reader.status == .failed { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN decode failed") }
        guard frames > 0 else { throw ProcessorError.conversionFailed("Real-CUGAN received zero decoded frames") }
        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in writer.finishWriting { continuation.resume() } }
        guard writer.status == .completed else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "native 2x finish failed") }
        let completedBytes = (try? FileManager.default.attributesOfItem(atPath: outURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        DiagnosticsLogger.shared.log("Real-CUGAN writer completed • native 2x=\(targetWidth)x\(targetHeight) • frames=\(frames) • file=\(formatStorageMB(completedBytes)) MB")
        keepOutput = true
        return outURL
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

    private func upscaleNative2x(
        _ source: CVPixelBuffer,
        model: MLModel,
        alpha: MLMultiArray,
        frameNumber: Int,
        profile: TileProfile
    ) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let targetWidth = sourceWidth * 2
        let targetHeight = sourceHeight * 2

        guard let tileInputPool, let stitchedFramePool else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile pools are unavailable")
        }
        var stitched: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, stitchedFramePool, &stitched) == kCVReturnSuccess, let stitched else {
            throw ProcessorError.conversionFailed("could not allocate native 2x stitched frame")
        }

        // One tile buffer is reused for every synchronous prediction in this frame.
        // This removes repeated IOSurface/CVPixelBuffer pool churn without touching pixels.
        var reusableTile: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &reusableTile) == kCVReturnSuccess, let tileBuffer = reusableTile else {
            throw ProcessorError.conversionFailed("could not allocate reusable Real-CUGAN input tile")
        }

        let sourceImage = CIImage(cvPixelBuffer: source).clampedToExtent()
        var tileIndex = 0
        let totalTiles = Int(ceil(Double(sourceWidth) / Double(profile.stepX))) * Int(ceil(Double(sourceHeight) / Double(profile.stepY)))

        var y = 0
        while y < sourceHeight {
            let coreHeight = min(profile.stepY, sourceHeight - y)
            var x = 0
            while x < sourceWidth {
                try Task.checkCancellation()
                let coreWidth = min(profile.stepX, sourceWidth - x)
                tileIndex += 1

                let inputRect = CGRect(x: x - profile.overlap, y: y - profile.overlap, width: profile.width, height: profile.height)
                let tileImage = sourceImage
                    .cropped(to: inputRect)
                    .transformed(by: CGAffineTransform(translationX: -inputRect.origin.x, y: -inputRect.origin.y))
                let renderStart = CFAbsoluteTimeGetCurrent()
                ciContext.render(tileImage, to: tileBuffer, bounds: CGRect(x: 0, y: 0, width: profile.width, height: profile.height), colorSpace: colorSpace)

                renderInSeconds += CFAbsoluteTimeGetCurrent() - renderStart

                try autoreleasepool {
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "image": MLFeatureValue(pixelBuffer: tileBuffer),
                        "alpha": MLFeatureValue(multiArray: alpha)
                    ])
                    let predictionStart = CFAbsoluteTimeGetCurrent()
                    let prediction = try model.prediction(from: provider)
                    predictionSeconds += CFAbsoluteTimeGetCurrent() - predictionStart
                    guard let modelOutput = prediction.featureValue(for: "output")?.imageBufferValue else {
                        throw ProcessorError.conversionFailed("Real-CUGAN tile did not return an image buffer")
                    }

                    let stitchStart = CFAbsoluteTimeGetCurrent()
                    let outputImage = CIImage(cvPixelBuffer: modelOutput)
                    let cropOrigin = profile.overlap * 2
                    let cropRect = CGRect(x: cropOrigin, y: cropOrigin, width: coreWidth * 2, height: coreHeight * 2)
                    let destinationRect = CGRect(x: x * 2, y: y * 2, width: coreWidth * 2, height: coreHeight * 2)
                    let translated = outputImage
                        .cropped(to: cropRect)
                        .transformed(by: CGAffineTransform(
                            translationX: destinationRect.origin.x - cropRect.origin.x,
                            y: destinationRect.origin.y - cropRect.origin.y
                        ))
                    ciContext.render(translated, to: stitched, bounds: destinationRect, colorSpace: colorSpace)
                    stitchSeconds += CFAbsoluteTimeGetCurrent() - stitchStart
                }

                if frameNumber == 1 && (tileIndex == 1 || tileIndex == totalTiles) {
                    DiagnosticsLogger.shared.log("Real-CUGAN first-frame tile \(tileIndex)/\(totalTiles) complete • core=\(coreWidth)x\(coreHeight)")
                }
                x += profile.stepX
            }
            y += profile.stepY
        }

        guard CVPixelBufferGetWidth(stitched) == targetWidth, CVPixelBufferGetHeight(stitched) == targetHeight else {
            throw ProcessorError.conversionFailed("Real-CUGAN stitched frame size mismatch")
        }

        // Performance Mode still needs bounded IOSurface/Core Image caches during long 4K runs.
        // This does not alter pixels or model execution; it only releases excess reusable resources.
        let memory = currentRenderPerformanceSnapshot()
        let cleanupInterval: Int
        if memory.availableMemoryMB < 1_500 {
            cleanupInterval = 1
        } else if memory.availableMemoryMB < 2_000 {
            cleanupInterval = 4
        } else {
            cleanupInterval = 8
        }
        if memory.availableMemoryMB < 1_200 && frameNumber % cleanupInterval == 0 {
            CVPixelBufferPoolFlush(tileInputPool, .excessBuffers)
            CVPixelBufferPoolFlush(stitchedFramePool, .excessBuffers)
            if frameNumber % 20 == 0 || memory.availableMemoryMB < 1_500 {
                DiagnosticsLogger.shared.log("Real-CUGAN memory cleanup • frame=\(frameNumber) • headroom=\(Int(memory.availableMemoryMB.rounded())) MB • interval=\(cleanupInterval)")
            }
        }

        return stitched
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