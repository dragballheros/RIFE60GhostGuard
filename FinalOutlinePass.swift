import Foundation
import AVFoundation
import CoreImage
import CoreVideo
import VideoToolbox

/// Final visual polish after Real-CUGAN. The 4K CUGAN frame remains the master.
/// Deep-shadow chroma noise is cleaned first, then Sharpie is generated from that
/// cleaned master, then only the light final compression polish is allowed after it.
final class FinalOutlinePass {
    private var transferSession: VTPixelTransferSession?
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private var blendPool: CVPixelBufferPool?
    private var workingPool: CVPixelBufferPool?
    private var fullOutlinePool: CVPixelBufferPool?

    private let colorPopGrade: ColorPopGrade?

    init(colorPopStrength: Double = 0) {
        self.colorPopGrade = colorPopStrength > 0 ? ColorPopGrade(strength: colorPopStrength) : nil
    }

    // Use a slightly higher-resolution style map so the Sharpie line expands less
    // when returned to 4K. Keep the blend strength unchanged so the line stays bold.
    private let enhancedWeight: CGFloat = 0.82
    private let maxSharpieLongEdge = 3072

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
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())

        let scale = min(1.0, Double(maxSharpieLongEdge) / Double(max(width, height)))
        let workingWidth = max(2, Int((Double(width) * scale / 2.0).rounded() * 2.0))
        let workingHeight = max(2, Int((Double(height) * scale / 2.0).rounded() * 2.0))
        DiagnosticsLogger.shared.log("Final visual pass entered • 4K master=\(width)x\(height) • Shadow Chroma → Sharpie → Final Compression • Sharpie style map=\(workingWidth)x\(workingHeight) • thermal=\(currentThermalStateName())")
        progress(0.001, "Final polish • Shadow cleanup → Sharpie → Compression Guard…")

        let enhancer = try autoreleasepool { try OutlineEnhancer() }
        let compressionGuard = CompressionGuard()
        try preparePools(fullWidth: width, fullHeight: height, workingWidth: workingWidth, workingHeight: workingHeight)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot attach final polish reader") }
        reader.add(output)

        let checkpointRoot = recoveryDirectory ?? FileManager.default.temporaryDirectory
        let freeAtStart = availableDiskSpaceBytes(at: checkpointRoot)
        if let freeAtStart, freeAtStart < 1_200_000_000 {
            throw ProcessorError.writer("Not enough free storage for the final 4K Sharpie pass • \(formatStorageMB(freeAtStart)) MB available • at least 1200 MB required")
        }
        DiagnosticsLogger.shared.log("Final Sharpie storage preflight • free=\(freeAtStart.map(formatStorageMB) ?? "unknown") MB")
        let targetTotalBytes = 1_080_000_000.0
        let usableBits = max((targetTotalBytes - 16_000_000.0) * 8.0, 8_000_000.0)
        let audioBits = max(finalAudioBitrate, 0) * seconds
        let sizeBudgetBitrate = max((usableBits - audioBits) / seconds * 0.96, 500_000.0)
        let qualityBitrate = Double(width * height) * 60.0 * 1.80
        let videoBitrate = Int(max(500_000.0, min(qualityBitrate, sizeBudgetBitrate)))
        DiagnosticsLogger.shared.log("Final polish bitrate • selected=\(videoBitrate) • high-quality master target=\(Int(targetTotalBytes)) bytes • qualityTarget=\(Int(qualityBitrate)) • sizeCeiling=\(Int(sizeBudgetBitrate)) • delivery optimizer target=900000000 bytes")

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: videoBitrate,
            AVVideoExpectedSourceFrameRateKey: 60,
            AVVideoMaxKeyFrameIntervalKey: 120,
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
            stageID: "final-outline",
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
        if checkpoint.resumeTime > .zero {
            let start = checkpoint.resumeTime
            reader.timeRange = CMTimeRange(start: start, duration: CMTimeMaximum(.zero, CMTimeSubtract(duration, start)))
            DiagnosticsLogger.shared.log("Final Sharpie incremental resume • from \(String(format: "%.3f", CMTimeGetSeconds(start)))s")
        }
        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "final polish reader failed") }

        var frames = 0, shadowCleanedFrames = 0, finalCleanedFrames = 0
        var outlineSeconds = 0.0, shadowSeconds = 0.0, finalCompressionSeconds = 0.0, encodeSeconds = 0.0
        let passStart = CFAbsoluteTimeGetCurrent()

        while let sample = output.copyNextSampleBuffer() {
            try await checkpoint.checkCancellation()
            guard let decoded = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            if CMTimeCompare(pts, checkpoint.resumeTime) <= 0 { continue }

            // IMPORTANT: dark chroma cleanup happens before Sharpie. It preserves
            // source luminance/edges, and the cleaned frame is also used as the blend
            // master so the old noisy CUGAN pixels cannot be mixed back in afterward.
            let shadowStart = CFAbsoluteTimeGetCurrent()
            let preSharpie: CVPixelBuffer
            if let cleaned = try? compressionGuard.cleanShadowChroma(decoded) {
                preSharpie = cleaned
                shadowCleanedFrames += 1
            } else {
                preSharpie = decoded
            }
            shadowSeconds += CFAbsoluteTimeGetCurrent() - shadowStart

            let outlineStart = CFAbsoluteTimeGetCurrent()
            let working = scale < 0.999 ? try resize(preSharpie, width: workingWidth, height: workingHeight, pool: workingPool) : preSharpie
            let enhancedWorking = try autoreleasepool { try enhancer.enhance(working) }
            let enhancedFull = scale < 0.999 ? try resize(enhancedWorking, width: width, height: height, pool: fullOutlinePool) : enhancedWorking
            let narrowed = try narrowTowardOriginal(enhanced: enhancedFull, original: preSharpie)
            outlineSeconds += CFAbsoluteTimeGetCurrent() - outlineStart

            // Only the light general compression polish remains after Sharpie.
            // The spatial shadow/chroma blur is never applied to finished Sharpie lines.
            let compressionStart = CFAbsoluteTimeGetCurrent()
            let polished: CVPixelBuffer
            if let cleaned = try? compressionGuard.cleanFinalCompression(narrowed) {
                polished = cleaned
                finalCleanedFrames += 1
            } else {
                polished = narrowed
            }
            finalCompressionSeconds += CFAbsoluteTimeGetCurrent() - compressionStart

            let encodeStart = CFAbsoluteTimeGetCurrent()
            try await checkpoint.append10Bit(polished, at: pts)
            encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStart
            frames += 1

            if frames == 1 || frames % 10 == 0 {
                DiagnosticsLogger.shared.log("Final polish frame \(frames) • shadowPreSharpie=\(String(format: "%.1f", shadowSeconds*1000/Double(frames)))ms • Sharpie=\(String(format: "%.1f", outlineSeconds*1000/Double(frames)))ms • finalCompression=\(String(format: "%.1f", finalCompressionSeconds*1000/Double(frames)))ms • shadowCleaned=\(shadowCleanedFrames) • finalCleaned=\(finalCleanedFrames) • thermal=\(currentThermalStateName())")
            }
            if frames % 3 == 0 {
                var t = PerformanceTelemetry()
                t.outlineMsPerFrame = outlineSeconds * 1000 / Double(frames)
                t.compressionMsPerFrame = (shadowSeconds + finalCompressionSeconds) * 1000 / Double(frames)
                t.encodeMsPerOutputFrame = encodeSeconds * 1000 / Double(frames)
                t.thermalState = currentThermalStateName()
                telemetry(t)
                let frac = min(max(CMTimeGetSeconds(pts) / seconds, 0), 1)
                let fps = Double(frames) / max(CFAbsoluteTimeGetCurrent()-passStart, 0.001)
                let remaining = fps > 0 ? max(seconds*60-Double(frames),0)/fps : 0
                progress(frac, "Final polish • Shadow cleanup → Sharpie → Compression Guard • \(frames) frames • ETA \(formatDuration(remaining))")
                await Task.yield()
            }
        }

        if reader.status == .failed { throw ProcessorError.reader(reader.error?.localizedDescription ?? "final polish decode failed") }
        guard frames > 0 else { throw ProcessorError.conversionFailed("Final polish pass received zero frames") }
        let completedURL = try await checkpoint.finish()
        let completedBytes = (try? FileManager.default.attributesOfItem(atPath: completedURL.path)[.size] as? NSNumber)?.int64Value ?? 0
        DiagnosticsLogger.shared.log("Final visual pass complete • frames=\(frames) • order=ShadowChroma→Sharpie→FinalCompression • Sharpie style map=\(workingWidth)x\(workingHeight) • Sharpie=\(String(format: "%.1f", outlineSeconds*1000/Double(frames)))ms/frame • HEVC Main10 • file=\(formatStorageMB(completedBytes)) MB")
        return completedURL
    }

    /// Single-image version of the per-frame final polish in run():
    /// Shadow Chroma -> Sharpie -> narrow toward original -> light final compression.
    func polishStill(_ decoded: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(decoded)
        let height = CVPixelBufferGetHeight(decoded)
        let scale = min(1.0, Double(maxSharpieLongEdge) / Double(max(width, height)))
        let workingWidth = max(2, Int((Double(width) * scale / 2.0).rounded() * 2.0))
        let workingHeight = max(2, Int((Double(height) * scale / 2.0).rounded() * 2.0))
        let needsResize = scale < 0.999

        let enhancer = try autoreleasepool { try OutlineEnhancer() }
        let compressionGuard = CompressionGuard()
        try preparePools(fullWidth: width, fullHeight: height, workingWidth: workingWidth, workingHeight: workingHeight)

        var preSharpie = decoded
        if let cleaned = try? compressionGuard.cleanShadowChroma(decoded) { preSharpie = cleaned }

        var working = preSharpie
        if needsResize { working = try resize(preSharpie, width: workingWidth, height: workingHeight, pool: workingPool) }
        let enhancedWorking = try autoreleasepool { try enhancer.enhance(working) }
        var enhancedFull = enhancedWorking
        if needsResize { enhancedFull = try resize(enhancedWorking, width: width, height: height, pool: fullOutlinePool) }
        let narrowed = try narrowTowardOriginal(enhanced: enhancedFull, original: preSharpie)

        if let polished = try? compressionGuard.cleanFinalCompression(narrowed) { return polished }
        return narrowed
    }

    private func makePool(width: Int, height: Int) throws -> CVPixelBufferPool {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess, let pool else { throw ProcessorError.conversionFailed("could not create final outline pool") }
        return pool
    }

    private func preparePools(fullWidth: Int, fullHeight: Int, workingWidth: Int, workingHeight: Int) throws {
        blendPool = try makePool(width: fullWidth, height: fullHeight)
        fullOutlinePool = try makePool(width: fullWidth, height: fullHeight)
        workingPool = try makePool(width: workingWidth, height: workingHeight)
    }

    private func resize(_ source: CVPixelBuffer, width: Int, height: Int, pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        guard let pool else { throw ProcessorError.conversionFailed("Sharpie resize pool unavailable") }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess, let destination else { throw ProcessorError.conversionFailed("could not allocate Sharpie resize frame") }
        let sx = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(source))
        let sy = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(source))
        let image = CIImage(cvPixelBuffer: source).transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        ciContext.render(image, to: destination, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: colorSpace)
        return destination
    }

    private func narrowTowardOriginal(enhanced: CVPixelBuffer, original: CVPixelBuffer) throws -> CVPixelBuffer {
        guard let blendPool else { throw ProcessorError.conversionFailed("final outline blend pool unavailable") }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, blendPool, &destination) == kCVReturnSuccess, let destination else { throw ProcessorError.conversionFailed("could not allocate final outline blend frame") }
        let extent = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(original), height: CVPixelBufferGetHeight(original))
        guard let filter = CIFilter(name: "CIDissolveTransition") else { throw ProcessorError.conversionFailed("final outline blend filter unavailable") }
        filter.setValue(CIImage(cvPixelBuffer: original), forKey: kCIInputImageKey)
        filter.setValue(CIImage(cvPixelBuffer: enhanced), forKey: kCIInputTargetImageKey)
        filter.setValue(enhancedWeight, forKey: kCIInputTimeKey)
        guard let blended = filter.outputImage?.cropped(to: extent) else { throw ProcessorError.conversionFailed("final outline blend failed") }
        ciContext.render(blended, to: destination, bounds: extent, colorSpace: colorSpace)
        return destination
    }

    private func append10Bit(_ source: CVPixelBuffer, at time: CMTime, input: AVAssetWriterInput, adaptor: AVAssetWriterInputPixelBufferAdaptor, pool: CVPixelBufferPool, writer: AVAssetWriter) async throws {
        while !input.isReadyForMoreMediaData { try Task.checkCancellation(); try await Task.sleep(nanoseconds: 1_000_000) }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess, let destination else { throw ProcessorError.conversionFailed("could not allocate final polish P010 frame") }
        if transferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) == noErr, let session else { throw ProcessorError.conversionFailed("could not create final polish pixel transfer session") }
            transferSession = session
        }
        let finalFrame: CVPixelBuffer
        if let colorPopGrade { finalFrame = try colorPopGrade.apply(source) } else { finalFrame = source }
        guard let transferSession, VTPixelTransferSessionTransferImage(transferSession, from: finalFrame, to: destination) == noErr else { throw ProcessorError.conversionFailed("final polish BGRA→P010 conversion failed") }
        guard adaptor.append(destination, withPresentationTime: time) else {
            let detail = writer.error?.localizedDescription ?? "writer status \(writer.status.rawValue)"
            let free = availableDiskSpaceBytes(at: writer.outputURL)
            DiagnosticsLogger.shared.log("Final Sharpie append rejected • \(detail) • free=\(free.map(formatStorageMB) ?? "unknown") MB")
            throw ProcessorError.writer("failed appending final polish frame • \(detail) • free storage \(free.map(formatStorageMB) ?? "unknown") MB")
        }
    }
}