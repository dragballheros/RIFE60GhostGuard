import Foundation
import AVFoundation
import CoreImage
import VideoToolbox

/// Final visual polish pass. This deliberately runs after Real-CUGAN so the upscaler
/// cannot soften, widen, or otherwise alter the finished Sharpie lines. Compression
/// Guard runs AFTER Sharpie so no later AI/image pass can bring compression artifacts back.
final class FinalOutlinePass {
    private var transferSession: VTPixelTransferSession?
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var blendPool: CVPixelBufferPool?

    // Keep the Sharpie look clearly stronger than the original anime line art, while
    // making it just a little narrower than the previous Sharpie revision.
    private let enhancedWeight: CGFloat = 0.94

    func run(
        sourceURL: URL,
        finalAudioBitrate: Double,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void
    ) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ProcessorError.missingVideoTrack
        }
        let duration = try await asset.load(.duration)
        let seconds = max(CMTimeGetSeconds(duration), 0.001)
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())

        DiagnosticsLogger.shared.log("Final visual pass entered • post-CUGAN/source=\(width)x\(height) • thinner Sharpie → final Compression Guard • thermal=\(currentThermalStateName())")
        progress(0.001, "Final polish • Loading thinner Sharpie + Compression Guard…")
        let enhancer = try autoreleasepool { try OutlineEnhancer() }
        let finalCompressionGuard = CompressionGuard()
        try prepareBlendPool(width: width, height: height)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelBufferPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot attach final polish reader") }
        reader.add(output)

        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-POST-CUGAN-SHARPIE-CLEAN-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: outURL)
        let writer = try AVAssetWriter(outputURL: outURL, fileType: .mov)

        // HQ HEVC Main10 policy: quality target first, <1 GB only acts as a ceiling.
        let targetTotalBytes = 950_000_000.0
        let containerReserveBytes = 16_000_000.0
        let usableBits = max((targetTotalBytes - containerReserveBytes) * 8.0, 8_000_000.0)
        let audioBits = max(finalAudioBitrate, 0) * seconds
        let sizeBudgetBitrate = max((usableBits - audioBits) / seconds * 0.96, 500_000.0)
        let qualityBitrate = Double(width * height) * 60.0 * 0.30
        let codecSafetyCeiling = 160_000_000.0
        let videoBitrate = Int(max(500_000.0, min(qualityBitrate, sizeBudgetBitrate, codecSafetyCeiling)))
        DiagnosticsLogger.shared.log("Final polish bitrate • selected=\(videoBitrate) • qualityTarget=\(Int(qualityBitrate)) • sizeCeiling=\(Int(sizeBudgetBitrate))")

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: videoBitrate,
            AVVideoQualityKey: 1.0,
            AVVideoExpectedSourceFrameRateKey: 60,
            AVVideoMaxKeyFrameIntervalKey: 120,
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
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelBufferPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ])
        guard writer.canAdd(input) else { throw ProcessorError.writer("cannot attach final polish HEVC Main10 writer") }
        writer.add(input)
        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "final polish reader failed") }
        guard writer.startWriting() else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "final polish writer failed") }
        writer.startSession(atSourceTime: .zero)
        guard let writerPool = adaptor.pixelBufferPool else { throw ProcessorError.writer("final polish P010 pool unavailable") }

        var frames = 0
        var cleanedFrames = 0
        var outlineSeconds = 0.0
        var finalCompressionSeconds = 0.0
        var encodeSeconds = 0.0
        let passStart = CFAbsoluteTimeGetCurrent()

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let decoded = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)

            // Sharpie first: CUGAN has already finished, so it cannot change the final width.
            let outlineStart = CFAbsoluteTimeGetCurrent()
            let enhanced = try autoreleasepool { try enhancer.enhance(decoded) }
            let narrowed = try narrowTowardOriginal(enhanced: enhanced, original: decoded)
            outlineSeconds += CFAbsoluteTimeGetCurrent() - outlineStart

            // Compression cleanup is intentionally the LAST image-processing operation.
            // If a frame cannot be cleaned for any reason, preserve the Sharpie result
            // rather than failing or reverting to an earlier frame.
            let compressionStart = CFAbsoluteTimeGetCurrent()
            let polished: CVPixelBuffer
            if let cleaned = try? finalCompressionGuard.clean(narrowed) {
                polished = cleaned
                cleanedFrames += 1
            } else {
                polished = narrowed
            }
            finalCompressionSeconds += CFAbsoluteTimeGetCurrent() - compressionStart

            let encodeStart = CFAbsoluteTimeGetCurrent()
            try await append10Bit(polished, at: pts, input: input, adaptor: adaptor, pool: writerPool)
            encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStart
            frames += 1

            if frames == 1 || frames % 10 == 0 {
                let outlineMs = outlineSeconds * 1000.0 / Double(frames)
                let compressionMs = finalCompressionSeconds * 1000.0 / Double(frames)
                DiagnosticsLogger.shared.log("Final polish frame \(frames) • Sharpie=\(String(format: "%.1f", outlineMs))ms/frame • finalCompression=\(String(format: "%.1f", compressionMs))ms/frame • cleaned=\(cleanedFrames) • thermal=\(currentThermalStateName())")
            }
            if frames % 3 == 0 {
                var t = PerformanceTelemetry()
                t.outlineMsPerFrame = outlineSeconds * 1000.0 / Double(frames)
                t.compressionMsPerFrame = finalCompressionSeconds * 1000.0 / Double(frames)
                t.encodeMsPerOutputFrame = encodeSeconds * 1000.0 / Double(frames)
                t.thermalState = currentThermalStateName()
                telemetry(t)
            }
            if frames % 3 == 0 {
                let frac = min(max(CMTimeGetSeconds(pts) / seconds, 0), 1)
                let fps = Double(frames) / max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)
                let remaining = fps > 0 ? max(seconds * 60.0 - Double(frames), 0) / fps : 0
                progress(frac, "Final polish • Sharpie → Compression Guard • \(frames) frames • ETA \(formatDuration(remaining))")
                await Task.yield()
            }
        }

        if reader.status == .failed { throw ProcessorError.reader(reader.error?.localizedDescription ?? "final polish decode failed") }
        guard frames > 0 else { throw ProcessorError.conversionFailed("Final polish pass received zero frames") }
        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "final polish finish failed") }
        let finalCompressionMs = finalCompressionSeconds * 1000.0 / Double(frames)
        DiagnosticsLogger.shared.log("Final visual pass complete • frames=\(frames) • post-CUGAN thinner Sharpie • final Compression Guard=\(String(format: "%.1f", finalCompressionMs))ms/frame • HEVC Main10")
        return outURL
    }

    private func prepareBlendPool(width: Int, height: Int) throws {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelBufferPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess,
              let pool else {
            throw ProcessorError.conversionFailed("could not create final outline blend pool")
        }
        blendPool = pool
    }

    private func narrowTowardOriginal(enhanced: CVPixelBuffer, original: CVPixelBuffer) throws -> CVPixelBuffer {
        guard let blendPool else { throw ProcessorError.conversionFailed("final outline blend pool unavailable") }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, blendPool, &destination) == kCVReturnSuccess,
              let destination else {
            throw ProcessorError.conversionFailed("could not allocate final outline blend frame")
        }
        let extent = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(original), height: CVPixelBufferGetHeight(original))
        let originalImage = CIImage(cvPixelBuffer: original)
        let enhancedImage = CIImage(cvPixelBuffer: enhanced)
        guard let filter = CIFilter(name: "CIDissolveTransition") else {
            throw ProcessorError.conversionFailed("final outline blend filter unavailable")
        }
        filter.setValue(originalImage, forKey: kCIInputImageKey)
        filter.setValue(enhancedImage, forKey: kCIInputTargetImageKey)
        filter.setValue(enhancedWeight, forKey: kCIInputTimeKey)
        guard let blended = filter.outputImage?.cropped(to: extent) else {
            throw ProcessorError.conversionFailed("final outline blend failed")
        }
        ciContext.render(blended, to: destination, bounds: extent, colorSpace: CGColorSpaceCreateDeviceRGB())
        return destination
    }

    private func append10Bit(
        _ source: CVPixelBuffer,
        at time: CMTime,
        input: AVAssetWriterInput,
        adaptor: AVAssetWriterInputPixelBufferAdaptor,
        pool: CVPixelBufferPool
    ) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess,
              let destination else {
            throw ProcessorError.conversionFailed("could not allocate final polish P010 frame")
        }
        if transferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) == noErr,
                  let session else {
                throw ProcessorError.conversionFailed("could not create final polish pixel transfer session")
            }
            transferSession = session
        }
        guard let transferSession,
              VTPixelTransferSessionTransferImage(transferSession, from: source, to: destination) == noErr else {
            throw ProcessorError.conversionFailed("final polish BGRA→P010 conversion failed")
        }
        guard adaptor.append(destination, withPresentationTime: time) else {
            throw ProcessorError.writer("failed appending final polish frame")
        }
    }
}
