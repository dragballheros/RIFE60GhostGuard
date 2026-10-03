import Foundation
import AVFoundation
import CoreVideo
import CoreML

struct RestorationPassResult: Sendable {
    let url: URL
    let sourceFrames: Int
    let compressionMsPerFrame: Double
    let outlineMsPerFrame: Double
}

final class RestorationPass {
    private let compressionEnabled: Bool
    private let outlineEnabled: Bool
    private let watermark: WatermarkConfiguration

    init(compressionEnabled: Bool, outlineEnabled: Bool, watermark: WatermarkConfiguration = WatermarkConfiguration()) {
        self.watermark = watermark
        self.compressionEnabled = compressionEnabled
        self.outlineEnabled = outlineEnabled
    }

    func run(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void
    ) async throws -> RestorationPassResult {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ProcessorError.missingVideoTrack
        }

        let duration = try await asset.load(.duration)
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())

        let intermediateURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("restored-source-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: intermediateURL)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw ProcessorError.reader("cannot attach restoration video output")
        }
        reader.add(output)

        let writer = try AVAssetWriter(outputURL: intermediateURL, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.proRes422HQ,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
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
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
        )

        guard writer.canAdd(input) else {
            throw ProcessorError.writer("cannot attach ProRes restoration writer")
        }
        writer.add(input)

        // Keep these resources scoped strictly to pass 1. They are destroyed when
        // this function returns, before the RIFE Metal graphs are created.
        let encodedWatermark = watermark.inEncodedOrientation(size: naturalSize, transform: transform)
        let watermarkRemover: AnimeWatermarkRemover?
        if encodedWatermark.enabled {
            watermarkRemover = try AnimeWatermarkRemover(configuration: encodedWatermark)
        } else {
            watermarkRemover = nil
        }
        let compressionGuard = compressionEnabled ? CompressionGuard() : nil
        let socialCompressionRestorer: SocialCompressionRestorer?
        if compressionEnabled {
            let modelURL = Bundle.main.url(forResource: "SocialCompressionGuard", withExtension: "mlmodelc")
            socialCompressionRestorer = modelURL.flatMap { try? SocialCompressionRestorer(modelURL: $0) }
            if socialCompressionRestorer != nil {
                DiagnosticsLogger.shared.log("Social Compression Guard Core ML model loaded • compute=all • source-cadence restoration enabled")
            } else {
                DiagnosticsLogger.shared.log("Social Compression Guard Core ML model unavailable • falling back to existing CompressionGuard")
            }
        } else {
            socialCompressionRestorer = nil
        }
        let outlineEnhancer: OutlineEnhancer?
        if outlineEnabled {
            progress(0.005, "Pass 1/2 • Loading outline restoration…")
            outlineEnhancer = try autoreleasepool { try OutlineEnhancer() }
        } else {
            outlineEnhancer = nil
        }

        guard reader.startReading() else {
            throw ProcessorError.reader(reader.error?.localizedDescription ?? "restoration reader failed")
        }
        guard writer.startWriting() else {
            throw ProcessorError.writer(writer.error?.localizedDescription ?? "restoration writer failed")
        }
        writer.startSession(atSourceTime: .zero)

        var sourceFrames = 0
        var cleaned = 0
        var outlined = 0
        var compressionSeconds = 0.0
        var outlineSeconds = 0.0
        var watermarkSeconds = 0.0

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let decoded = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            var frame = decoded
            sourceFrames += 1

            if let watermarkRemover {
                let started = CFAbsoluteTimeGetCurrent()
                frame = try watermarkRemover.apply(frame)
                watermarkSeconds += CFAbsoluteTimeGetCurrent() - started
                if sourceFrames == 1 || sourceFrames % 6 == 0 {
                    DiagnosticsLogger.shared.log("Anime watermark frame \(sourceFrames) • \(String(format: "%.1f", watermarkSeconds * 1000 / Double(sourceFrames)))ms/frame")
                }
            }

            if let socialCompressionRestorer {
                let started = CFAbsoluteTimeGetCurrent()
                if let result = try? socialCompressionRestorer.apply(frame) {
                    frame = result
                    cleaned += 1
                    compressionSeconds += CFAbsoluteTimeGetCurrent() - started
                    if sourceFrames == 1 || sourceFrames % 6 == 0 {
                        DiagnosticsLogger.shared.log("Social Compression Guard neural restoration • (cleaned) frames • (String(format: "%.1f", compressionSeconds * 1000 / Double(max(cleaned, 1))))ms/frame")
                    }
                } else if let compressionGuard {
                    if let result = try? compressionGuard.clean(frame) {
                        frame = result
                        cleaned += 1
                        compressionSeconds += CFAbsoluteTimeGetCurrent() - started
                    }
                }
            } else if let compressionGuard {
                let started = CFAbsoluteTimeGetCurrent()
                if let result = try? compressionGuard.clean(frame) {
                    frame = result
                    cleaned += 1
                    compressionSeconds += CFAbsoluteTimeGetCurrent() - started
                }
            }

            if let outlineEnhancer {
                let started = CFAbsoluteTimeGetCurrent()
                frame = try autoreleasepool { try outlineEnhancer.enhance(frame) }
                outlineSeconds += CFAbsoluteTimeGetCurrent() - started
                outlined += 1
            }

            while !input.isReadyForMoreMediaData {
                try Task.checkCancellation()
                try await Task.sleep(nanoseconds: automaticPerformanceModeEnabled() ? 250_000 : 1_000_000)
            }

            guard adaptor.append(frame, withPresentationTime: pts) else {
                throw ProcessorError.writer("failed writing restored source frame")
            }

            // Thermal mode is sampled only after the current frame is complete.
            try await thermalFrameBoundaryPacing()

            if sourceFrames % 3 == 0 {
                var t = PerformanceTelemetry()
                t.sourceFrames = sourceFrames
                t.compressionMsPerFrame = cleaned > 0 ? compressionSeconds * 1000 / Double(cleaned) : 0
                t.outlineMsPerFrame = outlined > 0 ? outlineSeconds * 1000 / Double(outlined) : 0
                t.thermalState = currentThermalStateName()
                telemetry(t)
            }

            if sourceFrames % 6 == 0 {
                let frac = min(max(CMTimeGetSeconds(pts) / max(CMTimeGetSeconds(duration), 0.001), 0), 1)
                progress(frac * 0.28, "Pass 1/2 • \(watermark.enabled ? "Anime watermark removal • " : "")\(sourceFrames) restored • \(currentThermalStateName())")
                if !automaticPerformanceModeEnabled() { await Task.yield() }
            }
        }

        if reader.status == .failed {
            throw ProcessorError.reader(reader.error?.localizedDescription ?? "restoration decode failed")
        }

        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }

        guard writer.status == .completed else {
            throw ProcessorError.writer(writer.error?.localizedDescription ?? "restoration finish failed")
        }

        var finalTelemetry = PerformanceTelemetry()
        finalTelemetry.sourceFrames = sourceFrames
        finalTelemetry.compressionMsPerFrame = cleaned > 0 ? compressionSeconds * 1000 / Double(cleaned) : 0
        finalTelemetry.outlineMsPerFrame = outlined > 0 ? outlineSeconds * 1000 / Double(outlined) : 0
        finalTelemetry.thermalState = currentThermalStateName()
        telemetry(finalTelemetry)

        return RestorationPassResult(
            url: intermediateURL,
            sourceFrames: sourceFrames,
            compressionMsPerFrame: finalTelemetry.compressionMsPerFrame,
            outlineMsPerFrame: finalTelemetry.outlineMsPerFrame
        )
    }
}
