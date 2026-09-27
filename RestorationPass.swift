import Foundation
import AVFoundation
import CoreVideo

struct RestorationPassResult: Sendable {
    let url: URL
    let sourceFrames: Int
    let compressionMsPerFrame: Double
    let outlineMsPerFrame: Double
}

final class RestorationPass {
    private let compressionEnabled: Bool
    private let outlineEnabled: Bool

    init(compressionEnabled: Bool, outlineEnabled: Bool) {
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
        let compressionGuard = compressionEnabled ? CompressionGuard() : nil
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

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let decoded = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            var frame = decoded
            sourceFrames += 1

            if let compressionGuard {
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
                try await Task.sleep(nanoseconds: 1_000_000)
            }

            guard adaptor.append(frame, withPresentationTime: pts) else {
                throw ProcessorError.writer("failed writing restored source frame")
            }

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
                progress(frac * 0.28, "Pass 1/2 • \(sourceFrames) restored • Thermal \(currentThermalStateName())")
                await Task.yield()
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
