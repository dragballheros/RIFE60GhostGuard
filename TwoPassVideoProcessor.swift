import Foundation
import AVFoundation

final class TwoPassVideoProcessor {
    private let config: ProcessorConfiguration

    init(configuration: ProcessorConfiguration) {
        self.config = configuration
    }

    func process(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void = { _ in }
    ) async throws -> URL {
        var restorationTelemetry = PerformanceTelemetry()
        var restoredURL: URL?
        defer {
            if let restoredURL {
                try? FileManager.default.removeItem(at: restoredURL)
            }
        }

        let needsRestoration = config.compressionProtection || config.outlineProtection
        let rifeSourceURL: URL

        if needsRestoration {
            progress(0.001, "Pass 1/2 • Starting restoration…")
            let restoration = RestorationPass(
                compressionEnabled: config.compressionProtection,
                outlineEnabled: config.outlineProtection
            )
            let result = try await restoration.run(
                sourceURL: sourceURL,
                progress: progress,
                telemetry: { t in
                    restorationTelemetry = t
                    telemetry(t)
                }
            )
            restoredURL = result.url
            rifeSourceURL = result.url

            // Pass 1 has returned, so its Core ML/Core Image objects can be torn
            // down before the RIFE Metal graphs are created.
            autoreleasepool { }

            // If iOS is already throttling heavily, give it a short recovery window.
            // This is capped so cooldown cannot dominate total processing time.
            if ProcessInfo.processInfo.thermalState == .serious ||
               ProcessInfo.processInfo.thermalState == .critical {
                progress(0.285, "Thermal handoff • Cooling briefly before RIFE HQ…")
                let deadline = Date().addingTimeInterval(4.0)
                while Date() < deadline {
                    try Task.checkCancellation()
                    let state = ProcessInfo.processInfo.thermalState
                    if state == .nominal || state == .fair { break }
                    try await Task.sleep(nanoseconds: 250_000_000)
                }
            }
        } else {
            rifeSourceURL = sourceURL
        }

        progress(0.29, "Pass 2/2 • Loading RIFE HQ only…")

        let pass2Config = ProcessorConfiguration(
            quality: config.quality,
            ghostProtection: config.ghostProtection,
            sceneCutProtection: config.sceneCutProtection,
            compressionProtection: false,
            outlineProtection: false,
            ghostSensitivity: config.ghostSensitivity,
            preserveAudio: false,
            targetFPS: config.targetFPS
        )

        let rife = RIFEVideoProcessor(configuration: pass2Config)
        let silentResult = try await rife.process(
            sourceURL: rifeSourceURL,
            progress: { p, message in
                let mapped = 0.29 + min(max(p / 0.95, 0), 1) * 0.65
                progress(mapped, "Pass 2/2 • \(message)")
            },
            telemetry: { pass2 in
                var combined = pass2
                combined.sourceFrames = max(pass2.sourceFrames, restorationTelemetry.sourceFrames)
                combined.compressionMsPerFrame = restorationTelemetry.compressionMsPerFrame
                combined.outlineMsPerFrame = restorationTelemetry.outlineMsPerFrame
                telemetry(combined)
            }
        )

        if config.preserveAudio {
            progress(0.95, "Finalizing • Restoring original audio…")
            let muxer = FinalAudioMuxer()
            let final = try await muxer.addOriginalAudio(videoURL: silentResult, sourceURL: sourceURL)
            progress(1.0, "Finished • Two-pass HQ")
            return final
        }

        progress(1.0, "Finished • Two-pass HQ")
        return silentResult
    }
}

final class FinalAudioMuxer {
    func addOriginalAudio(videoURL: URL, sourceURL: URL) async throws -> URL {
        let sourceAsset = AVURLAsset(url: sourceURL)
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-TwoPass-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: outputURL)

        let composition = AVMutableComposition()
        let processed = AVURLAsset(url: videoURL)

        guard let pv = try await processed.loadTracks(withMediaType: .video).first,
              let cv = composition.addMutableTrack(
                withMediaType: .video,
                preferredTrackID: kCMPersistentTrackID_Invalid
              ) else {
            throw ProcessorError.noOutput
        }

        let pDuration = try await processed.load(.duration)
        try cv.insertTimeRange(CMTimeRange(start: .zero, duration: pDuration), of: pv, at: .zero)
        cv.preferredTransform = try await pv.load(.preferredTransform)

        if let audio = try await sourceAsset.loadTracks(withMediaType: .audio).first,
           let ca = composition.addMutableTrack(
                withMediaType: .audio,
                preferredTrackID: kCMPersistentTrackID_Invalid
           ) {
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
