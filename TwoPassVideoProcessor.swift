import Foundation
import AVFoundation

final class TwoPassVideoProcessor {
    private let config: ProcessorConfiguration
    private let upscaleTo4K: Bool

    init(configuration: ProcessorConfiguration, upscaleTo4K: Bool = true) {
        self.config = configuration
        self.upscaleTo4K = upscaleTo4K
    }

    func process(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void = { _ in }
    ) async throws -> URL {
        var restorationTelemetry = PerformanceTelemetry()
        var rifeTelemetry = PerformanceTelemetry()
        var restoredURL: URL?
        var rifeIntermediateURL: URL?
        defer {
            if let restoredURL { try? FileManager.default.removeItem(at: restoredURL) }
            if let rifeIntermediateURL { try? FileManager.default.removeItem(at: rifeIntermediateURL) }
        }

        let sourceAsset = AVURLAsset(url: sourceURL)
        let audioBitrate: Double
        if let audio = try await sourceAsset.loadTracks(withMediaType: .audio).first {
            audioBitrate = Double(try await audio.load(.estimatedDataRate))
        } else {
            audioBitrate = 0
        }

        let needsRestoration = config.compressionProtection || config.outlineProtection
        let rifeSourceURL: URL

        if needsRestoration {
            progress(0.001, "Pass 1/3 • Starting restoration…")
            let restoration = RestorationPass(
                compressionEnabled: config.compressionProtection,
                outlineEnabled: config.outlineProtection
            )
            let result = try await restoration.run(
                sourceURL: sourceURL,
                progress: { p, message in
                    let local = min(max(p / 0.28, 0), 1)
                    progress(local * 0.18, message.replacingOccurrences(of: "Pass 1/2", with: "Pass 1/3"))
                },
                telemetry: { t in
                    restorationTelemetry = t
                    telemetry(t)
                }
            )
            restoredURL = result.url
            rifeSourceURL = result.url
            autoreleasepool { }
            try await thermalHandoff(progress: progress, position: 0.18, next: "RIFE HQ")
        } else {
            rifeSourceURL = sourceURL
        }

        progress(0.19, "Pass 2/3 • Loading full-frame RIFE HQ…")
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
        let rifeResult = try await rife.process(
            sourceURL: rifeSourceURL,
            progress: { p, message in
                let local = min(max(p / 0.95, 0), 1)
                progress(0.19 + local * 0.35, "Pass 2/3 • \(message)")
            },
            telemetry: { pass2 in
                rifeTelemetry = pass2
                var combined = pass2
                combined.sourceFrames = max(pass2.sourceFrames, restorationTelemetry.sourceFrames)
                combined.compressionMsPerFrame = restorationTelemetry.compressionMsPerFrame
                combined.outlineMsPerFrame = restorationTelemetry.outlineMsPerFrame
                telemetry(combined)
            }
        )
        rifeIntermediateURL = rifeResult

        let videoForMux: URL
        if upscaleTo4K {
            autoreleasepool { }
            try await thermalHandoff(progress: progress, position: 0.55, next: "Real-CUGAN 4K")
            progress(0.56, "Pass 3/3 • Loading Real-CUGAN Anime 2x…")
            let cugan = RealCUGANPass(intensity: 1.30)
            videoForMux = try await cugan.run(
                sourceURL: rifeResult,
                finalAudioBitrate: audioBitrate,
                progress: { p, message in
                    progress(0.56 + min(max(p, 0), 1) * 0.41, message)
                },
                telemetry: { cuganSample in
                    var combined = rifeTelemetry
                    combined.sourceFrames = max(rifeTelemetry.sourceFrames, restorationTelemetry.sourceFrames)
                    combined.compressionMsPerFrame = restorationTelemetry.compressionMsPerFrame
                    combined.outlineMsPerFrame = restorationTelemetry.outlineMsPerFrame
                    combined.upscaledFrames = cuganSample.upscaledFrames
                    combined.cuganMsPerFrame = cuganSample.cuganMsPerFrame
                    combined.upscaleFPS = cuganSample.upscaleFPS
                    combined.encodeMsPerOutputFrame = cuganSample.encodeMsPerOutputFrame
                    combined.thermalState = cuganSample.thermalState
                    telemetry(combined)
                }
            )
            try? FileManager.default.removeItem(at: rifeResult)
            rifeIntermediateURL = nil
        } else {
            videoForMux = rifeResult
            rifeIntermediateURL = nil
        }

        if config.preserveAudio {
            progress(0.98, "Finalizing • Restoring original audio…")
            let muxer = FinalAudioMuxer()
            let final = try await muxer.addOriginalAudio(videoURL: videoForMux, sourceURL: sourceURL)
            progress(1.0, upscaleTo4K ? "Finished • 4K60 • Real-CUGAN" : "Finished • 1080p60")
            return final
        }

        progress(1.0, upscaleTo4K ? "Finished • 4K60 • Real-CUGAN" : "Finished • 1080p60")
        return videoForMux
    }

    private func thermalHandoff(
        progress: @escaping @Sendable (Double, String) -> Void,
        position: Double,
        next: String
    ) async throws {
        let state = ProcessInfo.processInfo.thermalState
        guard state == .serious || state == .critical else { return }
        progress(position, "Thermal handoff • Cooling briefly before \(next)…")
        let deadline = Date().addingTimeInterval(4.0)
        while Date() < deadline {
            try Task.checkCancellation()
            let current = ProcessInfo.processInfo.thermalState
            if current == .nominal || current == .fair { break }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
    }
}

final class FinalAudioMuxer {
    func addOriginalAudio(videoURL: URL, sourceURL: URL) async throws -> URL {
        let sourceAsset = AVURLAsset(url: sourceURL)
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-4K60-\(UUID().uuidString).mp4")
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
