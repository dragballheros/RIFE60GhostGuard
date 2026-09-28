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
        recoveryDirectory: URL? = nil,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void = { _ in }
    ) async throws -> URL {
        var restorationTelemetry = PerformanceTelemetry()
        var rifeTelemetry = PerformanceTelemetry()
        var transientURLs: [URL] = []
        defer {
            for url in transientURLs { try? FileManager.default.removeItem(at: url) }
        }

        let fm = FileManager.default
        if let recoveryDirectory {
            try fm.createDirectory(at: recoveryDirectory, withIntermediateDirectories: true)
        }

        let sourceAsset = AVURLAsset(url: sourceURL)
        let sourceDuration = try await sourceAsset.load(.duration)
        guard let sourceVideoTrack = try await sourceAsset.loadTracks(withMediaType: .video).first else {
            throw ProcessorError.missingVideoTrack
        }
        let sourceSize = try await sourceVideoTrack.load(.naturalSize)
        let sourceWidth = Int(abs(sourceSize.width).rounded())
        let sourceHeight = Int(abs(sourceSize.height).rounded())
        let native2xWidth = sourceWidth * 2
        let native2xHeight = sourceHeight * 2

        let audioBitrate: Double
        if let audio = try await sourceAsset.loadTracks(withMediaType: .audio).first {
            audioBitrate = Double(try await audio.load(.estimatedDataRate))
        } else { audioBitrate = 0 }

        let restoredCheckpoint = recoveryDirectory?.appendingPathComponent("checkpoint-restored.mov")
        let rifeCheckpoint = recoveryDirectory?.appendingPathComponent("checkpoint-rife60.mov")
        let cuganCheckpoint = recoveryDirectory?.appendingPathComponent("checkpoint-cugan2x.mov")

        let needsRestoration = config.compressionProtection || config.outlineProtection
        let rifeSourceURL: URL

        if needsRestoration {
            if let restoredCheckpoint,
               await validVideo(restoredCheckpoint, expectedDuration: sourceDuration) {
                rifeSourceURL = restoredCheckpoint
                progress(0.18, "Recovered checkpoint • Restoration complete")
                RecoveryStore.update(progress: 0.18, message: "Recovered completed restoration checkpoint", force: true)
                DiagnosticsLogger.shared.log("Recovery: reused completed restoration checkpoint.")
            } else {
                if let restoredCheckpoint { try? fm.removeItem(at: restoredCheckpoint) }
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
                    telemetry: { t in restorationTelemetry = t; telemetry(t) }
                )

                if let restoredCheckpoint {
                    try persistCheckpoint(from: result.url, to: restoredCheckpoint)
                    try? fm.removeItem(at: result.url)
                    rifeSourceURL = restoredCheckpoint
                    RecoveryStore.update(progress: 0.18, message: "Pass 1/3 checkpoint saved • Restoration complete", force: true)
                    DiagnosticsLogger.shared.log("Checkpoint saved: restoration.")
                } else {
                    transientURLs.append(result.url)
                    rifeSourceURL = result.url
                }
                autoreleasepool { }
                try await thermalHandoff(progress: progress, position: 0.18, next: "RIFE HQ")
            }
        } else {
            rifeSourceURL = sourceURL
        }

        let rifeResult: URL
        if let rifeCheckpoint,
           await validVideo(rifeCheckpoint, expectedDuration: sourceDuration) {
            rifeResult = rifeCheckpoint
            progress(0.54, "Recovered checkpoint • RIFE HQ complete")
            RecoveryStore.update(progress: 0.54, message: "Recovered completed RIFE HQ checkpoint", force: true)
            DiagnosticsLogger.shared.log("Recovery: reused completed RIFE HQ checkpoint.")
        } else {
            if let rifeCheckpoint { try? fm.removeItem(at: rifeCheckpoint) }
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
            let generated = try await rife.process(
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

            if let rifeCheckpoint {
                try persistCheckpoint(from: generated, to: rifeCheckpoint)
                try? fm.removeItem(at: generated)
                rifeResult = rifeCheckpoint
                RecoveryStore.update(progress: 0.54, message: "Pass 2/3 checkpoint saved • RIFE HQ complete", force: true)
                DiagnosticsLogger.shared.log("Checkpoint saved: RIFE HQ.")
                if let restoredCheckpoint { try? fm.removeItem(at: restoredCheckpoint) }
            } else {
                transientURLs.append(generated)
                rifeResult = generated
            }
        }

        let videoForMux: URL
        if upscaleTo4K {
            if let cuganCheckpoint,
               await validVideo(
                    cuganCheckpoint,
                    expectedDuration: sourceDuration,
                    expectedWidth: native2xWidth,
                    expectedHeight: native2xHeight
               ) {
                videoForMux = cuganCheckpoint
                progress(0.97, "Recovered checkpoint • Real-CUGAN native 2× complete")
                RecoveryStore.update(progress: 0.97, message: "Recovered completed Real-CUGAN native 2× checkpoint", force: true)
                DiagnosticsLogger.shared.log("Recovery: reused completed Real-CUGAN native 2× checkpoint.")
            } else {
                if let cuganCheckpoint { try? fm.removeItem(at: cuganCheckpoint) }
                autoreleasepool { }
                try await thermalHandoff(progress: progress, position: 0.55, next: "Real-CUGAN native 2×")
                progress(0.56, "Pass 3/3 • Loading Real-CUGAN Anime native 2×…")
                let cugan = RealCUGANPass(intensity: 1.30)
                let generated = try await cugan.run(
                    sourceURL: rifeResult,
                    finalAudioBitrate: audioBitrate,
                    progress: { p, message in progress(0.56 + min(max(p, 0), 1) * 0.41, message) },
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

                if let cuganCheckpoint {
                    try persistCheckpoint(from: generated, to: cuganCheckpoint)
                    try? fm.removeItem(at: generated)
                    videoForMux = cuganCheckpoint
                    RecoveryStore.update(progress: 0.97, message: "Pass 3/3 checkpoint saved • Real-CUGAN native 2× complete", force: true)
                    DiagnosticsLogger.shared.log("Checkpoint saved: Real-CUGAN native 2×.")
                    if let rifeCheckpoint { try? fm.removeItem(at: rifeCheckpoint) }
                } else {
                    transientURLs.append(generated)
                    videoForMux = generated
                }
            }
        } else {
            videoForMux = rifeResult
        }

        let finalVideoInput: URL
        if recoveryDirectory != nil {
            let ext = videoForMux.pathExtension.isEmpty ? "mov" : videoForMux.pathExtension
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent("RIFE60-recovered-final-\(UUID().uuidString).\(ext)")
            try? fm.removeItem(at: copy)
            try fm.copyItem(at: videoForMux, to: copy)
            transientURLs.append(copy)
            finalVideoInput = copy
        } else {
            finalVideoInput = videoForMux
        }

        if config.preserveAudio {
            progress(0.98, "Finalizing • Restoring original audio…")
            let final = try await FinalAudioMuxer().addOriginalAudio(videoURL: finalVideoInput, sourceURL: sourceURL)
            progress(1.0, upscaleTo4K ? "Finished • Native 2× 60fps • Real-CUGAN" : "Finished • 60fps")
            return final
        }

        transientURLs.removeAll { $0 == finalVideoInput }
        progress(1.0, upscaleTo4K ? "Finished • Native 2× 60fps • Real-CUGAN" : "Finished • 60fps")
        return finalVideoInput
    }

    private func persistCheckpoint(from source: URL, to destination: URL) throws {
        let fm = FileManager.default
        try? fm.removeItem(at: destination)
        try fm.copyItem(at: source, to: destination)
    }

    private func validVideo(
        _ url: URL,
        expectedDuration: CMTime,
        expectedWidth: Int? = nil,
        expectedHeight: Int? = nil
    ) async -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path),
              let attrs = try? fm.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber,
              size.int64Value > 1_000_000 else { return false }

        let asset = AVURLAsset(url: url)
        do {
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { return false }
            let duration = try await asset.load(.duration)
            let expectedSeconds = CMTimeGetSeconds(expectedDuration)
            let actualSeconds = CMTimeGetSeconds(duration)
            guard expectedSeconds.isFinite, actualSeconds.isFinite,
                  abs(expectedSeconds - actualSeconds) <= 0.75 else { return false }
            if let expectedWidth, let expectedHeight {
                let size = try await track.load(.naturalSize)
                let w = Int(abs(size.width).rounded())
                let h = Int(abs(size.height).rounded())
                guard (w == expectedWidth && h == expectedHeight) ||
                      (w == expectedHeight && h == expectedWidth) else { return false }
            }
            return true
        } catch {
            DiagnosticsLogger.shared.log("Checkpoint validation failed for \(url.lastPathComponent): \(error.localizedDescription)")
            return false
        }
    }

    private func thermalHandoff(progress: @escaping @Sendable (Double, String) -> Void, position: Double, next: String) async throws {
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
        let outputType: AVFileType = exporter.supportedFileTypes.contains(.mp4) ? .mp4 : .mov
        let ext = outputType == .mp4 ? "mp4" : "mov"
        let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-Native2X60-\(UUID().uuidString).\(ext)")
        try? FileManager.default.removeItem(at: outputURL)
        exporter.outputURL = outputURL
        exporter.outputFileType = outputType
        exporter.shouldOptimizeForNetworkUse = true
        DiagnosticsLogger.shared.log("Final mux begin • container=\(ext) • passthrough HEVC Main10 + original audio")
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            exporter.exportAsynchronously { continuation.resume() }
        }
        guard exporter.status == .completed else {
            throw ProcessorError.writer(exporter.error?.localizedDescription ?? "audio mux failed")
        }
        let finalAsset = AVURLAsset(url: outputURL)
        guard try await finalAsset.loadTracks(withMediaType: .video).first != nil else {
            throw ProcessorError.noOutput
        }
        DiagnosticsLogger.shared.log("Final mux completed • \(outputURL.lastPathComponent)")
        try? FileManager.default.removeItem(at: videoURL)
        return outputURL
    }
}