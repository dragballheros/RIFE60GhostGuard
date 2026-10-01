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
        var cuganTelemetry = PerformanceTelemetry()
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
        } else {
            audioBitrate = 0
        }

        let restoredCheckpoint = recoveryDirectory?.appendingPathComponent("checkpoint-restored.mov")
        let rifeCheckpoint = recoveryDirectory?.appendingPathComponent("checkpoint-rife60.mov")
        let cuganCheckpoint = recoveryDirectory?.appendingPathComponent("checkpoint-cugan2x.mov")
        let outlineCheckpoint = recoveryDirectory?.appendingPathComponent("checkpoint-final-outline.mov")

        // ------------------------------------------------------------------
        // RESUME PLANNING
        //
        // Every finished stage deletes the checkpoint of the stage before it
        // (restored -> rife -> cugan -> outline) to save storage. Therefore a
        // resume must look for the FURTHEST valid checkpoint first and skip
        // everything upstream of it. Checking stages front-to-back (the old
        // behaviour) found the upstream checkpoints missing and silently
        // re-ran Compression Guard and RIFE from scratch.
        // ------------------------------------------------------------------
        let cuganEnd = config.outlineProtection ? 0.84 : 0.97
        let finalExpectedWidth = upscaleTo4K ? native2xWidth : sourceWidth
        let finalExpectedHeight = upscaleTo4K ? native2xHeight : sourceHeight

        var outlineCheckpointValid = false
        var cuganCheckpointValid = false
        var rifeCheckpointValid = false
        var restoredCheckpointValid = false

        if config.outlineProtection, let outlineCheckpoint {
            outlineCheckpointValid = await validVideo(
                outlineCheckpoint,
                expectedDuration: sourceDuration,
                expectedWidth: finalExpectedWidth,
                expectedHeight: finalExpectedHeight
            )
        }
        if !outlineCheckpointValid, upscaleTo4K, let cuganCheckpoint {
            cuganCheckpointValid = await validVideo(
                cuganCheckpoint,
                expectedDuration: sourceDuration,
                expectedWidth: native2xWidth,
                expectedHeight: native2xHeight
            )
        }
        // RIFE output is only needed if nothing downstream of it is already finished.
        let rifeNeeded = !outlineCheckpointValid && !cuganCheckpointValid
        if rifeNeeded, let rifeCheckpoint {
            rifeCheckpointValid = await validVideo(rifeCheckpoint, expectedDuration: sourceDuration)
        }
        if rifeNeeded, !rifeCheckpointValid, config.compressionProtection, let restoredCheckpoint {
            restoredCheckpointValid = await validVideo(restoredCheckpoint, expectedDuration: sourceDuration)
        }

        if outlineCheckpointValid || cuganCheckpointValid || rifeCheckpointValid || restoredCheckpointValid {
            let resumePoint = outlineCheckpointValid ? "final Sharpie"
                : cuganCheckpointValid ? "Real-CUGAN native 2×"
                : rifeCheckpointValid ? "RIFE HQ"
                : "Compression Guard"
            DiagnosticsLogger.shared.log("Recovery plan: furthest valid checkpoint is \(resumePoint). Stages before it will NOT be re-run.")
        } else if recoveryDirectory != nil {
            DiagnosticsLogger.shared.log("Recovery plan: no valid checkpoints found. Starting from the first stage.")
        }

        // IMPORTANT: Sharpie no longer runs here. Compression cleanup remains before
        // RIFE, while line art is now the LAST visual operation after Real-CUGAN.
        var rifeSourceURL: URL?

        if rifeNeeded && !rifeCheckpointValid {
            if config.compressionProtection {
                if restoredCheckpointValid, let restoredCheckpoint {
                    rifeSourceURL = restoredCheckpoint
                    progress(0.10, "Recovered checkpoint • Compression Guard complete")
                    RecoveryStore.update(progress: 0.10, message: "Recovered completed compression checkpoint", force: true)
                    DiagnosticsLogger.shared.log("Recovery: reused completed Compression Guard checkpoint.")
                } else {
                    if let restoredCheckpoint { try? fm.removeItem(at: restoredCheckpoint) }
                    progress(0.001, "Pass 1/4 • Starting Compression Guard…")
                    let restoration = RestorationPass(
                        compressionEnabled: config.compressionProtection,
                        outlineEnabled: false
                    )
                    let result = try await restoration.run(
                        sourceURL: sourceURL,
                        progress: { p, message in
                            let local = min(max(p / 0.28, 0), 1)
                            let renamed = message
                                .replacingOccurrences(of: "Pass 1/2", with: "Pass 1/4")
                                .replacingOccurrences(of: "restored", with: "cleaned")
                            progress(local * 0.10, renamed)
                        },
                        telemetry: { t in
                            restorationTelemetry = t
                            telemetry(t)
                        }
                    )

                    if let restoredCheckpoint {
                        try persistCheckpoint(from: result.url, to: restoredCheckpoint)
                        try? fm.removeItem(at: result.url)
                        rifeSourceURL = restoredCheckpoint
                        RecoveryStore.update(progress: 0.10, message: "Pass 1/4 checkpoint saved • Compression Guard complete", force: true)
                        DiagnosticsLogger.shared.log("Checkpoint saved: Compression Guard.")
                    } else {
                        transientURLs.append(result.url)
                        rifeSourceURL = result.url
                    }
                    autoreleasepool { }
                    try await thermalHandoff(progress: progress, position: 0.10, next: "RIFE HQ")
                }
            } else {
                rifeSourceURL = sourceURL
            }
        }

        var rifeResult: URL?
        if rifeNeeded {
            if rifeCheckpointValid, let rifeCheckpoint {
                rifeResult = rifeCheckpoint
                progress(0.40, "Recovered checkpoint • RIFE HQ complete")
                RecoveryStore.update(progress: 0.40, message: "Recovered completed RIFE HQ checkpoint", force: true)
                DiagnosticsLogger.shared.log("Recovery: reused completed RIFE HQ checkpoint.")
            } else {
                guard let rifeInput = rifeSourceURL else { throw ProcessorError.noOutput }
                if let rifeCheckpoint { try? fm.removeItem(at: rifeCheckpoint) }
                progress(0.11, "Pass 2/4 • Loading full-frame RIFE HQ…")
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
                    sourceURL: rifeInput,
                    progress: { p, message in
                        let local = min(max(p / 0.95, 0), 1)
                        progress(0.11 + local * 0.29, "Pass 2/4 • \(message)")
                    },
                    telemetry: { pass2 in
                        rifeTelemetry = pass2
                        var combined = pass2
                        combined.sourceFrames = max(pass2.sourceFrames, restorationTelemetry.sourceFrames)
                        combined.compressionMsPerFrame = restorationTelemetry.compressionMsPerFrame
                        combined.outlineMsPerFrame = 0
                        telemetry(combined)
                    }
                )

                if let rifeCheckpoint {
                    try persistCheckpoint(from: generated, to: rifeCheckpoint)
                    try? fm.removeItem(at: generated)
                    rifeResult = rifeCheckpoint
                    RecoveryStore.update(progress: 0.40, message: "Pass 2/4 checkpoint saved • RIFE HQ complete", force: true)
                    DiagnosticsLogger.shared.log("Checkpoint saved: RIFE HQ.")
                    if let restoredCheckpoint { try? fm.removeItem(at: restoredCheckpoint) }
                } else {
                    transientURLs.append(generated)
                    rifeResult = generated
                }
            }
        }

        // Real-CUGAN sees the clean RIFE output, never the Sharpie output. This avoids
        // CUGAN softening/changing the user's final line width.
        var postCUGANSource: URL?
        if !outlineCheckpointValid {
            if upscaleTo4K {
                if cuganCheckpointValid, let cuganCheckpoint {
                    postCUGANSource = cuganCheckpoint
                    progress(cuganEnd, "Recovered checkpoint • Real-CUGAN native 2× complete")
                    RecoveryStore.update(progress: cuganEnd, message: "Recovered completed Real-CUGAN native 2× checkpoint", force: true)
                    DiagnosticsLogger.shared.log("Recovery: reused completed Real-CUGAN native 2× checkpoint.")
                } else {
                    guard let cuganInput = rifeResult else { throw ProcessorError.noOutput }
                    if let cuganCheckpoint { try? fm.removeItem(at: cuganCheckpoint) }
                    autoreleasepool { }
                    try await thermalHandoff(progress: progress, position: 0.40, next: "Real-CUGAN native 2×")
                    progress(0.41, "Pass 3/4 • Loading Real-CUGAN Anime native 2×…")
                    let cugan = RealCUGANPass(intensity: 1.30)
                    let generated = try await cugan.run(
                        sourceURL: cuganInput,
                        finalAudioBitrate: audioBitrate,
                        progress: { p, message in
                            let local = min(max(p, 0), 1)
                            progress(0.41 + local * (cuganEnd - 0.41), message.replacingOccurrences(of: "Pass 3/3", with: "Pass 3/4"))
                        },
                        telemetry: { sample in
                            cuganTelemetry = sample
                            var combined = rifeTelemetry
                            combined.sourceFrames = max(rifeTelemetry.sourceFrames, restorationTelemetry.sourceFrames)
                            combined.compressionMsPerFrame = restorationTelemetry.compressionMsPerFrame
                            combined.outlineMsPerFrame = 0
                            combined.upscaledFrames = sample.upscaledFrames
                            combined.cuganMsPerFrame = sample.cuganMsPerFrame
                            combined.upscaleFPS = sample.upscaleFPS
                            combined.encodeMsPerOutputFrame = sample.encodeMsPerOutputFrame
                            combined.thermalState = sample.thermalState
                            combined.performanceMode = sample.performanceMode
                            combined.availableMemoryMB = sample.availableMemoryMB
                            combined.physicalMemoryMB = sample.physicalMemoryMB
                            telemetry(combined)
                        }
                    )

                    if let cuganCheckpoint {
                        try persistCheckpoint(from: generated, to: cuganCheckpoint)
                        try? fm.removeItem(at: generated)
                        postCUGANSource = cuganCheckpoint
                        RecoveryStore.update(progress: cuganEnd, message: "Pass 3/4 checkpoint saved • Real-CUGAN native 2× complete", force: true)
                        DiagnosticsLogger.shared.log("Checkpoint saved: Real-CUGAN native 2×.")
                        if let rifeCheckpoint { try? fm.removeItem(at: rifeCheckpoint) }
                    } else {
                        transientURLs.append(generated)
                        postCUGANSource = generated
                    }
                }
            } else {
                postCUGANSource = rifeResult
                if !config.outlineProtection { progress(0.97, "RIFE HQ complete") }
            }
        }

        // FINAL VISUAL PASS: apply the slightly narrower Sharpie model only after
        // CUGAN. If CUGAN is disabled it still remains after RIFE.
        let videoForMux: URL
        if config.outlineProtection {
            if outlineCheckpointValid, let outlineCheckpoint {
                videoForMux = outlineCheckpoint
                progress(0.97, "Recovered checkpoint • Final Sharpie outline complete")
                RecoveryStore.update(progress: 0.97, message: "Recovered completed final Sharpie checkpoint", force: true)
                DiagnosticsLogger.shared.log("Recovery: reused completed post-CUGAN Sharpie checkpoint.")
            } else {
                guard let outlineInput = postCUGANSource else { throw ProcessorError.noOutput }
                if let outlineCheckpoint { try? fm.removeItem(at: outlineCheckpoint) }
                autoreleasepool { }
                try await thermalHandoff(progress: progress, position: upscaleTo4K ? 0.84 : 0.40, next: "final Sharpie outline")
                let outlineStart = upscaleTo4K ? 0.85 : 0.41
                progress(outlineStart, upscaleTo4K ? "Pass 4/4 • Applying final Sharpie after Real-CUGAN…" : "Pass 3/3 • Applying final Sharpie after RIFE…")
                let finalOutline = FinalOutlinePass()
                let generated = try await finalOutline.run(
                    sourceURL: outlineInput,
                    finalAudioBitrate: audioBitrate,
                    progress: { p, message in
                        let local = min(max(p, 0), 1)
                        progress(outlineStart + local * (0.97 - outlineStart), message)
                    },
                    telemetry: { outlineSample in
                        var combined = rifeTelemetry
                        combined.sourceFrames = max(rifeTelemetry.sourceFrames, restorationTelemetry.sourceFrames)
                        combined.compressionMsPerFrame = restorationTelemetry.compressionMsPerFrame
                        combined.outlineMsPerFrame = outlineSample.outlineMsPerFrame
                        combined.upscaledFrames = cuganTelemetry.upscaledFrames
                        combined.cuganMsPerFrame = cuganTelemetry.cuganMsPerFrame
                        combined.upscaleFPS = cuganTelemetry.upscaleFPS
                        combined.encodeMsPerOutputFrame = outlineSample.encodeMsPerOutputFrame
                        combined.thermalState = outlineSample.thermalState
                        combined.performanceMode = outlineSample.performanceMode
                        combined.availableMemoryMB = outlineSample.availableMemoryMB
                        combined.physicalMemoryMB = outlineSample.physicalMemoryMB
                        telemetry(combined)
                    }
                )

                if let outlineCheckpoint {
                    try persistCheckpoint(from: generated, to: outlineCheckpoint)
                    try? fm.removeItem(at: generated)
                    videoForMux = outlineCheckpoint
                    RecoveryStore.update(progress: 0.97, message: "Final Sharpie checkpoint saved • post-CUGAN outline complete", force: true)
                    DiagnosticsLogger.shared.log("Checkpoint saved: final post-CUGAN Sharpie outline.")
                    if let cuganCheckpoint { try? fm.removeItem(at: cuganCheckpoint) }
                    if !upscaleTo4K, let rifeCheckpoint { try? fm.removeItem(at: rifeCheckpoint) }
                } else {
                    transientURLs.append(generated)
                    videoForMux = generated
                }
            }
        } else {
            guard let finalInput = postCUGANSource else { throw ProcessorError.noOutput }
            videoForMux = finalInput
        }

        let finalVideoInput: URL
        if recoveryDirectory != nil {
            let ext = videoForMux.pathExtension.isEmpty ? "mov" : videoForMux.pathExtension
            let copy = fm.temporaryDirectory
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
            let description = upscaleTo4K
                ? (config.outlineProtection ? "Finished • Native 2× 60fps • Real-CUGAN → Final Sharpie" : "Finished • Native 2× 60fps • Real-CUGAN")
                : (config.outlineProtection ? "Finished • 60fps • Final Sharpie" : "Finished • 60fps")
            progress(1.0, description)
            return final
        }

        transientURLs.removeAll { $0 == finalVideoInput }
        let description = upscaleTo4K
            ? (config.outlineProtection ? "Finished • Native 2× 60fps • Real-CUGAN → Final Sharpie" : "Finished • Native 2× 60fps • Real-CUGAN")
            : (config.outlineProtection ? "Finished • 60fps • Final Sharpie" : "Finished • 60fps")
        progress(1.0, description)
        return finalVideoInput
    }

    private func persistCheckpoint(from source: URL, to destination: URL) throws {
        let fm = FileManager.default
        try? fm.removeItem(at: destination)
        do {
            try fm.moveItem(at: source, to: destination)
            DiagnosticsLogger.shared.log("Checkpoint persisted by move • avoided temporary file duplication")
        } catch {
            DiagnosticsLogger.shared.log("Checkpoint move unavailable • falling back to copy: \(error.localizedDescription)")
            try fm.copyItem(at: source, to: destination)
        }
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
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-Native2X60-\(UUID().uuidString).\(ext)")
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
