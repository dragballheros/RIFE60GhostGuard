import Foundation
import AVFoundation

final class TwoPassVideoProcessor {
    private let config: ProcessorConfiguration
    private let upscaleTo4K: Bool

    private let colorPopStrength: Double
    private let watermark: WatermarkConfiguration
    private let upscaleFirst: Bool

    init(configuration: ProcessorConfiguration, upscaleTo4K: Bool = true, colorPopEnabled: Bool = false, colorPopStrength: Double = 0.5, watermark: WatermarkConfiguration = WatermarkConfiguration(), upscaleFirst: Bool = false) {
        self.upscaleFirst = upscaleFirst && upscaleTo4K
        self.watermark = watermark
        self.colorPopStrength = colorPopEnabled && colorPopStrength.isFinite ? min(max(colorPopStrength, 0), 1) : 0
        self.config = configuration
        self.upscaleTo4K = upscaleTo4K
    }

    func process(
        sourceURL: URL,
        recoveryDirectory: URL? = nil,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void = { _ in }
    ) async throws -> URL {
        if colorPopStrength > 0 {
            DiagnosticsLogger.shared.log("Color Pop active • strength=\(String(format: "%.2f", colorPopStrength)) • applied once in the final active visual stage")
        }
        if upscaleFirst {
            return try await processUpscaleFirst(sourceURL: sourceURL, recoveryDirectory: recoveryDirectory, progress: progress, telemetry: telemetry)
        }
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

        let needsRestoration = config.compressionProtection || watermark.enabled || config.ghostProtection
        let restorationName: String = {
            if watermark.enabled && config.compressionProtection { return "Anime watermark removal / Compression Guard / Source Frame Guard" }
            if watermark.enabled { return "Anime watermark removal / Source Frame Guard" }
            if config.compressionProtection { return "Compression Guard / Source Frame Guard" }
            return "Source Frame Guard"
        }()
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
        if rifeNeeded, !rifeCheckpointValid, needsRestoration, let restoredCheckpoint {
            restoredCheckpointValid = await validVideo(restoredCheckpoint, expectedDuration: sourceDuration)
        }

        if outlineCheckpointValid || cuganCheckpointValid || rifeCheckpointValid || restoredCheckpointValid {
            let resumePoint = outlineCheckpointValid ? "final Sharpie"
                : cuganCheckpointValid ? "Real-CUGAN native 2×"
                : rifeCheckpointValid ? "RIFE HQ"
                : restorationName
            DiagnosticsLogger.shared.log("Recovery plan: furthest valid checkpoint is \(resumePoint). Stages before it will NOT be re-run.")
        } else if recoveryDirectory != nil {
            DiagnosticsLogger.shared.log("Recovery plan: no valid checkpoints found. Starting from the first stage.")
        }

        // IMPORTANT: Sharpie no longer runs here. Compression cleanup remains before
        // RIFE, while line art runs after Real-CUGAN, followed by optional Color Pop.
        var rifeSourceURL: URL?

        if rifeNeeded && !rifeCheckpointValid {
            if needsRestoration {
                if restoredCheckpointValid, let restoredCheckpoint {
                    rifeSourceURL = restoredCheckpoint
                    progress(0.10, "Recovered checkpoint • \(restorationName) complete")
                    RecoveryStore.update(progress: 0.10, message: "Recovered completed source restoration checkpoint", force: true)
                    DiagnosticsLogger.shared.log("Recovery: reused completed \(restorationName) checkpoint.")
                } else {
                    if let restoredCheckpoint { try? fm.removeItem(at: restoredCheckpoint) }
                    progress(0.001, "Pass 1/4 • Starting \(restorationName)…")
                    let restoration = RestorationPass(
                        compressionEnabled: config.compressionProtection,
                        outlineEnabled: false,
                        watermark: watermark
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
                        },
                        recoveryDirectory: recoveryDirectory
                    )

                    if let restoredCheckpoint {
                        try persistCheckpoint(from: result.url, to: restoredCheckpoint)
                        try? fm.removeItem(at: result.url)
                        rifeSourceURL = restoredCheckpoint
                        RecoveryStore.update(progress: 0.10, message: "Pass 1/4 checkpoint saved • \(restorationName) complete", force: true)
                        DiagnosticsLogger.shared.log("Checkpoint saved: \(restorationName).")
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
                let rife = RIFEVideoProcessor(configuration: pass2Config, colorPopStrength: (!upscaleTo4K && !config.outlineProtection) ? colorPopStrength : 0)
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
                    },
                    recoveryDirectory: recoveryDirectory
                )

                if let rifeCheckpoint {
                    try persistCheckpoint(from: generated, to: rifeCheckpoint)
                    try? fm.removeItem(at: generated)
                    rifeResult = rifeCheckpoint
                    RecoveryStore.update(progress: 0.40, message: "Pass 2/4 checkpoint saved • RIFE HQ complete", force: true)
                    DiagnosticsLogger.shared.log("Checkpoint saved: RIFE HQ.")
                    IncrementalVideoCheckpointWriter.cleanup(recoveryDirectory: recoveryDirectory, stageID: "rife")
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
                    let cugan = RealCUGANPass(intensity: 1.30, colorPopStrength: config.outlineProtection ? 0 : colorPopStrength)
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
                        },
                        recoveryDirectory: recoveryDirectory
                    )

                    if let cuganCheckpoint {
                        try persistCheckpoint(from: generated, to: cuganCheckpoint)
                        try? fm.removeItem(at: generated)
                        postCUGANSource = cuganCheckpoint
                        RecoveryStore.update(progress: cuganEnd, message: "Pass 3/4 checkpoint saved • Real-CUGAN native 2× complete", force: true)
                        DiagnosticsLogger.shared.log("Checkpoint saved: Real-CUGAN native 2×.")
                        IncrementalVideoCheckpointWriter.cleanup(recoveryDirectory: recoveryDirectory, stageID: "cugan")
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
                let finalOutline = FinalOutlinePass(colorPopStrength: colorPopStrength)
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
                    },
                    recoveryDirectory: recoveryDirectory
                )

                if let outlineCheckpoint {
                    try persistCheckpoint(from: generated, to: outlineCheckpoint)
                    try? fm.removeItem(at: generated)
                    videoForMux = outlineCheckpoint
                    RecoveryStore.update(progress: 0.97, message: "Final Sharpie checkpoint saved • post-CUGAN outline complete", force: true)
                    DiagnosticsLogger.shared.log("Checkpoint saved: final post-CUGAN Sharpie outline.")
                    IncrementalVideoCheckpointWriter.cleanup(recoveryDirectory: recoveryDirectory, stageID: "final-outline")
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

    /// Separate checkpoint names prevent the two stage orders from sharing outputs.
    /// Reverse validation always resumes AFTER the furthest completed stage.
    private func processUpscaleFirst(sourceURL: URL, recoveryDirectory: URL?, progress: @escaping @Sendable (Double, String) -> Void, telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void) async throws -> URL {
        let fm = FileManager.default
        if let recoveryDirectory { try fm.createDirectory(at: recoveryDirectory, withIntermediateDirectories: true) }
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ProcessorError.missingVideoTrack }
        let duration = try await asset.load(.duration)
        let size = try await track.load(.naturalSize)
        let width = Int(abs(size.width).rounded()), height = Int(abs(size.height).rounded())
        let outputWidth = width * 2, outputHeight = height * 2
        let audioBitrate: Double
        if let audio = try await asset.loadTracks(withMediaType: .audio).first { audioBitrate = Double(try await audio.load(.estimatedDataRate)) } else { audioBitrate = 0 }
        let restore = config.compressionProtection || watermark.enabled
        let stages = UpscaleFirstPlan.stages(restoration: restore, outline: config.outlineProtection)
        let checkpoints = stages.map { recoveryDirectory?.appendingPathComponent("checkpoint-upscale-first-\($0).mov") }
        var inputURL = sourceURL
        var startIndex = 0
        let completedIndex = await UpscaleFirstPlan.furthestValid(stages: stages) { index, stage in
            guard let checkpoint = checkpoints[index] else { return false }
            let isNative = stage == "restored"
            return await self.validVideo(checkpoint, expectedDuration: duration, expectedWidth: isNative ? width : outputWidth, expectedHeight: isNative ? height : outputHeight)
        }
        if let index = completedIndex, let checkpoint = checkpoints[index] {
            inputURL = checkpoint; startIndex = index + 1
            DiagnosticsLogger.shared.log("Upscale-first recovery: furthest valid checkpoint is \(stages[index]); skipping all earlier stages")
            progress(0.97 * Double(startIndex) / Double(stages.count), "Recovered upscale-first \(stages[index]) checkpoint")
        }
        var transient: [URL] = []
        defer { for url in transient { try? fm.removeItem(at: url) } }
        var combined = PerformanceTelemetry()
        if startIndex < stages.count {
            for index in startIndex..<stages.count {
                try Task.checkCancellation()
                let stage = stages[index]
                let lower = 0.97 * Double(index) / Double(stages.count)
                let span = 0.97 / Double(stages.count)
                let publish: @Sendable (Double, String) -> Void = { p, message in
                    progress(lower + min(max(p, 0), 1) * span, "Upscale-first • \(stage) • \(message)")
                }
                let generated: URL
                switch stage {
                case "restored":
                    let result = try await RestorationPass(compressionEnabled: config.compressionProtection, outlineEnabled: false, watermark: watermark).run(sourceURL: inputURL, progress: { p, message in publish(p / 0.28, message) }, telemetry: { sample in combined = sample; telemetry(sample) })
                    generated = result.url
                case "cugan":
                    try requireUpscaleFirstMemory(width: outputWidth, height: outputHeight)
                    generated = try await RealCUGANPass(intensity: 1.30).run(sourceURL: inputURL, finalAudioBitrate: audioBitrate, progress: publish, telemetry: { sample in
                        combined.upscaledFrames = sample.upscaledFrames; combined.cuganMsPerFrame = sample.cuganMsPerFrame; combined.upscaleFPS = sample.upscaleFPS
                        telemetry(combined)
                    })
                case "rife":
                    // Recheck after the CUGAN model has been released. Never silently
                    // change order on recovery: reject and retain completed checkpoints.
                    try requireUpscaleFirstMemory(width: outputWidth, height: outputHeight)
                    let settings = ProcessorConfiguration(quality: config.quality, ghostProtection: config.ghostProtection, sceneCutProtection: config.sceneCutProtection, compressionProtection: false, outlineProtection: false, ghostSensitivity: config.ghostSensitivity, preserveAudio: false, targetFPS: config.targetFPS)
                    generated = try await RIFEVideoProcessor(configuration: settings, colorPopStrength: config.outlineProtection ? 0 : colorPopStrength, guardUpscaleFirstMemory: true).process(sourceURL: inputURL, progress: { p, message in publish(p / 0.95, message) }, telemetry: { sample in
                        let saved = combined
                        combined = sample; combined.compressionMsPerFrame = saved.compressionMsPerFrame; combined.upscaledFrames = saved.upscaledFrames; combined.cuganMsPerFrame = saved.cuganMsPerFrame; combined.upscaleFPS = saved.upscaleFPS
                        telemetry(combined)
                    })
                default:
                    generated = try await FinalOutlinePass(colorPopStrength: colorPopStrength).run(sourceURL: inputURL, finalAudioBitrate: audioBitrate, progress: publish, telemetry: { sample in combined.outlineMsPerFrame = sample.outlineMsPerFrame; telemetry(combined) })
                }
                if let checkpoint = checkpoints[index] {
                    try persistCheckpoint(from: generated, to: checkpoint)
                    inputURL = checkpoint
                    RecoveryStore.update(progress: lower + span, message: "Upscale-first \(stage) checkpoint complete", force: true)
                    for previous in 0..<index { if let old = checkpoints[previous] { try? fm.removeItem(at: old) } }
                } else { inputURL = generated; transient.append(generated) }
                try await thermalHandoff(progress: progress, position: lower + span, next: index + 1 < stages.count ? stages[index + 1] : "final export")
            }
        }
        // Mux/delivery must not consume the checkpoint: retries can skip all stages.
        let finalCopy = fm.temporaryDirectory.appendingPathComponent("RIFE60-upscale-first-final-\(UUID().uuidString).mov")
        try fm.copyItem(at: inputURL, to: finalCopy)
        transient.append(finalCopy)
        if config.preserveAudio {
            let final = try await FinalAudioMuxer().addOriginalAudio(videoURL: finalCopy, sourceURL: sourceURL)
            progress(1, "Finished • Upscale-first native 2× 60fps")
            return final
        }
        transient.removeAll { $0 == finalCopy }
        progress(1, "Finished • Upscale-first native 2× 60fps")
        return finalCopy
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
