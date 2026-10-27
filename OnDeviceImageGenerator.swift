import Foundation
import UniformTypeIdentifiers
import Darwin
import _MediaGenerationKit
import DrawThingsCLILib
import DataModels
import Atomics

private final class AIImageGenerationMemoryWatchdog: @unchecked Sendable {
    private let lock = NSLock()
    private var triggered = false

    func trigger() {
        lock.lock()
        triggered = true
        lock.unlock()
    }

    var didTrigger: Bool {
        lock.lock()
        defer { lock.unlock() }
        return triggered
    }
}

/// Small append-only journal so the last successful stage is preserved even if iOS
/// terminates the process before the app can export its regular diagnostic log.
private final class AIImageGenerationMemoryJournal: @unchecked Sendable {
    static let shared = AIImageGenerationMemoryJournal()
    private let lock = NSLock()

    private init() {}

    func record(_ stage: String, step: Int? = nil) {
        let memory = currentRenderPerformanceSnapshot()
        let safeStage = stage.replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "|", with: "/")
        let stepText = step.map { String($0) } ?? "-"
        let line = "\(Date().timeIntervalSince1970)|\(safeStage)|step=\(stepText)|availableMB=\(Int(memory.availableMemoryMB))|physicalMB=\(Int(memory.physicalMemoryMB))|tier=\(memory.tier.rawValue)|thermal=\(memory.thermalAndMode)\n"
        guard let data = line.data(using: .utf8) else { return }

        lock.lock()
        defer { lock.unlock() }
        do {
            let base = try FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            let directory = base.appendingPathComponent("RIFE60GhostGuard/Diagnostics", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let url = directory.appendingPathComponent("AIImageGenerationMemory.log")
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            // Keep the journal bounded: repeated generations must not create an
            // unbounded append-only file. Avoid fsync for every sampler callback;
            // closing this short-lived handle persists the small record without
            // forcing a synchronous storage flush on every denoising step.
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
            let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
            if size > 256 * 1024 {
                try handle.close()
                let existing = (try? Data(contentsOf: url)) ?? Data()
                let tail = existing.suffix(128 * 1024)
                try Data(tail).write(to: url, options: .atomic)
            }
        } catch {
            DiagnosticsLogger.shared.log("AI Image Studio memory journal write failed: \(error.localizedDescription)")
        }
    }
}

enum OnDeviceImageGenerator {
    static let defaultModelID = "animagine_xl_v3.1_q6p_q8p.ckpt"
    static let animaModelID = "anima_base_1.0_q8p.ckpt"

    static func memoryJournalURL() -> URL? {
        guard let base = try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ) else { return nil }
        return base
            .appendingPathComponent("RIFE60GhostGuard/Diagnostics", isDirectory: true)
            .appendingPathComponent("AIImageGenerationMemory.log")
    }

    static func recordGenerationFailure() {
        AIImageGenerationMemoryJournal.shared.record("ui-generation-failure")
    }

    static func previousGenerationWarning() -> String? {
        guard let url = memoryJournalURL(),
              let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        do {
            let fileSize = try handle.seekToEnd()
            guard fileSize > 0 else { return nil }
            let byteCount = Int(min(fileSize, 16 * 1024))
            try handle.seek(toOffset: fileSize - UInt64(byteCount))
            guard let data = try handle.read(upToCount: byteCount),
                  let text = String(data: data, encoding: .utf8),
                  let lastLine = text.split(whereSeparator: \.isNewline).last else {
                return nil
            }
            let fields = lastLine.split(separator: "|", maxSplits: 2, omittingEmptySubsequences: false)
            guard fields.count > 1 else { return nil }
            let stage = String(fields[1])
            // "generation-complete" is retained as a legacy terminal marker from
            // earlier builds. New builds distinguish sampler completion from a
            // successfully written output file.
            let terminalStages: Set<String> = [
                "generation-complete", "image-output-saved",
                "generation-cancelled", "ui-generation-failure"
            ]
            guard !terminalStages.contains(stage), !stage.hasPrefix("benchmark-complete-") else { return nil }
            return "The previous AI generation may have stopped before its output was safely saved. The next attempt will preserve the requested aspect ratio and use memory-based sizing, with Hires/LoRA disabled for safety. Export memory diagnostics below to inspect the last recorded stage."
        } catch {
            return nil
        }
    }

    // No fixed 2.25 GiB stop threshold: the engine uses on-demand disk-backed weights,
    // low-memory device capability, partial CPU offload, and the smallest supported tiles.
    // iOS can still terminate the process if transient Metal allocations exceed its budget.

    /// Saves the process-global Draw Things knobs this generator mutates and puts them back on exit.
    fileprivate struct EngineStateSnapshot {
        let restore: () -> Void

        static func capture() -> EngineStateSnapshot {
            let capacity = DeviceCapability.memoryCapacity
            let partialOffload = DeviceCapability.isPartialOffloadPreferred.load(ordering: .acquiring)
            let environment = MediaGenerationEnvironment.default
            let weightsCache = environment.maxTotalWeightsCacheSize
            let externalURLs = environment.externalUrls
            return EngineStateSnapshot {
                DeviceCapability.memoryCapacity = capacity
                DeviceCapability.isPartialOffloadPreferred.store(partialOffload, ordering: .releasing)
                var restored = MediaGenerationEnvironment.default
                restored.externalUrls = externalURLs
                restored.maxTotalWeightsCacheSize = weightsCache
                MediaGenerationEnvironment.default = restored
            }
        }
    }

    /// Converts a LoRA into Draw Things format and refuses it when the detected architecture does not match the selected model.
    static func importLoRA(from sourceURL: URL, modelID: String) throws -> StoredLocalLoRA {
        let ext = sourceURL.pathExtension.lowercased()
        guard ["safetensors", "ckpt", "pt", "bin"].contains(ext) else {
            throw AIImageStudioError("Choose a LoRA checkpoint file (.safetensors, .ckpt, .pt, or .bin).")
        }
        let profile = ImageModelProfile.profile(for: modelID)
        let directory = try modelsDirectoryURL()
        let stem = sourceURL.deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: " ", with: "_")
        let outputName = stem.hasSuffix("_lora_f16") ? "\(stem).ckpt" : "\(stem)_lora_f16.ckpt"
        let imported = try LoRAImporter.import(
            downloadedFile: sourceURL.path,
            name: stem,
            filename: outputName,
            scaleFactor: 1,
            forceVersion: nil,
            progress: { _ in }
        )
        let detected = String(describing: imported.0).lowercased()
        let destination = directory.appendingPathComponent(outputName)
        let expectedToken = profile.family == .sdxl ? "sdxl" : "cosmos"
        guard detected.contains(expectedToken) else {
            try? FileManager.default.removeItem(at: destination)
            throw AIImageStudioError("This LoRA was detected as \(detected), which does not match \(profile.displayName). It was not selected.")
        }
        let trigger = loraTrigger(in: sourceURL) ?? ""
        let sidecar = StoredLoRASidecar(version: detected, trigger: trigger, sourceFile: sourceURL.lastPathComponent)
        let sidecarURL = directory.appendingPathComponent(outputName + ".json")
        if let data = try? JSONEncoder().encode(sidecar) {
            try? data.write(to: sidecarURL, options: .atomic)
        }
        return StoredLocalLoRA(file: outputName, weight: 0.7, version: detected, trigger: trigger)
    }

    private struct StoredLoRASidecar: Codable {
        var version: String
        var trigger: String
        var sourceFile: String
    }

    private static func loraTrigger(in url: URL) -> String? {
        guard url.pathExtension.lowercased() == "safetensors",
              let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8), prefix.count == 8 else { return nil }
        let headerLength = prefix.enumerated().reduce(UInt64(0)) { partial, item in
            partial | (UInt64(item.element) << (item.offset * 8))
        }
        guard headerLength > 0, headerLength <= 16 * 1024 * 1024,
              let header = try? handle.read(upToCount: Int(headerLength)),
              header.count == Int(headerLength),
              let root = try? JSONSerialization.jsonObject(with: header) as? [String: Any],
              let metadata = root["__metadata__"] as? [String: Any] else { return nil }
        let keys = ["modelspec.trigger_phrase", "ss_trigger_words", "ss_trigger_word", "trigger_word", "trigger"]
        for key in keys {
            if let raw = metadata[key] as? String {
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        return nil
    }

    static func modelsDirectoryURL() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = base.appendingPathComponent("RIFE60GhostGuard/Models", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func notify(
        _ message: String,
        progress: @escaping @MainActor @Sendable (String) -> Void
    ) {
        Task { @MainActor in progress(message) }
    }

    private static func multipleOf64(_ value: Int) -> Int {
        max(64, Int((Double(max(64, value)) / 64.0).rounded()) * 64)
    }

    /// First-pass size for hires fix. Draw Things treats hiresFixWidth as the smaller start size.
    private static func smallerHiresPass(_ value: Int, scale: Double) -> Int {
        let divisor = max(scale, 1.05)
        let raw = Int((Double(value) / divisor).rounded(.down))
        let snapped = max(64, (raw / 64) * 64)
        let strictlySmaller = max(64, ((value - 64) / 64) * 64)
        return min(snapped, strictlySmaller)
    }

    static func generate(
        settings: AIImageStudioSettings,
        positivePrompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Double,
        seed: Int64,
        samplerName: String,
        shift: Double,
        enableLoRA: Bool,
        loraWeight: Double,
        hiresEnabled: Bool,
        hiresScale: Double,
        hiresSteps: Int,
        hiresDenoise: Double,
        clipSkip: Int,
        benchmarkMode: Bool = false,
        progress: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> URL {
        let benchmarkStartedAt = Date()
        if benchmarkMode {
            AIImageGenerationMemoryJournal.shared.record("benchmark-start")
            DiagnosticsLogger.shared.log("AI Image Studio benchmark started • stages=model resolution, model load, text encoding, input encoding, denoising, decoding, PNG write • smoke profile=128x128/1 step to isolate the first sampler-to-decoder transition")
        }
        guard !positivePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AIImageStudioError("The positive prompt cannot be empty.")
        }
        let modelID = settings.localModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelID.isEmpty else {
            throw AIImageStudioError("Choose a supported local model catalog ID.")
        }

        // If a prior attempt ended without a terminal journal marker, treat this
        // run as a recovery retry. Recovery disables memory-heavy Hires/LoRA paths,
        // but output size is still chosen by the same live-memory policy as normal runs.
        let recoveringAfterInterruption = previousGenerationWarning() != nil && !benchmarkMode
        // Preserve the requested diffusion step count in normal and recovery runs.
        // Step count is a quality/runtime control, not the primary peak-memory control;
        // memory pressure is handled through canvas size, decode strategy, on-demand
        // weights, partial offload, and disabling memory-heavy Hires/LoRA paths.
        var effectiveSteps = max(1, min(60, steps))

        // Diffusion inference competes with the rest of iOS for unified memory.
        // Pick a conservative working size before loading SDXL, then enable the
        // engine's native tiled paths when headroom is limited.
        var memory = currentRenderPerformanceSnapshot()
        if memory.availableMemoryMB < 1_024 {
            DiagnosticsLogger.shared.log("AI Image Studio entering low-memory mode before model load • available=\(Int(memory.availableMemoryMB)) MB")
        }

        let profile = ImageModelProfile.profile(for: modelID)
        let requestedWidth = multipleOf64(width)
        let requestedHeight = multipleOf64(height)
        // Native SDXL canvases such as 832x1216 stay intact. The old <4 GB cap
        // forced 384–768 px on a 6 GB phone and broke anatomy, especially with LoRAs.
        var outputWidth = requestedWidth
        var outputHeight = requestedHeight
        let useTiledDiffusion = false
        // Tiled VAE decode uses 512 px tiles (8 units) and 128 px overlap (2 units).
        // The crash was the 64 px / 1-unit tile, not tiling itself.
        var useTiledDecoding = true
        let decodingTilePixels = 512
        let decodingOverlapPixels = 128

        DiagnosticsLogger.shared.log(
            "AI Image Studio memory preflight • available=\(Int(memory.availableMemoryMB)) MB • physical=\(Int(memory.physicalMemoryMB)) MB • tier=\(memory.tier.rawValue) • requested=\(requestedWidth)x\(requestedHeight) • selected=\(outputWidth)x\(outputHeight) • tiledDiffusion=\(useTiledDiffusion) • tiledDecoding=\(useTiledDecoding)"
        )

        AIImageGenerationMemoryJournal.shared.record("pre-model-policy")
        let modelsDirectory = try modelsDirectoryURL()
        // Draw Things exposes a supported partial-offload policy in DataModels.
        // Force its conservative capacity tier and CPU partial-offload path before
        // constructing the pipeline. This trades speed for a smaller resident GPU
        // working set on memory-constrained iPhones.
        // Use Draw Things' lowest-memory execution tier for this SDXL workload.
        // The medium tier can still retain more model state than this device can
        // safely accommodate after pipeline construction. Restore the app's
        // established medium tier when this generation call exits.
        let previousEngineState = EngineStateSnapshot.capture()
        defer { previousEngineState.restore() }
        DeviceCapability.memoryCapacity = .low
        DeviceCapability.isPartialOffloadPreferred.store(true, ordering: .releasing)
        DiagnosticsLogger.shared.log("AI Image Studio memory policy • partial CPU offload enabled • capacity=low • weights cache disabled")

        var environment = MediaGenerationEnvironment.default
        environment.externalUrls = [modelsDirectory]
        // This device is below Draw Things' 47 GiB weights-cache threshold.
        // Keep the cache explicitly disabled on constrained devices.
        environment.maxTotalWeightsCacheSize = 0
        MediaGenerationEnvironment.default = environment

        notify(
            recoveringAfterInterruption
                ? "Recovery Safe Mode: checking local model files…"
                : "Checking local model files…",
            progress: progress
        )
        AIImageGenerationMemoryJournal.shared.record("model-resolution-start")
        if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-model-resolution-start") }
        let resolvedModel = try await environment.ensure(modelID, offline: false) { state in
            let message: String
            switch state {
            case .resolving:
                message = "Resolving model catalog entry…"
            case .verifying(let file, let index, let total):
                message = "Checking \(index)/\(total): \(file)"
            case .downloading(let file, let index, let total, let bytes, let expected):
                if expected > 0 {
                    let percent = Int(Double(bytes) / Double(expected) * 100.0)
                    message = "Downloading \(index)/\(total): \(file) (\(min(100, percent))%)"
                } else {
                    message = "Downloading \(index)/\(total): \(file)"
                }
            }
            if benchmarkMode {
                switch state {
                case .resolving:
                    AIImageGenerationMemoryJournal.shared.record("benchmark-model-catalog-resolving")
                case .verifying:
                    AIImageGenerationMemoryJournal.shared.record("benchmark-model-files-verifying")
                case .downloading:
                    AIImageGenerationMemoryJournal.shared.record("benchmark-model-files-downloading")
                }
            }
            Task { @MainActor in progress(message) }
        }
        AIImageGenerationMemoryJournal.shared.record("model-resolution-complete")
        if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-model-resolution-complete") }

        try Task.checkCancellation()

        // Downloading/verifying model files can take time and memory headroom can
        // change while it runs. Re-evaluate before constructing the heavy SDXL
        // pipeline rather than trusting the initial snapshot.
        memory = currentRenderPerformanceSnapshot()
        if memory.availableMemoryMB < 1_024 {
            DiagnosticsLogger.shared.log("AI Image Studio model-load headroom is low • continuing with on-demand weights and tiled inference • available=\(Int(memory.availableMemoryMB)) MB")
        }
        outputWidth = requestedWidth
        outputHeight = requestedHeight
        DiagnosticsLogger.shared.log(
            "AI Image Studio refreshed pre-load memory • available=\(Int(memory.availableMemoryMB)) MB • selected=\(outputWidth)x\(outputHeight) • decodeTilePixels=\(decodingTilePixels) • overlapPixels=\(decodingOverlapPixels)"
        )

        notify(
            recoveringAfterInterruption
                ? "Recovery Safe Mode: loading local Metal model…"
                : "Loading local Metal model…",
            progress: progress
        )
        AIImageGenerationMemoryJournal.shared.record("model-load-start")
        if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-model-load-start") }
        var pipeline: MediaGenerationPipeline? = try await MediaGenerationPipeline.fromPretrained(
            resolvedModel.file,
            backend: .local
        )
        memory = currentRenderPerformanceSnapshot()
        AIImageGenerationMemoryJournal.shared.record("model-load-complete")
        if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-model-load-complete") }
        DiagnosticsLogger.shared.log(
            "AI Image Studio model loaded • available=\(Int(memory.availableMemoryMB)) MB • physical=\(Int(memory.physicalMemoryMB)) MB • thermal=\(memory.thermalAndMode)"
        )
        if memory.availableMemoryMB < 1_024 {
            DiagnosticsLogger.shared.log("AI Image Studio continuing below former inference reserve • available=\(Int(memory.availableMemoryMB)) MB • using minimum tiles and reduced dimensions")
        }
        // If model loading leaves less than 3 GiB, reduce latent dimensions while
        // preserving the requested aspect ratio. Upscaling can happen afterward via
        // the existing Real-CUGAN stage, which runs after this pipeline is released.
        if recoveringAfterInterruption {
            DiagnosticsLogger.shared.log("AI Image Studio Recovery Safe Mode • size \(outputWidth)x\(outputHeight) • steps=\(effectiveSteps) • Hires/LoRA disabled for this retry")
        }
        if benchmarkMode {
            outputWidth = 128
            outputHeight = 128
            effectiveSteps = 1
            useTiledDecoding = false
            AIImageGenerationMemoryJournal.shared.record("benchmark-configured-128x128-1step")
        }
        guard let pipeline else {
            throw AIImageStudioError("The local image engine did not create a pipeline.")
        }
        let appliedShift = profile.appliedShift(shift)
        let appliedSampler = profile.family == .flowMatching && !ImageModelProfile.flowCompatibleSamplers.contains(samplerName)
            ? profile.samplerName
            : samplerName
        let engineSeed = UInt32(truncatingIfNeeded: max(0, seed))
        DiagnosticsLogger.shared.log(
            "AI Image Studio execution policy • model=\(profile.displayName) • recovery=\(recoveringAfterInterruption) • size=\(outputWidth)x\(outputHeight) • steps=\(effectiveSteps) • cfg=\(cfg) • shift=\(appliedShift) • sampler=\(appliedSampler) • clipSkip=\(clipSkip) • tiledDecode=\(useTiledDecoding) • requestedSeed=\(seed) • engineSeed=\(engineSeed)"
        )
        if seed > Int64(UInt32.max) {
            notify("Engine seed is \(engineSeed) (the low 32 bits of \(seed)). The Draw Things seed field is 32-bit, so this is the value that will reproduce the image.", progress: progress)
        }
        pipeline.configuration.width = outputWidth
        pipeline.configuration.height = outputHeight
        pipeline.configuration.steps = effectiveSteps
        pipeline.configuration.batchCount = 1
        pipeline.configuration.batchSize = 1
        pipeline.configuration.tiledDecoding = useTiledDecoding
        if useTiledDecoding {
            pipeline.configuration.decodingTileWidth = decodingTilePixels
            pipeline.configuration.decodingTileHeight = decodingTilePixels
            pipeline.configuration.decodingTileOverlap = decodingOverlapPixels
        }
        pipeline.configuration.tiledDiffusion = useTiledDiffusion
        pipeline.configuration.guidanceScale = Float(max(0, min(20, cfg)))
        pipeline.configuration.seed = engineSeed
        pipeline.configuration.shift = Float(appliedShift)
        pipeline.configuration.stage2Shift = Float(profile.usesShift ? appliedShift : 1)
        pipeline.configuration.resolutionDependentShift = false
        switch appliedSampler {
        case "Euler a":
            pipeline.configuration.sampler = DataModels.SamplerType.eulerA
        case "Euler a Trailing":
            pipeline.configuration.sampler = DataModels.SamplerType.eulerATrailing
        case "DPM++ SDE Karras", "DPM++ SDE Karras (closest to Civitai GPU sampler)":
            pipeline.configuration.sampler = DataModels.SamplerType.dPMPPSDEKarras
        case "DPM++ 2M Karras":
            pipeline.configuration.sampler = DataModels.SamplerType.dPMPP2MKarras
        case "DPM++ SDE Trailing":
            pipeline.configuration.sampler = DataModels.SamplerType.dPMPPSDETrailing
        case "DPM++ 2M Trailing":
            pipeline.configuration.sampler = DataModels.SamplerType.dPMPP2MTrailing
        case "DDIM":
            pipeline.configuration.sampler = DataModels.SamplerType.DDIM
        case "UniPC":
            pipeline.configuration.sampler = DataModels.SamplerType.uniPC
        default:
            pipeline.configuration.sampler = profile.family == .flowMatching
                ? DataModels.SamplerType.eulerATrailing
                : DataModels.SamplerType.eulerA
        }
        pipeline.configuration.clipSkip = max(1, min(2, clipSkip))

        if hiresEnabled && !benchmarkMode && !recoveringAfterInterruption && memory.availableMemoryMB >= 1_800 {
            let firstWidth = smallerHiresPass(outputWidth, scale: hiresScale)
            let firstHeight = smallerHiresPass(outputHeight, scale: hiresScale)
            pipeline.configuration.hiresFix = true
            pipeline.configuration.hiresFixWidth = firstWidth
            pipeline.configuration.hiresFixHeight = firstHeight
            pipeline.configuration.hiresFixStrength = Float(max(0.05, min(0.95, hiresDenoise)))
            pipeline.configuration.stage2Steps = max(1, min(60, hiresSteps))
            pipeline.configuration.stage2Guidance = Float(max(0, min(20, cfg)))
            DiagnosticsLogger.shared.log("AI Image Studio hires fix • first pass \(firstWidth)x\(firstHeight) • final \(outputWidth)x\(outputHeight)")
        } else {
            pipeline.configuration.hiresFix = false
            if hiresEnabled && recoveringAfterInterruption {
                notify("Recovery Safe Mode: high-resolution diffusion is disabled for this retry.", progress: progress)
            } else if hiresEnabled && memory.availableMemoryMB < 1_800 && !benchmarkMode {
                notify("High-resolution diffusion needs at least 1.8 GB free after the model loads. This run uses the requested canvas only.", progress: progress)
            }
        }

        let selectedLoRAs: [(file: String, weight: Float)] = {
            if !settings.localLoras.isEmpty {
                return settings.localLoras.map { ($0.file, Float(loraWeight)) }
            }
            let single = settings.localLoraFile.trimmingCharacters(in: .whitespacesAndNewlines)
            if !single.isEmpty { return [(single, Float(loraWeight))] }
            return []
        }()
        let canApplyLoRA = !benchmarkMode
            && !recoveringAfterInterruption
            && enableLoRA
            && !selectedLoRAs.isEmpty
            && memory.availableMemoryMB >= 1_200
        if canApplyLoRA {
            pipeline.configuration.loras = selectedLoRAs.map {
                DataModels.LoRA(file: $0.file, weight: $0.weight, mode: .all)
            }
            DiagnosticsLogger.shared.log("AI Image Studio LoRA enabled • count=\(selectedLoRAs.count) • available=\(Int(memory.availableMemoryMB)) MB • output=\(outputWidth)x\(outputHeight)")
        } else {
            pipeline.configuration.loras = []
            if enableLoRA && recoveringAfterInterruption {
                notify("Recovery Safe Mode: local LoRA is disabled for this retry; the next normal run can try it again.", progress: progress)
            } else if enableLoRA && !selectedLoRAs.isEmpty && memory.availableMemoryMB < 1_200 && !benchmarkMode {
                notify("LoRA needs at least 1.2 GB free after model load. Generation continues without it.", progress: progress)
            }
        }

        let generationMode = recoveringAfterInterruption
            ? "Recovery Safe Mode retry"
            : "Generating locally on the iPhone GPU/Metal…"
        notify(
            "\(generationMode) • \(outputWidth)x\(outputHeight) • \(effectiveSteps) steps • \(Int(memory.availableMemoryMB)) MB initial headroom",
            progress: progress
        )
        DiagnosticsLogger.shared.log("AI Image Studio generation begin • steps=\(effectiveSteps) • size=\(outputWidth)x\(outputHeight) • tiledDiffusion=\(useTiledDiffusion) • tiledDecoding=\(useTiledDecoding) • decodeTilePixels=\(decodingTilePixels) • overlapPixels=\(decodingOverlapPixels)")
        // Poll memory during inference because SDXL can allocate transient Metal buffers during early denoising.
        // Keep the pipeline's strong reference inside the task only. Once the
        // task completes, its operation can release that capture before PNG encoding.
        let preDenoisingMemory = currentRenderPerformanceSnapshot()
        DiagnosticsLogger.shared.log(
            "AI Image Studio immediately before generate() • available=\(Int(preDenoisingMemory.availableMemoryMB)) MB • physical=\(Int(preDenoisingMemory.physicalMemoryMB)) MB • size=\(outputWidth)x\(outputHeight) • steps=\(effectiveSteps) • tiledDiffusion=\(useTiledDiffusion) • tiledDecode=\(useTiledDecoding) • partialOffload=true"
        )
        AIImageGenerationMemoryJournal.shared.record("pre-denoising-allocation")
        AIImageGenerationMemoryJournal.shared.record("pipeline-configured")
        if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-pre-denoising") }
        AIImageGenerationMemoryJournal.shared.record("generate-call-start")
        var releasablePipeline: MediaGenerationPipeline? = pipeline
        let generationTask = Task { [activePipeline = pipeline] in
            try await activePipeline.generate(prompt: positivePrompt, negativePrompt: negativePrompt) { state, _ in
            let message: String
            switch state {
            case .resolvingBackend(_):
                message = "Preparing on-device backend…"
                AIImageGenerationMemoryJournal.shared.record("engine-resolving-backend")
            case .resolvingModel(let name):
                message = "Resolving \(name)…"
                AIImageGenerationMemoryJournal.shared.record("engine-resolving-model")
            case .preparing:
                message = "Preparing local generation…"
                AIImageGenerationMemoryJournal.shared.record("engine-preparing")
                if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-engine-preparing") }
            case .ensuringResources:
                message = "Checking local model resources…"
                AIImageGenerationMemoryJournal.shared.record("engine-ensuring-resources")
            case .uploading(_, _), .downloading(_, _):
                message = "Processing local model resources…"
                AIImageGenerationMemoryJournal.shared.record("engine-resource-transfer")
            case .encodingText:
                message = "Encoding prompt on device…"
                AIImageGenerationMemoryJournal.shared.record("engine-encoding-text")
                if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-text-encoding") }
            case .encodingInputs:
                message = "Preparing generation inputs…"
                AIImageGenerationMemoryJournal.shared.record("engine-encoding-inputs")
                if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-input-encoding") }
            case .generating(let step, let total):
                let currentMemory = currentRenderPerformanceSnapshot()
                message = "Local GPU generation: step \(step)/\(total) • \(Int(currentMemory.availableMemoryMB)) MB free"
                // Capture every step so a crash after the first few denoising
                // iterations can be correlated with memory pressure in device logs.
                DiagnosticsLogger.shared.log(
                    "AI Image Studio step \(step)/\(total) • available=\(Int(currentMemory.availableMemoryMB)) MB • physical=\(Int(currentMemory.physicalMemoryMB)) MB • tier=\(currentMemory.tier.rawValue) • thermal=\(currentMemory.thermalAndMode)"
                )
                if step == 1 || step == total || step.isMultiple(of: 2) {
                    AIImageGenerationMemoryJournal.shared.record("denoising", step: step)
                }
                if benchmarkMode {
                    AIImageGenerationMemoryJournal.shared.record("benchmark-denoising", step: step)
                }
            case .decoding:
                message = "Decoding image…"
                let decodeMemory = currentRenderPerformanceSnapshot()
                DiagnosticsLogger.shared.log("AI Image Studio decoder entry • available=\(Int(decodeMemory.availableMemoryMB)) MB • physical=\(Int(decodeMemory.physicalMemoryMB)) MB")
                AIImageGenerationMemoryJournal.shared.record("decoder-start")
                if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-decoder-start") }
            case .postprocessing:
                message = "Running local high-resolution pass…"
            case .cancelling:
                message = "Cancelling generation…"
            case .completed:
                message = "Generation complete."
                AIImageGenerationMemoryJournal.shared.record("sampler-complete")
                if benchmarkMode {
                    let completedMemory = currentRenderPerformanceSnapshot()
                    AIImageGenerationMemoryJournal.shared.record("benchmark-engine-state-completed")
                    DiagnosticsLogger.shared.log("AI Image Studio benchmark engine state completed • available=\(Int(completedMemory.availableMemoryMB)) MB • physical=\(Int(completedMemory.physicalMemoryMB)) MB")
                    AIImageGenerationMemoryJournal.shared.record("benchmark-sampler-complete")
                }
            case .cancelled:
                message = "Generation cancelled."
                AIImageGenerationMemoryJournal.shared.record("generation-cancelled")
            }
            if case .generating = state {
                let headroom = currentRenderPerformanceSnapshot().availableMemoryMB
                if headroom < 768 {
                    DiagnosticsLogger.shared.log("AI Image Studio critical memory warning at step boundary • available=\(Int(headroom)) MB • continuing with engine-managed offload; iOS may still terminate on an allocation spike")
                }
            }
            Task { @MainActor in progress(message) }
            }
        }
        let lowMemoryWarningLogged = AIImageGenerationMemoryWatchdog()
        let memoryMonitorTask = Task {
            // Persist only threshold crossings, not every polling sample, to keep
            // the journal useful without continuously synchronizing the file.
            var loggedThresholds: Set<Int> = []
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 250_000_000)
                if Task.isCancelled { return }
                let snapshot = currentRenderPerformanceSnapshot()
                let available = Int(snapshot.availableMemoryMB)
                for threshold in [1_536, 1_024, 768] where available < threshold && !loggedThresholds.contains(threshold) {
                    loggedThresholds.insert(threshold)
                    DiagnosticsLogger.shared.log("AI Image Studio memory threshold crossed • threshold=\(threshold) MB • available=\(available) MB • tier=\(snapshot.tier.rawValue) • thermal=\(snapshot.thermalAndMode)")
                    AIImageGenerationMemoryJournal.shared.record("memory-threshold-\(threshold)MB")
                }
                if snapshot.availableMemoryMB < 768 && !lowMemoryWarningLogged.didTrigger {
                    lowMemoryWarningLogged.trigger()
                    DiagnosticsLogger.shared.log("AI Image Studio critical memory pressure • available=\(available) MB • inference may be terminated by iOS during a transient Metal allocation")
                    AIImageGenerationMemoryJournal.shared.record("critical-memory-pressure")
                }
            }
        }
        defer { memoryMonitorTask.cancel() }
        let generationResult = await generationTask.result
        if case .failure(let error) = generationResult {
            if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-engine-error") }
            let failureMemory = currentRenderPerformanceSnapshot()
            DiagnosticsLogger.shared.log("AI Image Studio engine generation failed • error=\(error.localizedDescription) • available=\(Int(failureMemory.availableMemoryMB)) MB • physical=\(Int(failureMemory.physicalMemoryMB)) MB")
            AIImageGenerationMemoryJournal.shared.record("engine-generation-error")
        }
        if benchmarkMode {
            let returnedMemory = currentRenderPerformanceSnapshot()
            AIImageGenerationMemoryJournal.shared.record("benchmark-engine-returned")
            DiagnosticsLogger.shared.log("AI Image Studio benchmark engine returned • available=\(Int(returnedMemory.availableMemoryMB)) MB • physical=\(Int(returnedMemory.physicalMemoryMB)) MB")
        }
        var results = try generationResult.get()
        if benchmarkMode {
            AIImageGenerationMemoryJournal.shared.record("benchmark-results-received")
        }
        guard let result = results.first else {
            releasablePipeline = nil
            throw AIImageStudioError("The local image engine returned no image.")
        }
        // Release the diffusion pipeline before encoding the decoded image. PNG
        // compression can allocate substantial CPU-side buffers; retaining SDXL's
        // model and Metal resources during that phase needlessly compounds peak RAM.
        releasablePipeline = nil
        let beforeEncoding = currentRenderPerformanceSnapshot()
        DiagnosticsLogger.shared.log(
            "AI Image Studio pipeline released before PNG encoding • available=\(Int(beforeEncoding.availableMemoryMB)) MB • physical=\(Int(beforeEncoding.physicalMemoryMB)) MB"
        )

        // Generated images are user output, not scratch files. Keep them in the app's
        // persistent Application Support area so iOS temporary-file cleanup cannot
        // remove a completed result. Never duplicate the PNG into a full Data buffer.
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let outputDirectory = support.appendingPathComponent("RIFE60GhostGuard/GeneratedImages", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let outputURL = outputDirectory.appendingPathComponent("RIFE60_AI_\(UUID().uuidString).png")
        do {
            if benchmarkMode {
                AIImageGenerationMemoryJournal.shared.record("benchmark-png-write-start")
            }
            try result.write(to: outputURL, type: .png)
            AIImageGenerationMemoryJournal.shared.record("image-output-saved")
            if benchmarkMode {
                AIImageGenerationMemoryJournal.shared.record("benchmark-png-write-complete")
                AIImageGenerationMemoryJournal.shared.record("benchmark-complete-\(Int(Date().timeIntervalSince(benchmarkStartedAt) * 1000))ms")
                DiagnosticsLogger.shared.log("AI Image Studio benchmark completed • elapsedMs=\(Int(Date().timeIntervalSince(benchmarkStartedAt) * 1000)) • output=\(outputWidth)x\(outputHeight) • steps=\(effectiveSteps)")
            }
        } catch {
            results.removeAll(keepingCapacity: false)
            throw error
        }
        results.removeAll(keepingCapacity: false)
        notify("PNG saved to Generated Images. Diffusion pipeline was released before encoding.", progress: progress)
        return outputURL
    }
}

enum OnDeviceLoRATrainer {
    private final class CompletionFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        func finish() { lock.lock(); value = true; lock.unlock() }
        var isFinished: Bool { lock.lock(); defer { lock.unlock() }; return value }
    }

    private static func notify(
        _ message: String,
        progress: @escaping @MainActor @Sendable (String, Double?) -> Void,
        fraction: Double? = nil
    ) {
        Task { @MainActor in progress(message, fraction) }
    }

    static func train(
        settings: AIImageStudioSettings,
        images: [LocalTrainingImage],
        triggerWord: String,
        rank: Int,
        epochs: Int,
        learningRate: Double,
        progress: @escaping @MainActor @Sendable (String, Double?) -> Void
    ) async throws -> URL {
        guard images.count >= 3 else {
            throw AIImageStudioError("Select at least three training images.")
        }
        guard !triggerWord.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AIImageStudioError("Enter a trigger word before training.")
        }

        // LoRA training invokes the native Draw Things model compiler. Native
        // Metal/Core ML compilation traps cannot be caught by Swift do/catch, so
        // refuse to enter that code path unless the phone has a meaningful memory
        // reserve. Build 187's crash trace showed only ~3.0 GiB available before
        // loading the model, which is not enough headroom for the trainer's compile
        // and optimizer allocations alongside the app's video/Metal subsystems.
        let preflight = currentRenderPerformanceSnapshot()
        DiagnosticsLogger.shared.log(
            "Local LoRA training preflight • available=\(Int(preflight.availableMemoryMB)) MB • physical=\(Int(preflight.physicalMemoryMB)) MB • tier=\(preflight.tier.rawValue) • thermal=\(preflight.thermalAndMode)"
        )
        guard preflight.availableMemoryMB >= 3_200 else {
            throw AIImageStudioError(
                "Training was stopped before loading the model because only \(Int(preflight.availableMemoryMB)) MB of memory is available. Close other GPU/video tasks, let iOS recover memory, then retry when at least 3.2 GB is available. This prevents the native model compiler from being entered under known unsafe memory pressure."
            )
        }

        // Match the inference path's constrained-device policy before invoking
        // the CLI trainer. Keep weights file-backed and avoid a resident model cache.
        let previousEngineState = EngineStateSnapshot.capture()
        defer { previousEngineState.restore() }
        DeviceCapability.memoryCapacity = .low
        DeviceCapability.isPartialOffloadPreferred.store(true, ordering: .releasing)
        var trainingEnvironment = MediaGenerationEnvironment.default
        trainingEnvironment.maxTotalWeightsCacheSize = 0
        MediaGenerationEnvironment.default = trainingEnvironment

        let modelsDirectory = try OnDeviceImageGenerator.modelsDirectoryURL()
        let datasetDirectory = modelsDirectory
            .appendingPathComponent("LocalTrainingData", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: datasetDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: datasetDirectory)
        }

        for (index, image) in images.enumerated() {
            try Task.checkCancellation()
            let imageName = "training_\(index + 1).png"
            try FileManager.default.copyItem(at: image.sourceURL, to: datasetDirectory.appendingPathComponent(imageName))
            let caption = ([triggerWord, image.caption]
                .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
                .joined(separator: ", ")
            try Data(caption.utf8).write(
                to: datasetDirectory.appendingPathComponent("training_\(index + 1).txt"),
                options: .atomic
            )
            notify("Preparing local dataset: \(index + 1)/\(images.count)", progress: progress,
                   fraction: Double(index + 1) / Double(images.count) * 0.1)
        }

        // A short, bounded first run is materially safer on a 6 GiB iPhone.
        // The UI's epoch/rank controls remain meaningful, but never let the CLI
        // expand this single in-process training session beyond 300 optimizer steps.
        let steps = max(100, min(300, images.count * max(1, epochs)))
        let prefix = "rife_local_lora_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(10))"
        let arguments = [
            "train", "lora",
            "--models-dir", modelsDirectory.path,
            "--model", settings.localModelID,
            "--dataset", datasetDirectory.path,
            "--output", prefix,
            "--name", triggerWord,
            "--steps", String(steps),
            "--rank", String(max(4, min(64, rank))),
            "--learning-rate", String(format: "%.7g", max(0.000001, min(0.001, learningRate))),
            "--resolution", "512",
            "--save-every", "50",
            "--seed", "1647498191",
            "--memory-saver", "minimal",
            "--weights-memory", "justInTime",
            "--no-cotrain-text-model"
        ]

        let logURL = modelsDirectory.appendingPathComponent("\(prefix).training.log")
        guard let stream = fopen(logURL.path, "w+") else {
            throw AIImageStudioError("Could not create the local training log.")
        }
        defer { fclose(stream) }
        let context = DrawThingsCLIContext(
            input: stdin,
            output: stream,
            error: stream,
            environment: ProcessInfo.processInfo.environment,
            executablePath: nil,
            isStandardOutputTTY: false,
            resolvePath: { path in
                if (path as NSString).isAbsolutePath { return path }
                return URL(fileURLWithPath: path, relativeTo: datasetDirectory).standardizedFileURL.path
            }
        )

        notify("Starting local LoRA training with iPhone GPU/Metal. Keep the app open and keep the cooler attached.",
               progress: progress, fraction: 0.1)
        let completion = CompletionFlag()
        let task = Task.detached(priority: .userInitiated) {
            defer { completion.finish() }
            return DrawThingsCLI.run(arguments: arguments, context: context)
        }

        var logOffset: UInt64 = 0
        while !completion.isFinished {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000_000)
            guard let handle = try? FileHandle(forReadingFrom: logURL) else { continue }
            defer { try? handle.close() }
            do {
                try handle.seek(toOffset: logOffset)
                let chunk = try handle.readToEnd() ?? Data()
                logOffset += UInt64(chunk.count)
                guard !chunk.isEmpty, let text = String(data: chunk, encoding: .utf8) else { continue }
                let lines = text.split(whereSeparator: \.isNewline).map(String.init)
                if let stepLine = lines.last(where: { $0.contains("[LoRA] Step") }) {
                    let percent: Double? = {
                        guard let range = stepLine.range(of: #"Step\s+(\d+)/(\d+)"#, options: .regularExpression) else { return nil }
                        let pair = stepLine[range].split(separator: "/")
                        guard pair.count == 2,
                              let current = Int(pair[0].split(separator: " ").last ?? ""),
                              let total = Int(pair[1]), total > 0 else { return nil }
                        return 0.1 + 0.85 * min(1, Double(current) / Double(total))
                    }()
                    notify(stepLine, progress: progress, fraction: percent)
                } else if let lastLine = lines.last(where: { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
                    notify(lastLine, progress: progress, fraction: nil)
                }
            } catch {
                // Best-effort progress parsing. The trainer remains the source of truth.
            }
        }

        let exitCode = await task.value
        guard exitCode == 0 else {
            throw AIImageStudioError("Local LoRA training failed with exit code \(exitCode). Check available storage and memory; SDXL training may exceed this iPhone's limits.")
        }
        let expectedFile = modelsDirectory.appendingPathComponent("\(prefix)_\(steps)_lora_f32.ckpt")
        guard FileManager.default.fileExists(atPath: expectedFile.path) else {
            throw AIImageStudioError("Training returned success but the resulting LoRA checkpoint was not found in the local Models folder.")
        }
        notify("LoRA trained locally and saved on this iPhone.", progress: progress, fraction: 1)
        return expectedFile
    }
}