import Foundation
import UniformTypeIdentifiers
import Darwin
import _MediaGenerationKit
import DrawThingsCLILib
import DataModels

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

    /// Convert imported third-party adapters into Draw Things' native LoRA format.
    /// Raw A1111/ComfyUI safetensors must not be passed directly as DataModels.LoRA:
    /// the engine expects a mapped *_lora_f16.ckpt file. The pinned importer can
    /// identify Cosmos ANIMA 2.5 2B LoRAs and map their tensor names correctly.
    static func prepareLoRAForGeneration(_ fileName: String) throws -> String {
        let directory = try modelsDirectoryURL()
        let sourceURL = directory.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw AIImageStudioError("The selected LoRA file is missing from the local Models folder: \\(fileName)")
        }

        let ext = sourceURL.pathExtension.lowercased()
        let stem = sourceURL.deletingPathExtension().lastPathComponent
        let isNativeDrawThingsCheckpoint = ext == "ckpt"
            && (stem.hasSuffix("_lora_f16") || stem.hasSuffix("_lora_f32"))
        if isNativeDrawThingsCheckpoint { return sourceURL.lastPathComponent }

        let outputName = stem + "_lora_f16.ckpt"
        let outputURL = directory.appendingPathComponent(outputName)
        if let attributes = try? FileManager.default.attributesOfItem(atPath: outputURL.path),
           let size = attributes[.size] as? NSNumber, size.int64Value > 0 {
            return outputName
        }

        do {
            var importer = LoRAImporter(file: sourceURL)
            try importer.import(to: outputURL)
        } catch {
            throw AIImageStudioError(
                "Could not convert the selected LoRA into Draw Things format. Confirm it is a valid LoRA for the selected model. Details: \\(error.localizedDescription)"
            )
        }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: outputURL.path),
              let size = attributes[.size] as? NSNumber, size.int64Value > 0 else {
            throw AIImageStudioError("LoRA conversion finished without creating a valid native checkpoint.")
        }
        DiagnosticsLogger.shared.log(
            "LoRA import converted to Draw Things format • source=\\(sourceURL.lastPathComponent) • native=\\(outputName) • detectedModelVersion=\\(String(describing: importer.version))"
        )
        return outputName
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

        let requestedWidth = multipleOf64(width)
        let requestedHeight = multipleOf64(height)
        var outputWidth: Int
        var outputHeight: Int
        if memory.availableMemoryMB < 1_400 {
            // Keep the latent surface very small when the phone is under pressure.
            // The existing Real-CUGAN pipeline can upscale after diffusion releases its model.
            outputWidth = 384
            outputHeight = 384
        } else if memory.availableMemoryMB < 1_800 {
            let scale = min(1.0, 1024.0 / Double(max(requestedWidth, requestedHeight)))
            outputWidth = multipleOf64(Int(Double(requestedWidth) * scale))
            outputHeight = multipleOf64(Int(Double(requestedHeight) * scale))
        } else {
            outputWidth = requestedWidth
            outputHeight = requestedHeight
        }
        var useTiledDiffusion = true
        // Keep VAE decoding untiled on this iPhone profile. The crash report
        // terminates inside FirstStage.decode while ccv compiles a convolution;
        // the 1-unit (64px) tiled decoder is the highest-confidence trigger.
        var useTiledDecoding = false
        // Draw Things stores tile dimensions and overlap in 64-pixel units.
        // Keep the human-readable sizes in pixels here and convert at assignment.
        // CPU partial-offload is enabled below; 64px tiles minimize transient activation memory.
        var diffusionTileSize = 64
        var decodingTileSize = 64
        let tileOverlapPixels = 0

        DiagnosticsLogger.shared.log(
            "AI Image Studio memory preflight • available=\(Int(memory.availableMemoryMB)) MB • physical=\(Int(memory.physicalMemoryMB)) MB • tier=\(memory.tier.rawValue) • requested=\(requestedWidth)x\(requestedHeight) • selected=\(outputWidth)x\(outputHeight) • tiledDiffusion=\(useTiledDiffusion) • tiledDecoding=\(useTiledDecoding)"
        )

        AIImageGenerationMemoryJournal.shared.record("pre-model-policy")
        let modelsDirectory = try modelsDirectoryURL()
        let hasSelectedLoRA = !settings.localLoraFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        var preparedLoRAFile: String?
        if !benchmarkMode && !recoveringAfterInterruption && enableLoRA && hasSelectedLoRA
            && memory.availableMemoryMB >= 1_200 {
            // Convert/import before loading the large diffusion pipeline so the
            // conversion does not compete with model weights and Metal allocations.
            preparedLoRAFile = try Self.prepareLoRAForGeneration(settings.localLoraFile)
        }
        // Prioritize inference throughput on this device. Forced CPU partial-offload,
        // the lowest capability tier, and a zero-byte weights cache can add repeated
        // weight transfers during every denoising step. Preserve the process defaults
        // and restore all shared settings when this generation call exits.
        let previousMemoryCapacity = DeviceCapability.memoryCapacity
        let previousPartialOffload = DeviceCapability.isPartialOffloadPreferred.load(ordering: .acquiring)
        let previousEnvironment = MediaGenerationEnvironment.default
        DeviceCapability.memoryCapacity = .medium
        DeviceCapability.isPartialOffloadPreferred.store(false, ordering: .releasing)

        var environment = previousEnvironment
        environment.externalUrls = [modelsDirectory]
        MediaGenerationEnvironment.default = environment
        defer {
            MediaGenerationEnvironment.default = previousEnvironment
            DeviceCapability.memoryCapacity = previousMemoryCapacity
            DeviceCapability.isPartialOffloadPreferred.store(previousPartialOffload, ordering: .releasing)
        }
        DiagnosticsLogger.shared.log("AI Image Studio performance policy • partial CPU offload disabled • capacity=medium • default weights cache retained")

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
        if memory.availableMemoryMB < 1_400 {
            outputWidth = 384
            outputHeight = 384
        } else if memory.availableMemoryMB < 1_800 {
            let scale = min(1.0, 1024.0 / Double(max(requestedWidth, requestedHeight)))
            outputWidth = multipleOf64(Int(Double(requestedWidth) * scale))
            outputHeight = multipleOf64(Int(Double(requestedHeight) * scale))
        } else {
            outputWidth = requestedWidth
            outputHeight = requestedHeight
        }
        useTiledDiffusion = true
        useTiledDecoding = false
        DiagnosticsLogger.shared.log(
            "AI Image Studio refreshed pre-load memory • available=\(Int(memory.availableMemoryMB)) MB • selected=\(outputWidth)x\(outputHeight) • diffusionTilePixels=\(diffusionTileSize) • decodeTilePixels=\(decodingTileSize) • overlapPixels=\(tileOverlapPixels)"
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
        // Preserve the requested canvas whenever at least 1.8 GiB remains after model
        // load. The former <4 GiB branch silently clamped normal 832x1216 requests to
        // 512px on the long edge even when the device was stable, producing visibly
        // pixelated results. Retain the conservative fallback only under 1.8 GiB.
        if memory.availableMemoryMB < 1_200 {
            let scale = min(1.0, 384.0 / Double(max(requestedWidth, requestedHeight)))
            outputWidth = multipleOf64(Int(Double(requestedWidth) * scale))
            outputHeight = multipleOf64(Int(Double(requestedHeight) * scale))
        } else if memory.availableMemoryMB < 1_800 {
            let scale = min(1.0, 512.0 / Double(max(requestedWidth, requestedHeight)))
            outputWidth = multipleOf64(Int(Double(requestedWidth) * scale))
            outputHeight = multipleOf64(Int(Double(requestedHeight) * scale))
        }
        if recoveringAfterInterruption {
            DiagnosticsLogger.shared.log("AI Image Studio Recovery Safe Mode • preserving memory-policy output size \(outputWidth)x\(outputHeight) • steps=\(effectiveSteps) • Hires/LoRA disabled")
        }
        if benchmarkMode {
            // Exercise the actual local pipeline with a deliberately small workload.
            // This validates stages and resource lifecycle, not full-resolution throughput.
            outputWidth = 128
            outputHeight = 128
            effectiveSteps = 1
            useTiledDiffusion = false
            useTiledDecoding = false
            AIImageGenerationMemoryJournal.shared.record("benchmark-configured-128x128-1step")
        }
        // Disable both tiled diffusion and tiled VAE decoding on this iPhone
        // profile. The crash report shows SIGABRT in ccv_nnc_hint_tensor_forward
        // while FirstStage.decode compiles a convolution. A 1-unit (64px) VAE
        // tile can produce an unsupported convolution shape. Keep the memory
        // budget bounded with the canvas-size policy above rather than compiling
        // the tiny-tile decoder graph.
        useTiledDiffusion = false
        useTiledDecoding = false
        diffusionTileSize = 64
        decodingTileSize = 64
        DiagnosticsLogger.shared.log(
            "AI Image Studio execution policy • recovery=\(recoveringAfterInterruption) • size=\(outputWidth)x\(outputHeight) • steps=\(effectiveSteps) • tiledDiffusion=\(useTiledDiffusion) • tiledDecode=\(useTiledDecoding) • available=\(Int(currentRenderPerformanceSnapshot().availableMemoryMB)) MB"
        )
        pipeline!.configuration.width = outputWidth
        pipeline!.configuration.height = outputHeight
        pipeline!.configuration.steps = effectiveSteps
        pipeline!.configuration.batchCount = 1
        pipeline!.configuration.batchSize = 1
        pipeline!.configuration.tiledDecoding = useTiledDecoding
        if useTiledDecoding {
            pipeline!.configuration.decodingTileWidth = decodingTileSize / 64
            pipeline!.configuration.decodingTileHeight = decodingTileSize / 64
            pipeline!.configuration.decodingTileOverlap = tileOverlapPixels / 64
        }
        // Keep diffusion tiling disabled for this device until the bundled
        // engine's tiled UNet compilation path is validated on-device.
        pipeline!.configuration.tiledDiffusion = false
        pipeline!.configuration.guidanceScale = Float(max(0, min(20, cfg)))
        pipeline!.configuration.seed = UInt32(truncatingIfNeeded: max(0, seed))
        pipeline!.configuration.shift = Float(max(0.1, min(12.0, shift)))
        switch samplerName {
        case "Euler a":
            pipeline!.configuration.sampler = DataModels.SamplerType.eulerA
        case "DPM++ SDE Karras (closest to Civitai GPU sampler)":
            pipeline!.configuration.sampler = DataModels.SamplerType.dPMPPSDEKarras
        case "DPM++ 2M Karras":
            pipeline!.configuration.sampler = DataModels.SamplerType.dPMPP2MKarras
        case "DDIM":
            pipeline!.configuration.sampler = DataModels.SamplerType.DDIM
        case "UniPC":
            pipeline!.configuration.sampler = DataModels.SamplerType.uniPC
        default:
            break
        }
        pipeline!.configuration.clipSkip = max(1, min(2, clipSkip))

        if hiresEnabled && !benchmarkMode {
            // Hires diffusion multiplies the latent working surface. Keep it disabled
            // for this device profile; use the app's separate Real-CUGAN stage instead.
            if memory.availableMemoryMB >= 4_096 && !recoveringAfterInterruption {
                pipeline!.configuration.hiresFix = true
                pipeline!.configuration.hiresFixWidth = multipleOf64(Int(Double(outputWidth) * hiresScale))
                pipeline!.configuration.hiresFixHeight = multipleOf64(Int(Double(outputHeight) * hiresScale))
                pipeline!.configuration.hiresFixStrength = Float(max(0.05, min(0.95, hiresDenoise)))
                pipeline!.configuration.stage2Steps = max(1, min(60, hiresSteps))
                pipeline!.configuration.stage2Guidance = Float(max(0, min(20, cfg)))
            } else {
                pipeline!.configuration.hiresFix = false
                notify(
                    recoveringAfterInterruption
                        ? "Recovery Safe Mode: high-resolution diffusion is disabled for this retry; upscale afterward with Real-CUGAN."
                        : "Memory Safe Mode: high-resolution diffusion was disabled; you can upscale the result with Real-CUGAN afterward.",
                    progress: progress
                )
            }
        } else {
            pipeline!.configuration.hiresFix = false
        }

        // Match the regular generation memory policy instead of requiring 4 GiB
        // of free memory after model load. The canvas has already been reduced at
        // low headroom (384px below 1.2 GiB, 512px below 1.8 GiB, and at most
        // 768px below 4 GiB). Permit LoRA from 1.2 GiB free upward, where the
        // minimum canvas policy is active, while retaining the recovery retry
        // safeguard because a prior attempt may have been terminated by iOS.
        let canApplyLoRA = !benchmarkMode
            && !recoveringAfterInterruption
            && enableLoRA
            && preparedLoRAFile != nil
            && memory.availableMemoryMB >= 1_200
        if canApplyLoRA, let preparedLoRAFile {
            pipeline!.configuration.loras = [
                DataModels.LoRA(file: preparedLoRAFile, weight: Float(loraWeight), mode: .all)
            ]
            DiagnosticsLogger.shared.log(
                "AI Image Studio LoRA enabled • available=\(Int(memory.availableMemoryMB)) MB • output=\(outputWidth)x\(outputHeight) • adaptive low-memory canvas policy active"
            )
        } else {
            pipeline!.configuration.loras = []
            if enableLoRA && recoveringAfterInterruption {
                notify("Recovery Safe Mode: local LoRA is disabled for this retry; the next normal run can try it again.", progress: progress)
            } else if enableLoRA && hasSelectedLoRA && memory.availableMemoryMB < 1_200 && !benchmarkMode {
                notify("Memory Safe Mode: LoRA needs at least 1.2 GB free after model load; generation will continue at the reduced canvas size.", progress: progress)
            }
        }

        let generationMode = recoveringAfterInterruption
            ? "Recovery Safe Mode retry"
            : "Generating locally on the iPhone GPU/Metal…"
        notify(
            "\(generationMode) • \(outputWidth)x\(outputHeight) • \(effectiveSteps) steps • \(Int(memory.availableMemoryMB)) MB initial headroom",
            progress: progress
        )
        DiagnosticsLogger.shared.log("AI Image Studio generation begin • steps=\(effectiveSteps) • size=\(outputWidth)x\(outputHeight) • tiledDiffusion=\(useTiledDiffusion) • diffusionTilePixels=\(diffusionTileSize) • tiledDecoding=\(useTiledDecoding) • decodeTilePixels=\(decodingTileSize) • overlapPixels=\(tileOverlapPixels)")
        // Poll memory during inference because SDXL can allocate transient Metal buffers during early denoising.
        // Keep the pipeline's strong reference inside the task only. Once the
        // task completes, its operation can release that capture before PNG encoding.
        let preDenoisingMemory = currentRenderPerformanceSnapshot()
        DiagnosticsLogger.shared.log(
            "AI Image Studio immediately before generate() • available=\(Int(preDenoisingMemory.availableMemoryMB)) MB • physical=\(Int(preDenoisingMemory.physicalMemoryMB)) MB • size=\(outputWidth)x\(outputHeight) • steps=\(effectiveSteps) • tiledDiffusion=\(useTiledDiffusion) • tiledDecode=\(useTiledDecoding) • partialOffload=false"
        )
        AIImageGenerationMemoryJournal.shared.record("pre-denoising-allocation")
        AIImageGenerationMemoryJournal.shared.record("pipeline-configured")
        if benchmarkMode { AIImageGenerationMemoryJournal.shared.record("benchmark-pre-denoising") }
        AIImageGenerationMemoryJournal.shared.record("generate-call-start")
        let generationTask = Task { [activePipeline = pipeline!] in
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
            pipeline = nil
            throw AIImageStudioError("The local image engine returned no image.")
        }
        // Release the diffusion pipeline before encoding the decoded image. PNG
        // compression can allocate substantial CPU-side buffers; retaining SDXL's
        // model and Metal resources during that phase needlessly compounds peak RAM.
        pipeline = nil
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
        // A 3.2 GB hard gate made the button appear broken on this 6 GB iPhone,
        // where normal free memory is often around 2.1 GB. Use Draw Things' explicit
        // memory-saver/JIT mode below in the 1.8–3.2 GB band; refuse only when even
        // that mode is unlikely to leave enough room for native compiler allocations.
        guard preflight.availableMemoryMB >= 1_800 else {
            throw AIImageStudioError(
                "Training could not start safely: only \(Int(preflight.availableMemoryMB)) MB is available. Close generation/video tasks, wait for iOS to recover memory, and retry when at least 1.8 GB is available."
            )
        }
        let lowMemoryTraining = preflight.availableMemoryMB < 3_200

        // Restore all shared process settings when the native trainer exits.
        let previousMemoryCapacity = DeviceCapability.memoryCapacity
        let previousPartialOffload = DeviceCapability.isPartialOffloadPreferred.load(ordering: .acquiring)
        let previousEnvironment = MediaGenerationEnvironment.default
        DeviceCapability.memoryCapacity = lowMemoryTraining ? .low : .medium
        DeviceCapability.isPartialOffloadPreferred.store(lowMemoryTraining, ordering: .releasing)
        var trainingEnvironment = previousEnvironment
        if lowMemoryTraining {
            trainingEnvironment.maxTotalWeightsCacheSize = 0
        }
        MediaGenerationEnvironment.default = trainingEnvironment

        let modelsDirectory = try OnDeviceImageGenerator.modelsDirectoryURL()
        let datasetDirectory = modelsDirectory
            .appendingPathComponent("LocalTrainingData", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: datasetDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: datasetDirectory)
            MediaGenerationEnvironment.default = previousEnvironment
            DeviceCapability.memoryCapacity = previousMemoryCapacity
            DeviceCapability.isPartialOffloadPreferred.store(previousPartialOffload, ordering: .releasing)
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

        // Bound optimizer work on this 6 GB phone. Under memory pressure,
        // use a shorter first run and explicit native memory-saving flags rather
        // than letting a normal configuration attempt a high-memory compile.
        let requestedSteps = images.count * max(1, epochs)
        let steps = lowMemoryTraining
            ? max(100, min(120, requestedSteps))
            : max(100, min(300, requestedSteps))
        let prefix = "rife_local_lora_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(10))"
        var arguments = [
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
            "--save-every", "0",
            "--seed", "1647498191",
            "--no-cotrain-text-model"
        ]
        if lowMemoryTraining {
            arguments += ["--memory-saver", "minimal", "--weights-memory", "justInTime"]
            notify(
                "Low-memory training mode • memory saver enabled • \(steps) optimizer steps. Training may be slower, but this avoids the normal high-memory path.",
                progress: progress,
                fraction: 0.1
            )
        }

        let logURL = modelsDirectory.appendingPathComponent("\(prefix).training.log")
        guard let stream = fopen(logURL.path, "w+") else {
            throw AIImageStudioError("Could not create the local training log.")
        }
        defer {
            fclose(stream)
            try? FileManager.default.removeItem(at: logURL)
        }
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