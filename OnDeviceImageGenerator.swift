import Foundation
import UniformTypeIdentifiers
import Darwin
import _MediaGenerationKit
import DrawThingsCLILib
import DataModels

enum OnDeviceImageGenerator {
    static let defaultModelID = "animagine_xl_v3.1_q6p_q8p.ckpt"

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

    static func generate(
        settings: AIImageStudioSettings,
        positivePrompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Double,
        seed: Int64,
        enableLoRA: Bool,
        loraWeight: Double,
        hiresEnabled: Bool,
        hiresScale: Double,
        hiresSteps: Int,
        hiresDenoise: Double,
        clipSkip: Int,
        progress: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> URL {
        guard !positivePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AIImageStudioError("The positive prompt cannot be empty.")
        }
        let modelID = settings.localModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelID.isEmpty else {
            throw AIImageStudioError("Choose a supported local model catalog ID.")
        }

        // Diffusion inference competes with the rest of iOS for unified memory.
        // Pick a conservative working size before loading SDXL, then enable the
        // engine's native tiled paths when headroom is limited.
        let memory = currentRenderPerformanceSnapshot()
        guard memory.availableMemoryMB >= 450 else {
            throw AIImageStudioError(
                "AI Image Studio paused before loading the model because only \(Int(memory.availableMemoryMB)) MB of memory headroom is available. Close other apps, wait for memory to recover, and retry."
            )
        }

        let requestedWidth = multipleOf64(width)
        let requestedHeight = multipleOf64(height)
        let outputWidth: Int
        let outputHeight: Int
        if memory.availableMemoryMB < 1_000 {
            outputWidth = multipleOf64(min(requestedWidth, 512))
            outputHeight = multipleOf64(min(requestedHeight, 768))
        } else if memory.availableMemoryMB < 1_800 {
            let scale = min(1.0, 1024.0 / Double(max(requestedWidth, requestedHeight)))
            outputWidth = multipleOf64(Int(Double(requestedWidth) * scale))
            outputHeight = multipleOf64(Int(Double(requestedHeight) * scale))
        } else {
            outputWidth = requestedWidth
            outputHeight = requestedHeight
        }
        let useTiledDiffusion = memory.availableMemoryMB < 1_800
        let useTiledDecoding = memory.availableMemoryMB < 2_400

        DiagnosticsLogger.shared.log(
            "AI Image Studio memory preflight • available=\(Int(memory.availableMemoryMB)) MB • physical=\(Int(memory.physicalMemoryMB)) MB • tier=\(memory.tier.rawValue) • requested=\(requestedWidth)x\(requestedHeight) • selected=\(outputWidth)x\(outputHeight) • tiledDiffusion=\(useTiledDiffusion) • tiledDecoding=\(useTiledDecoding)"
        )

        let modelsDirectory = try modelsDirectoryURL()
        var environment = MediaGenerationEnvironment.default
        environment.externalUrls = [modelsDirectory]
        // This device is below Draw Things' 47 GiB weights-cache threshold.
        // Keep the cache explicitly disabled on constrained devices.
        environment.maxTotalWeightsCacheSize = 0
        MediaGenerationEnvironment.default = environment

        notify("Checking local model files…", progress: progress)
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
            Task { @MainActor in progress(message) }
        }

        try Task.checkCancellation()
        notify("Loading local Metal model…", progress: progress)
        var pipeline: MediaGenerationPipeline? = try await MediaGenerationPipeline.fromPretrained(
            resolvedModel.file,
            backend: .local
        )
        pipeline!.configuration.width = outputWidth
        pipeline!.configuration.height = outputHeight
        pipeline!.configuration.steps = max(1, min(60, steps))
        pipeline!.configuration.batchCount = 1
        pipeline!.configuration.batchSize = 1
        pipeline!.configuration.tiledDecoding = useTiledDecoding
        if useTiledDecoding {
            pipeline!.configuration.decodingTileWidth = 512
            pipeline!.configuration.decodingTileHeight = 512
            pipeline!.configuration.decodingTileOverlap = 32
        }
        pipeline!.configuration.tiledDiffusion = useTiledDiffusion
        if useTiledDiffusion {
            pipeline!.configuration.diffusionTileWidth = 512
            pipeline!.configuration.diffusionTileHeight = 512
            pipeline!.configuration.diffusionTileOverlap = 32
        }
        pipeline!.configuration.guidanceScale = Float(max(0, min(20, cfg)))
        pipeline!.configuration.seed = UInt32(truncatingIfNeeded: max(0, seed))
        pipeline!.configuration.clipSkip = max(1, min(2, clipSkip))

        if hiresEnabled {
            // Hires diffusion can double or quadruple the latent working surface.
            // Skip it automatically when the current memory budget is constrained.
            if memory.availableMemoryMB >= 2_400 {
                pipeline!.configuration.hiresFix = true
                pipeline!.configuration.hiresFixWidth = multipleOf64(Int(Double(outputWidth) * hiresScale))
                pipeline!.configuration.hiresFixHeight = multipleOf64(Int(Double(outputHeight) * hiresScale))
                pipeline!.configuration.hiresFixStrength = Float(max(0.05, min(0.95, hiresDenoise)))
                pipeline!.configuration.stage2Steps = max(1, min(60, hiresSteps))
                pipeline!.configuration.stage2Guidance = Float(max(0, min(20, cfg)))
            } else {
                pipeline!.configuration.hiresFix = false
                notify("Memory Safe Mode: high-resolution diffusion was disabled; you can upscale the result with Real-CUGAN afterward.", progress: progress)
            }
        } else {
            pipeline!.configuration.hiresFix = false
        }

        if enableLoRA && !settings.localLoraFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            pipeline!.configuration.loras = [
                DataModels.LoRA(file: settings.localLoraFile, weight: Float(loraWeight), mode: .all)
            ]
        } else {
            pipeline!.configuration.loras = []
        }

        notify(
            "Generating locally on the iPhone GPU/Metal… \(outputWidth)x\(outputHeight) • \(Int(memory.availableMemoryMB)) MB initial headroom",
            progress: progress
        )
        var results = try await pipeline!.generate(prompt: positivePrompt, negativePrompt: negativePrompt) { state, _ in
            let message: String
            switch state {
            case .resolvingBackend(_):
                message = "Preparing on-device backend…"
            case .resolvingModel(let name):
                message = "Resolving \(name)…"
            case .preparing:
                message = "Preparing local generation…"
            case .ensuringResources:
                message = "Checking local model resources…"
            case .uploading(_, _), .downloading(_, _):
                message = "Processing local model resources…"
            case .encodingText:
                message = "Encoding prompt on device…"
            case .encodingInputs:
                message = "Preparing generation inputs…"
            case .generating(let step, let total):
                let currentMemory = currentRenderPerformanceSnapshot()
                message = "Local GPU generation: step \(step)/\(total) • \(Int(currentMemory.availableMemoryMB)) MB free"
                DiagnosticsLogger.shared.log(
                    "AI Image Studio step \(step)/\(total) • available=\(Int(currentMemory.availableMemoryMB)) MB • thermal=\(currentMemory.thermalAndMode)"
                )
            case .decoding:
                message = "Decoding image…"
            case .postprocessing:
                message = "Running local high-resolution pass…"
            case .cancelling:
                message = "Cancelling generation…"
            case .completed:
                message = "Generation complete."
            case .cancelled:
                message = "Generation cancelled."
            }
            Task { @MainActor in progress(message) }
        }
        guard let result = results.first else {
            pipeline = nil
            throw AIImageStudioError("The local image engine returned no image.")
        }
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60_AI_\(UUID().uuidString).png")
        do {
            try result.write(to: outputURL, type: .png)
        } catch {
            pipeline = nil
            results.removeAll(keepingCapacity: false)
            throw error
        }
        // The UI consumes the file URL directly. Do not read the full PNG into
        // Data here, which duplicates the encoded image while the model is resident.
        results.removeAll(keepingCapacity: false)
        pipeline = nil
        notify("PNG saved. Releasing diffusion pipeline memory…", progress: progress)
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

        let modelsDirectory = try OnDeviceImageGenerator.modelsDirectoryURL()
        let datasetDirectory = modelsDirectory
            .appendingPathComponent("LocalTrainingData", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: datasetDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: datasetDirectory) }

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

        let steps = max(100, min(1000, images.count * max(1, epochs)))
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
            "--save-every", "0",
            "--seed", "1647498191",
            "--memory-saver", "minimal",
            "--weights-memory", "justInTime",
            "--no-cotrain-text-model"
        ]

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
