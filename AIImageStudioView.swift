import SwiftUI
import PhotosUI
import Photos
import UniformTypeIdentifiers
import UIKit

struct AIImageStudioView: View {
    let canUpscale: Bool
    let onUpscale: (URL) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var settings = AIImageStudioSettings.load()
    @State private var runPodKey = AIImageStudioKeychain.read("runpod")
    @State private var trainerToken = AIImageStudioKeychain.read("trainer")
    @State private var activeTab = "Generate"

    @State private var positivePrompt = """
newest, very awa, masterpiece, high quality, high resolution, amazing quality, best quality, good lighting, detailed eyes, anime coloring, anime screencap, looking at viewer, large breasts, parted lips, :o, looking to the side, blush, standing, arched back, shoulders tilted, one hand behind neck, other hand resting on thigh, detailed background, 8k, blurry background, beach, night, 1girl, solo, mizuhara chizuru, long hair, brown hair, brown eyes, sky blue micro bikini, tight clothes, cleavage, covered nipples, covered pussy
"""
    @State private var negativePrompt = """
worst quality, bad quality, low quality, lowres, scan artifacts, jpeg artifacts, sketch, bad quality, jpeg, artifacts, signature, username, text, logo, bad anatomy, artist name, artist logo, extra limbs, extra digit, extra legs, extra arms, blurry background, simple background, huge breasts, puckered anus
"""
    @State private var width = 848.0
    @State private var height = 1200.0
    @State private var steps = 8.0
    @State private var cfg = 1.0
    @State private var shift = 3.0
    @State private var samplerName = "Euler a"
    @State private var schedulerName = "Normal"
    @State private var seedText = "1647498191"
    @State private var randomSeed = false
    @State private var turboWeight = 1.0
    @State private var characterWeight = 0.7
    @State private var enableCharacterLora = true
    @State private var isGenerating = false
    @State private var generationStatus = ""
    @State private var errorText = ""
    @State private var generatedImage: UIImage?
    @State private var generatedURL: URL?
    @State private var showSaveAlert = false
    @State private var saveMessage = ""

    @State private var trainingItems: [PhotosPickerItem] = []
    @State private var trainingCaption = "anime illustration, clean lineart, detailed eyes, high quality"
    @State private var triggerWord = "my_anime_subject"
    @State private var trainingRank = 16.0
    @State private var trainingEpochs = 10.0
    @State private var learningRate = 0.0001
    @State private var isTraining = false
    @State private var trainingStatus = ""
    @State private var trainedLoRAURL: URL?
    @State private var trainerProgress = 0.0

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Studio", selection: $activeTab) {
                        Text("Generate").tag("Generate")
                        Text("LoRA Trainer").tag("Train")
                        Text("Settings").tag("Settings")
                    }
                    .pickerStyle(.segmented)
                }

                if activeTab == "Generate" {
                    generationSections
                } else if activeTab == "Train" {
                    trainingSections
                } else {
                    connectionSections
                }

                if !errorText.isEmpty {
                    Section("Error") { Text(errorText).foregroundStyle(.red) }
                }
            }
            .navigationTitle("AI Image Studio")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save Settings") { saveSettings() }
                }
            }
            .alert("AI Image Studio", isPresented: $showSaveAlert) {
                Button("OK", role: .cancel) {}
            } message: { Text(saveMessage) }
        }
    }

    private var generationSections: some View {
        Group {
            Section("Positive Prompt") {
                TextEditor(text: $positivePrompt).frame(minHeight: 150)
                Text("The prompt from the supplied PNG is loaded by default. Edit it freely before generating.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Negative Prompt") {
                TextEditor(text: $negativePrompt).frame(minHeight: 110)
            }
            Section("Model Files on GPU") {
                TextField("Checkpoint filename", text: $settings.checkpointName)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                TextField("Text encoder filename", text: $settings.textEncoderName)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                TextField("VAE filename", text: $settings.vaeName)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Text("These filenames must exist in the matching ComfyUI model folders on your GPU worker. The app does not silently substitute a different checkpoint.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("LoRAs") {
                TextField("Turbo LoRA filename", text: $settings.turboLoraName)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                HStack {
                    Text("Turbo-ANIMA strength")
                    Spacer()
                    Text(String(format: "%.2f", turboWeight)).monospacedDigit()
                }
                Slider(value: $turboWeight, in: 0...1.5, step: 0.05)
                Toggle("Enable character LoRA", isOn: $enableCharacterLora)
                TextField("Character LoRA filename", text: $settings.characterLoraName)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .disabled(!enableCharacterLora)
                HStack {
                    Text("Character strength")
                    Spacer()
                    Text(String(format: "%.2f", characterWeight)).monospacedDigit()
                }
                Slider(value: $characterWeight, in: 0...1.5, step: 0.05).disabled(!enableCharacterLora)
                Text("The character LoRA from the source image is not yet verified. Disable it until that exact file is installed, otherwise the worker will report a missing-file error.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Generation Settings") {
                Picker("Sampler", selection: $samplerName) {
                    Text("Euler a").tag("Euler a")
                    Text("Euler").tag("Euler")
                    Text("DPM++ 2M").tag("DPM++ 2M")
                    Text("DPM++ 2M SDE").tag("DPM++ 2M SDE")
                }
                Picker("Schedule type", selection: $schedulerName) {
                    Text("Normal").tag("Normal")
                    Text("Karras").tag("Karras")
                    Text("Simple").tag("Simple")
                    Text("SGM Uniform").tag("SGM Uniform")
                }
                HStack {
                    Text("Steps")
                    Spacer()
                    Text("\(Int(steps))").monospacedDigit()
                }
                Slider(value: $steps, in: 1...40, step: 1)
                HStack {
                    Text("CFG scale")
                    Spacer()
                    Text(String(format: "%.1f", cfg)).monospacedDigit()
                }
                Slider(value: $cfg, in: 0...10, step: 0.5)
                HStack {
                    Text("Shift")
                    Spacer()
                    Text(String(format: "%.1f", shift)).monospacedDigit()
                }
                Slider(value: $shift, in: 0...6, step: 0.5)
                HStack {
                    Text("Width")
                    Spacer()
                    Text("\(Int(width)) px").monospacedDigit()
                }
                Slider(value: $width, in: 512...1536, step: 8)
                HStack {
                    Text("Height")
                    Spacer()
                    Text("\(Int(height)) px").monospacedDigit()
                }
                Slider(value: $height, in: 512...1536, step: 8)
                TextField("Seed", text: $seedText)
                    .keyboardType(.numberPad)
                Toggle("Random seed", isOn: $randomSeed)
                Text("Defaults recovered from metadata: 8 steps, CFG 1, shift 3, Euler a, Normal, seed 1647498191. The generated resolution defaults to the PNG dimensions, 848 × 1200.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Generate") {
                if let image = generatedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    if let url = generatedURL {
                        ShareLink(item: url) { Label("Share / Export PNG", systemImage: "square.and.arrow.up") }
                    }
                    Button {
                        saveGeneratedImage()
                    } label: { Label("Save to Photos", systemImage: "photo.badge.arrow.down") }
                    Button {
                        guard let url = generatedURL, canUpscale else { return }
                        onUpscale(url)
                        dismiss()
                    } label: { Label("Upscale with Real-CUGAN + Sharpie", systemImage: "arrow.up.left.and.arrow.down.right") }
                    .disabled(generatedURL == nil || !canUpscale)
                    if !canUpscale {
                        Text("Finish or clear the current processing/recovery job before sending a generated image to the local upscale pipeline.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if isGenerating {
                    ProgressView(generationStatus.isEmpty ? "Generating…" : generationStatus)
                } else {
                    Button {
                        Task { await generateImage() }
                    } label: {
                        Label("Generate Anime Image", systemImage: "sparkles")
                    }
                    .buttonStyle(.borderedProminent)
                }
                if !generationStatus.isEmpty { Text(generationStatus).font(.caption).foregroundStyle(.secondary) }
                Text("Generation runs on your configured remote GPU endpoint. Local Real-CUGAN upscaling runs inside RIFE60GhostGuard after the generated PNG is queued.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var trainingSections: some View {
        Group {
            Section("Dataset") {
                PhotosPicker(selection: $trainingItems, maxSelectionCount: 40, matching: .images) {
                    Label("Select training images", systemImage: "photo.stack")
                }
                Text("\(trainingItems.count) images selected")
                    .font(.caption).foregroundStyle(.secondary)
                TextField("Trigger word", text: $triggerWord)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Text("Shared caption or style description")
                TextEditor(text: $trainingCaption).frame(minHeight: 80)
                Text("For a character LoRA, use clear varied images of the same subject, consistent captions, and a unique trigger word. Only upload images you have permission to use.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Training Parameters") {
                HStack { Text("Network rank"); Spacer(); Text("\(Int(trainingRank))") }
                Slider(value: $trainingRank, in: 4...64, step: 4)
                HStack { Text("Epochs"); Spacer(); Text("\(Int(trainingEpochs))") }
                Slider(value: $trainingEpochs, in: 1...30, step: 1)
                HStack { Text("Learning rate"); Spacer(); Text(String(format: "%.5f", learningRate)) }
                Slider(value: $learningRate, in: 0.00001...0.0003, step: 0.00001)
                Text("Uses the ANIMA LoRA training endpoint contract documented in AI_IMAGE_STUDIO_SETUP.md. Training requires a separate GPU trainer service with kohya-ss/sd-scripts installed; the stock ComfyUI generation endpoint does not train LoRAs.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Train LoRA") {
                if isTraining {
                    ProgressView(value: trainerProgress)
                    Text(trainingStatus).font(.caption)
                } else {
                    Button {
                        Task { await trainLoRA() }
                    } label: { Label("Start ANIMA LoRA Training", systemImage: "cpu") }
                    .disabled(trainingItems.count < 3 || settings.trainerBaseURL.isEmpty)
                    Text("Select at least three images and configure the trainer URL and token in Settings.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let url = trainedLoRAURL {
                    ShareLink(item: url) { Label("Export trained .safetensors", systemImage: "square.and.arrow.down") }
                }
            }
        }
    }

    private var connectionSections: some View {
        Group {
            Section("Image Generation Endpoint") {
                TextField("RunPod Serverless endpoint ID", text: $settings.runPodEndpointID)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                SecureField("RunPod API key", text: $runPodKey)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Text("Use a RunPod Serverless endpoint running the ComfyUI worker with a custom ANIMA workflow. The endpoint must have the checkpoint, text encoder, VAE and enabled LoRAs installed.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("LoRA Trainer Endpoint") {
                TextField("Trainer base URL (HTTPS)", text: $settings.trainerBaseURL)
                    .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                SecureField("Trainer API token", text: $trainerToken)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Text("This is a separate training API, not the RunPod ComfyUI endpoint. It must implement POST /api/anima/lora/train and GET /api/anima/lora/train/{job_id}; the response contract is described in the setup guide.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Model Downloads and Compatibility") {
                Link("Official ANIMA model files", destination: URL(string: "https://huggingface.co/circlestone-labs/Anima")!)
                Link("Turbo-ANIMA-v2.9 model page", destination: URL(string: "https://civarchive.com/models/2619830?modelVersionId=3139645")!)
                Link("LoRA training documentation", destination: URL(string: "https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md")!)
                Text("Model filenames must match files installed on the remote ComfyUI instance. Large model weights are not bundled in the IPA. The character LoRA fingerprint 160fca5c6aae is still unverified.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                Button("Save Endpoint Settings", action: saveSettings)
                Button("Clear Saved Endpoint Credentials", role: .destructive) {
                    try? AIImageStudioKeychain.write("", account: "runpod")
                    try? AIImageStudioKeychain.write("", account: "trainer")
                    runPodKey = ""
                    trainerToken = ""
                    showAlert("Credentials cleared from this app's Keychain.")
                }
            }
        }
    }

    private func saveSettings() {
        do {
            try settings.save()
            try AIImageStudioKeychain.write(runPodKey, account: "runpod")
            try AIImageStudioKeychain.write(trainerToken, account: "trainer")
            showAlert("Settings saved. API credentials are stored in iOS Keychain.")
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func showAlert(_ message: String) {
        saveMessage = message
        showSaveAlert = true
    }

    private func generateImage() async {
        isGenerating = true
        errorText = ""
        generatedImage = nil
        generatedURL = nil
        defer { isGenerating = false }
        do {
            try settings.save()
            try AIImageStudioKeychain.write(runPodKey, account: "runpod")
            let resolvedSeed: Int64
            if randomSeed { resolvedSeed = Int64.random(in: 0...Int64.max / 2) }
            else if let value = Int64(seedText), value >= 0 { resolvedSeed = value }
            else { throw AIImageStudioError("Enter a non-negative integer seed or enable Random seed.") }
            generationStatus = "Preparing workflow…"
            let data = try await AIImageStudioClient.generate(
                settings: settings,
                apiKey: runPodKey,
                positivePrompt: positivePrompt,
                negativePrompt: negativePrompt,
                width: max(64, Int(width / 8) * 8),
                height: max(64, Int(height / 8) * 8),
                steps: Int(steps),
                cfg: cfg,
                shift: shift,
                samplerName: samplerName,
                schedulerName: schedulerName,
                seed: resolvedSeed,
                turboWeight: turboWeight,
                characterWeight: characterWeight,
                enableCharacterLora: enableCharacterLora,
                progress: { generationStatus = $0 }
            )
            guard let image = UIImage(data: data), let png = image.pngData() else {
                throw AIImageStudioError("The endpoint returned data that could not be decoded as an image.")
            }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60_AI_\(UUID().uuidString).png")
            try png.write(to: url, options: .atomic)
            generatedImage = image
            generatedURL = url
            generationStatus = "Generation complete • \(image.size.width.rounded()) × \(image.size.height.rounded())"
        } catch {
            errorText = error.localizedDescription
            generationStatus = "Generation failed"
        }
    }

    private func saveGeneratedImage() {
        guard let image = generatedImage else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                Task { @MainActor in showAlert("Photos access was not granted. Use Share / Export PNG to save through Files instead.") }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            }) { success, error in
                Task { @MainActor in
                    showAlert(success ? "Image saved to Photos." : "Could not save image: \(error?.localizedDescription ?? "unknown error")")
                }
            }
        }
    }

    private func trainLoRA() async {
        isTraining = true
        trainerProgress = 0
        errorText = ""
        trainedLoRAURL = nil
        defer { isTraining = false }
        do {
            try settings.save()
            try AIImageStudioKeychain.write(trainerToken, account: "trainer")
            var images: [AITrainingImage] = []
            var totalBytes = 0
            for (index, item) in trainingItems.enumerated() {
                trainingStatus = "Reading dataset image \(index + 1) of \(trainingItems.count)…"
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    throw AIImageStudioError("Could not read training image \(index + 1).")
                }
                totalBytes += data.count
                guard totalBytes <= 180 * 1024 * 1024 else {
                    throw AIImageStudioError("The selected dataset exceeds 180 MB. Select fewer or smaller images.")
                }
                let ext = item.supportedContentTypes.first?.preferredFilenameExtension ?? "png"
                images.append(AITrainingImage(filename: "training_\(index + 1).\(ext)", data: data))
                trainerProgress = Double(index + 1) / Double(trainingItems.count) * 0.2
            }
            let jobID = try await AIImageStudioClient.submitLoRATraining(
                settings: settings,
                token: trainerToken,
                images: images,
                caption: trainingCaption,
                triggerWord: triggerWord,
                rank: Int(trainingRank),
                epochs: Int(trainingEpochs),
                learningRate: learningRate
            )
            trainingStatus = "Training job queued: \(jobID)"
            let deadline = Date().addingTimeInterval(6 * 60 * 60)
            while Date() < deadline {
                try Task.checkCancellation()
                try await Task.sleep(nanoseconds: 5_000_000_000)
                let state = try await AIImageStudioClient.trainingStatus(settings: settings, token: trainerToken, jobID: jobID)
                let status = (state["status"] as? String ?? "unknown").lowercased()
                let progress = state["progress"] as? Double ?? 0
                trainerProgress = 0.2 + min(max(progress, 0), 1) * 0.75
                trainingStatus = "Training \(status) • \(Int(progress * 100))%"
                if status == "failed" || status == "cancelled" {
                    throw AIImageStudioError(state["error"] as? String ?? "The training service reported \(status).")
                }
                if status == "completed" {
                    guard let downloadURL = state["lora_url"] as? String ?? state["download_url"] as? String else {
                        throw AIImageStudioError("Training completed but the service did not return lora_url or download_url.")
                    }
                    trainedLoRAURL = try await AIImageStudioClient.downloadFile(from: downloadURL, token: trainerToken)
                    trainerProgress = 1
                    trainingStatus = "LoRA training complete and file downloaded."
                    return
                }
            }
            throw AIImageStudioError("Training is still running after six hours. The job ID is \(jobID); check the trainer dashboard.")
        } catch {
            errorText = error.localizedDescription
            trainingStatus = "Training failed or connection lost."
        }
    }
}
