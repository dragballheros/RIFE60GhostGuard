import SwiftUI
import PhotosUI
import Photos
import UniformTypeIdentifiers
import UIKit

private struct AIImageStudioDraft: Codable {
    var modelProfile: String
    var positivePrompt: String
    var negativePrompt: String
    var width: Double
    var height: Double
    var steps: Double
    var cfg: Double
    var shift: Double
    var hiresEnabled: Bool
    var hiresScale: Double
    var hiresSteps: Double
    var hiresDenoise: Double
    var upscalerName: String
    var clipSkip: Double
    var ensd: Double
    var tokenMergingRatio: Double
    var tokenMergingHiresRatio: Double
    var samplerName: String
    var schedulerName: String
    var seedText: String
    var randomSeed: Bool
    var turboWeight: Double
    var characterWeight: Double
    var enableCharacterLora: Bool
    static let key = "RIFE60.AIImageStudio.GenerationDraft.v1"
}

struct AIImageStudioView: View {
    let canUpscale: Bool
    let onUpscale: (URL) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var settings = AIImageStudioSettings.load()
    @State private var activeTab = "Generate"
    @State private var modelProfile = UserDefaults.standard.string(forKey: "RIFE60.AIImageStudio.SelectedProfile") ?? "Turbo-ANIMA"
    @State private var didRestoreDraft = false

    @State private var positivePrompt = """
newest, very awa, masterpiece, high quality, high resolution, amazing quality, best quality, good lighting, detailed eyes, anime coloring, anime screencap, looking at viewer, large breasts, parted lips, :o, looking to the side, blush, standing, arched back, shoulders tilted, one hand behind neck, other hand resting on thigh, detailed background, 8k, blurry background, beach, night, 1girl, solo, mizuhara chizuru, long hair, brown hair, brown eyes, sky blue micro bikini, tight clothes, cleavage, covered nipples, covered pussy
"""
    @State private var negativePrompt = """
worst quality, bad quality, low quality, lowres, scan artifacts, jpeg artifacts, sketch, bad quality, jpeg, artifacts, signature, username, text, logo, bad anatomy, artist name, artist logo, extra limbs, extra digit, extra legs, extra arms, blurry background, simple background, huge breasts, puckered anus
"""
    @State private var width = 832.0
    @State private var height = 1216.0
    @State private var steps = 28.0
    @State private var cfg = 5.5
    @State private var shift = 3.0
    @State private var hiresEnabled = false
    @State private var hiresScale = 2.0
    @State private var hiresSteps = 20.0
    @State private var hiresDenoise = 0.5
    @State private var upscalerName = "RealESRGAN_x4plus_anime_6B.pth"
    @State private var clipSkip = 2.0
    @State private var ensd = 31337.0
    @State private var tokenMergingRatio = 0.1
    @State private var tokenMergingHiresRatio = 0.1
    @State private var samplerName = "Euler a"
    @State private var schedulerName = "Normal"
    @State private var seedText = "1647498191"
    @State private var randomSeed = false
    @State private var turboWeight = 1.0
    @State private var characterWeight = 0.7
    @State private var enableCharacterLora = false
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
            .onAppear { restoreGenerationDraft() }
            .onDisappear { saveGenerationDraft(); try? settings.save() }
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
            Section("Generation Preset") {
                Picker("Model profile", selection: $modelProfile) {
                    Text("On-device Anime (Animagine XL)").tag("Turbo-ANIMA")
                }
                .onChange(of: modelProfile) { profile in applyModelProfile(profile) }
                Text("Animagine XL v3.1 runs locally on the iPhone GPU/Metal. The first generation downloads the model and required files; allow several GB of free storage.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Positive Prompt") {
                TextEditor(text: $positivePrompt).frame(minHeight: 150)
                Text("The prompt from the supplied PNG is loaded by default. Edit it freely before generating.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Negative Prompt") {
                TextEditor(text: $negativePrompt).frame(minHeight: 110)
            }
            Section("On-device Model") {
                TextField("Supported local model catalog ID", text: $settings.localModelID)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Text("Default: Animagine XL v3.1 (8-bit). Model weights are downloaded into this app's local Models folder, then inference runs on the iPhone.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("The supplied screenChantvMerge_v20 ANIMA checkpoint and Turbo-ANIMA-v2.9 LoRA are not directly compatible with this engine as-is. This local profile uses a supported SDXL anime model instead of sending work to a remote endpoint.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Local LoRA") {
                Toggle("Apply trained local LoRA", isOn: $enableCharacterLora)
                TextField("Trained LoRA filename", text: $settings.localLoraFile)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .disabled(!enableCharacterLora)
                HStack {
                    Text("LoRA strength")
                    Spacer()
                    Text(String(format: "%.2f", characterWeight)).monospacedDigit()
                }
                Slider(value: $characterWeight, in: 0...1.5, step: 0.05).disabled(!enableCharacterLora)
                Text("A LoRA trained by this app is stored locally and can be applied to the compatible SDXL model. External ANIMA and ComfyUI LoRAs are not assumed compatible.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Generation Settings") {
                Text("The local engine uses the model's recommended sampler and schedule. Steps, CFG, resolution, seed, optional high-resolution diffusion, and your selected local LoRA are applied directly.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack { Text("Steps"); Spacer(); Text("\(Int(steps))").monospacedDigit() }
                Slider(value: $steps, in: 4...40, step: 1)
                HStack { Text("CFG scale"); Spacer(); Text(String(format: "%.1f", cfg)).monospacedDigit() }
                Slider(value: $cfg, in: 1...10, step: 0.5)
                HStack { Text("Width"); Spacer(); Text("\(Int(width)) px").monospacedDigit() }
                Slider(value: $width, in: 512...1024, step: 64)
                HStack { Text("Height"); Spacer(); Text("\(Int(height)) px").monospacedDigit() }
                Slider(value: $height, in: 512...1536, step: 64)
                TextField("Seed", text: $seedText).keyboardType(.numberPad)
                Toggle("Random seed", isOn: $randomSeed)
                Toggle("Enable high-resolution diffusion pass", isOn: $hiresEnabled)
                if hiresEnabled {
                    HStack { Text("Hires upscale"); Spacer(); Text(String(format: "%.1fx", hiresScale)) }
                    Slider(value: $hiresScale, in: 1.25...2.0, step: 0.25)
                    HStack { Text("Hires steps"); Spacer(); Text("\(Int(hiresSteps))") }
                    Slider(value: $hiresSteps, in: 4...30, step: 1)
                    HStack { Text("Hires denoising"); Spacer(); Text(String(format: "%.2f", hiresDenoise)) }
                    Slider(value: $hiresDenoise, in: 0.1...0.8, step: 0.05)
                    Text("The high-resolution pass uses more unified memory and can be terminated by iOS. For reliability on this phone, leave it off and run the existing Real-CUGAN + Sharpie pipeline after generation.")
                        .font(.caption).foregroundStyle(.secondary)
                }
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
                Text("Image generation runs on this iPhone. No remote GPU endpoint or paid inference service is used. The generated PNG can be sent directly into the existing local Real-CUGAN + Sharpie upscale pipeline.")
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
                Text("Training runs on this iPhone using the local Draw Things trainer and Metal/GPU. The supported path trains an SDXL LoRA for Animagine XL, not the separate ANIMA checkpoint. Memory is a hard limit even with a cooler, so begin with a small dataset, keep the app foregrounded, and expect long training times.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Train LoRA") {
                if isTraining {
                    ProgressView(value: trainerProgress)
                    Text(trainingStatus).font(.caption)
                } else {
                    Button {
                        Task { await trainLoRA() }
                    } label: { Label("Train LoRA on this iPhone", systemImage: "cpu") }
                    .disabled(trainingItems.count < 3)
                    Text("Select at least three images. Training uses the same local model folder as generation and does not upload the dataset.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let url = trainedLoRAURL {
                    ShareLink(item: url) { Label("Export trained local LoRA checkpoint", systemImage: "square.and.arrow.down") }
                }
            }
        }
    }

    private var connectionSections: some View {
        Group {
            Section("Local GPU/Metal Processing") {
                Label("No remote endpoint", systemImage: "iphone.gen3")
                Text("Generation, compatible SDXL LoRA training, and model storage are local to this iPhone. No RunPod account, API key, or paid inference service is required.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Models are downloaded on first use and stored in Application Support/RIFE60GhostGuard/Models. Keep several GB of free storage available.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Model Compatibility") {
                Text("The default model is Animagine XL v3.1 (8-bit), exposed by the embedded local engine. The original screenChantvMerge_v20 ANIMA checkpoint, Turbo-ANIMA-v2.9 LoRA, and WAI Illustrious checkpoint are not assumed compatible. Exact ANIMA support needs a native model-import or architecture path.")
                    .font(.caption).foregroundStyle(.secondary)
                Link("Embedded local engine source and license", destination: URL(string: "https://github.com/drawthingsai/draw-things-community/tree/4f288803ac898525012c0fe8d998c85bcb920b70")!)
            }
            Section {
                Button("Reset model to Animagine XL v3.1 (8-bit)") {
                    settings.localModelID = OnDeviceImageGenerator.defaultModelID
                    try? settings.save()
                    showAlert("The local model selection was reset.")
                }
                Button("Clear selected local LoRA", role: .destructive) {
                    settings.localLoraFile = ""
                    enableCharacterLora = false
                    try? settings.save()
                    showAlert("The trained LoRA is no longer selected. Its checkpoint remains on this iPhone.")
                }
                Button("Save Local Settings", action: saveSettings)
            }
        }
    }

    private func applyModelProfile(_ profile: String) {
        modelProfile = "Turbo-ANIMA"
        UserDefaults.standard.set(modelProfile, forKey: "RIFE60.AIImageStudio.SelectedProfile")
        settings.localModelID = OnDeviceImageGenerator.defaultModelID
        errorText = ""
        generationStatus = ""
        generatedImage = nil
        generatedURL = nil
        width = 832
        height = 1216
        steps = 28
        cfg = 5.5
        shift = 0
        hiresEnabled = false
        hiresScale = 1.5
        hiresSteps = 12
        hiresDenoise = 0.35
        clipSkip = 2
        samplerName = "Model recommended"
        schedulerName = "Model recommended"
    }

    private func restoreGenerationDraft() {
        guard !didRestoreDraft else { return }
        didRestoreDraft = true
        guard let data = UserDefaults.standard.data(forKey: AIImageStudioDraft.key),
              let draft = try? JSONDecoder().decode(AIImageStudioDraft.self, from: data) else {
            applyModelProfile("Turbo-ANIMA")
            return
        }
        modelProfile = "Turbo-ANIMA"
        positivePrompt = draft.positivePrompt
        negativePrompt = draft.negativePrompt
        // Migrate the old remote ANIMA/Illustrious controls to valid local SDXL defaults.
        width = 832
        height = 1216
        steps = 28
        cfg = 5.5
        shift = 0
        hiresEnabled = false
        hiresScale = 1.5
        hiresSteps = 12
        hiresDenoise = 0.35
        upscalerName = draft.upscalerName
        clipSkip = 2
        ensd = draft.ensd
        tokenMergingRatio = draft.tokenMergingRatio
        tokenMergingHiresRatio = draft.tokenMergingHiresRatio
        samplerName = "Model recommended"
        schedulerName = "Model recommended"
        seedText = draft.seedText
        randomSeed = draft.randomSeed
        turboWeight = draft.turboWeight
        characterWeight = draft.characterWeight
        enableCharacterLora = draft.enableCharacterLora
        UserDefaults.standard.set(modelProfile, forKey: "RIFE60.AIImageStudio.SelectedProfile")
    }

    private func saveGenerationDraft() {
        let draft = AIImageStudioDraft(
            modelProfile: modelProfile,
            positivePrompt: positivePrompt,
            negativePrompt: negativePrompt,
            width: width,
            height: height,
            steps: steps,
            cfg: cfg,
            shift: shift,
            hiresEnabled: hiresEnabled,
            hiresScale: hiresScale,
            hiresSteps: hiresSteps,
            hiresDenoise: hiresDenoise,
            upscalerName: upscalerName,
            clipSkip: clipSkip,
            ensd: ensd,
            tokenMergingRatio: tokenMergingRatio,
            tokenMergingHiresRatio: tokenMergingHiresRatio,
            samplerName: samplerName,
            schedulerName: schedulerName,
            seedText: seedText,
            randomSeed: randomSeed,
            turboWeight: turboWeight,
            characterWeight: characterWeight,
            enableCharacterLora: enableCharacterLora
        )
        if let data = try? JSONEncoder().encode(draft) {
            UserDefaults.standard.set(data, forKey: AIImageStudioDraft.key)
        }
        UserDefaults.standard.set(modelProfile, forKey: "RIFE60.AIImageStudio.SelectedProfile")
        try? settings.save()
        saveGenerationDraft()
    }

    private func saveSettings() {
        do {
            try settings.save()
            saveGenerationDraft()
            showAlert("Local settings saved. Models, prompts, and trained LoRAs stay on this iPhone.")
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
            saveGenerationDraft()
            let resolvedSeed: Int64
            if randomSeed { resolvedSeed = Int64.random(in: 0...Int64.max / 2) }
            else if let value = Int64(seedText), value >= 0 { resolvedSeed = value }
            else { throw AIImageStudioError("Enter a non-negative integer seed or enable Random seed.") }
            generationStatus = "Preparing local GPU/Metal generation…"
            let data = try await OnDeviceImageGenerator.generate(
                settings: settings,
                positivePrompt: positivePrompt,
                negativePrompt: negativePrompt,
                width: Int(width),
                height: Int(height),
                steps: Int(steps),
                cfg: cfg,
                seed: resolvedSeed,
                enableLoRA: enableCharacterLora,
                loraWeight: characterWeight,
                hiresEnabled: hiresEnabled,
                hiresScale: hiresScale,
                hiresSteps: Int(hiresSteps),
                hiresDenoise: hiresDenoise,
                clipSkip: Int(clipSkip),
                progress: { generationStatus = $0 }
            )
            guard let image = UIImage(data: data) else {
                throw AIImageStudioError("The local engine returned data that could not be decoded as an image.")
            }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60_AI_\(UUID().uuidString).png")
            // Save the local PNG for sharing, Photos, and local upscaling.
            try data.write(to: url, options: .atomic)
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
            let stagingDirectory = FileManager.default.temporaryDirectory
                .appendingPathComponent("RIFE60-LoRA-Input-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: stagingDirectory) }

            var images: [LocalTrainingImage] = []
            for (index, item) in trainingItems.enumerated() {
                trainingStatus = "Staging image \(index + 1) of \(trainingItems.count)…"
                guard let source = try await item.loadTransferable(type: Data.self),
                      let image = UIImage(data: source),
                      let png = image.pngData() else {
                    throw AIImageStudioError("Could not decode training image \(index + 1).")
                }
                let filename = "training_\(index + 1).png"
                let sourceURL = stagingDirectory.appendingPathComponent(filename)
                try png.write(to: sourceURL, options: .atomic)
                images.append(LocalTrainingImage(
                    filename: filename,
                    sourceURL: sourceURL,
                    caption: trainingCaption
                ))
                trainerProgress = Double(index + 1) / Double(trainingItems.count) * 0.1
            }

            trainingStatus = "Preparing on-device LoRA training…"
            let result = try await OnDeviceLoRATrainer.train(
                settings: settings,
                images: images,
                triggerWord: triggerWord,
                rank: Int(trainingRank),
                epochs: Int(trainingEpochs),
                learningRate: learningRate,
                progress: { message, fraction in
                    trainingStatus = message
                    if let fraction { trainerProgress = fraction }
                }
            )
            trainedLoRAURL = result
            settings.localLoraFile = result.lastPathComponent
            enableCharacterLora = true
            try settings.save()
            saveGenerationDraft()
            trainerProgress = 1
            trainingStatus = "Training complete locally. The new LoRA is selected for local generation."
        } catch {
            errorText = error.localizedDescription
            trainingStatus = "Local training failed or was stopped by iOS."
        }
    }

}
