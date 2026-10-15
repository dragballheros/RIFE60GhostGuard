import SwiftUI
import PhotosUI
import Photos
import UniformTypeIdentifiers
import UIKit

/// Multiline prompt editor using UIKit's mature TextKit 1 selection path.
/// iOS 26 introduced TextKit 2 multi-range natural selection. These prompts
/// need ordinary single-direction editing, so explicitly avoid that newer
/// selection path and never restore a saved range during an active selection.
private struct StablePromptTextView: UIViewRepresentable {
    @Binding var text: String
    let accessibilityLabel: String

    func makeUIView(context: Context) -> UITextView {
        // Use TextKit 1 intentionally: this editor does not need TextKit 2's
        // multi-range selection features, and this avoids its newer selection path.
        let view = UITextView(usingTextLayoutManager: false)
        view.text = text
        view.delegate = context.coordinator
        view.font = UIFont.preferredFont(forTextStyle: .body)
        view.adjustsFontForContentSizeCategory = true
        view.backgroundColor = .clear
        view.textColor = .label
        view.isEditable = true
        view.isSelectable = true
        view.isScrollEnabled = true
        view.autocapitalizationType = .none
        view.autocorrectionType = .no
        view.smartQuotesType = .no
        view.smartDashesType = .no
        view.textContainerInset = UIEdgeInsets(top: 8, left: 0, bottom: 8, right: 0)
        view.textContainer.lineFragmentPadding = 0
        view.accessibilityLabel = accessibilityLabel
        view.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return view
    }

    func updateUIView(_ view: UITextView, context: Context) {
        let coordinator = context.coordinator
        coordinator.parent = self
        view.accessibilityLabel = accessibilityLabel

        // SwiftUI may publish an external prompt update while UIKit is tracking
        // a selection. Defer it until editing ends rather than replace the
        // backing string underneath active selection indices.
        guard view.text != text else {
            coordinator.pendingExternalText = nil
            return
        }

        if view.isFirstResponder {
            coordinator.pendingExternalText = text
            return
        }

        coordinator.pendingExternalText = nil
        view.text = text
    }

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    final class Coordinator: NSObject, UITextViewDelegate {
        var parent: StablePromptTextView
        var pendingExternalText: String?

        init(parent: StablePromptTextView) { self.parent = parent }

        func textViewDidChange(_ textView: UITextView) {
            // A new user edit wins over any deferred external update.
            pendingExternalText = nil
            let changedText = textView.text ?? ""
            if parent.text != changedText { parent.text = changedText }
        }

        func textViewDidEndEditing(_ textView: UITextView) {
            guard let pendingExternalText else { return }
            self.pendingExternalText = nil
            if textView.text != pendingExternalText {
                // Do not touch selectedRange here. UIKit owns caret/selection
                // normalization when the complete backing text is replaced.
                textView.text = pendingExternalText
            }
        }
    }
}

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
    @State private var modelProfile = "Animagine XL v3.1 (Local)"
    @State private var didRestoreDraft = false
    @State private var showLoRAImporter = false
    @State private var selectedLoRATrigger = ""
    @State private var selectedLoRAStatus = ""

    @State private var positivePrompt = """
masterpiece, best quality, score_7, safe, 1girl, sysdeep_chizuru, beautiful young woman, long brown hair, expressive brown eyes, delicate facial features, fair skin, gentle smile, soft blush, natural proportions, detailed eyes, glossy hair, clean lineart, detailed shading, vibrant colors, soft natural lighting
"""
    @State private var negativePrompt = """
worst quality, low quality, score_1, score_2, artist name, blurry, jpeg artifacts, chromatic aberration, bad anatomy, extra limbs, extra digits, signature, watermark, text, logo
"""
    @State private var width = 832.0
    @State private var height = 1216.0
    @State private var steps = 30.0
    @State private var cfg = 4.0
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
    @State private var samplerName = "DPM++ SDE Karras (closest to Civitai GPU sampler)"
    @State private var schedulerName = "Normal"
    @State private var seedText = "943875898392407"
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
                    VStack(alignment: .leading, spacing: 0) {
                        Text("Studio section")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .padding(.bottom, 6)
                        studioOptionRow("Generate", value: "Generate")
                        studioOptionRow("LoRA Trainer", value: "Train")
                        studioOptionRow("Settings", value: "Settings")
                    }
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
            .onAppear {
                restoreGenerationDraft()
                if let warning = OnDeviceImageGenerator.previousGenerationWarning() {
                    generationStatus = warning
                }
            }
            .onDisappear { saveGenerationDraft(); try? settings.save() }
            .fileImporter(isPresented: $showLoRAImporter, allowedContentTypes: [.data], allowsMultipleSelection: false) { result in
                importSelectedLoRA(result)
            }
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

    private func studioOptionRow(_ title: String, value: String) -> some View {
        Button {
            activeTab = value
        } label: {
            HStack(spacing: 12) {
                Text(title).foregroundStyle(.primary)
                Spacer(minLength: 8)
                Image(systemName: activeTab == value ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(activeTab == value ? Color.accentColor : Color.secondary)
            }
            .contentShape(Rectangle())
            .padding(.vertical, 8)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(activeTab == value ? .isSelected : [])
    }

    private func samplerOptionRow(_ title: String, value: String) -> some View {
        Button {
            samplerName = value
            saveGenerationDraft()
        } label: {
            HStack(spacing: 12) {
                Text(title).foregroundStyle(.primary)
                Spacer(minLength: 8)
                Image(systemName: samplerName == value ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(samplerName == value ? Color.accentColor : Color.secondary)
            }
            .contentShape(Rectangle())
            .padding(.vertical, 8)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(samplerName == value ? .isSelected : [])
    }

    private var generationSections: some View {
        Group {
            Section("Available Models") {
                Text("Choose one installed or supported catalog model. Changing this selection only updates the active model ID; model weights are loaded later when you tap Generate. If loading fails, your current selection remains unchanged.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                modelSelectionRow(
                    title: "Animagine XL v3.1",
                    subtitle: "SDXL anime model",
                    modelID: OnDeviceImageGenerator.defaultModelID
                )
                modelSelectionRow(
                    title: "Anima Base v1.0",
                    subtitle: "Exact base architecture for SysDeep Chizuru Anima v1",
                    modelID: OnDeviceImageGenerator.animaModelID
                )

                LabeledContent("Active model ID", value: settings.localModelID)
                Text(settings.localModelID == OnDeviceImageGenerator.animaModelID
                     ? "Anima Base v1.0 is the exact model family used to train SysDeep Chizuru Anima v1. The character LoRA is separate and must be imported or already installed."
                     : "Animagine XL v3.1 uses SDXL architecture. Use SDXL-compatible LoRAs, not ANIMA-family LoRAs.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !settings.localLoraFile.isEmpty && !selectedLoRATrigger.isEmpty {
                Section("Selected LoRA Trigger") {
                    Text(weightedLoRATrigger)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.disabled)
                        .padding(10)
                        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
                        .accessibilityLabel("LoRA trigger phrase and weight, read only")
                    Text(enableCharacterLora
                         ? "This trigger is automatically included at the start of the positive prompt. Its displayed weight follows LoRA strength."
                         : "The trigger is identified, but the LoRA is currently turned off.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("Positive Prompt") {
                ScrollView {
                    Text(positivePrompt)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .textSelection(.disabled)
                        .padding(10)
                }
                .frame(minHeight: 120, maxHeight: 240)
                .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityLabel("Positive prompt, read only")
                HStack(spacing: 12) {
                    Button {
                        UIPasteboard.general.string = effectivePositivePrompt
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.bordered)
                    Button {
                        guard let clipboardText = UIPasteboard.general.string else { return }
                        positivePrompt = promptBody(fromClipboard: clipboardText)
                    } label: {
                        Label("Paste", systemImage: "clipboard")
                    }
                    .buttonStyle(.bordered)
                }
                Text("Read-only preview. Copy to Notes to edit, then copy your changes and tap Paste to replace this prompt.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Negative Prompt") {
                ScrollView {
                    Text(negativePrompt)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .textSelection(.disabled)
                        .padding(10)
                }
                .frame(minHeight: 96, maxHeight: 200)
                .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityLabel("Negative prompt, read only")
                HStack(spacing: 12) {
                    Button {
                        UIPasteboard.general.string = negativePrompt
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.bordered)
                    Button {
                        guard let clipboardText = UIPasteboard.general.string else { return }
                        negativePrompt = clipboardText
                    } label: {
                        Label("Paste", systemImage: "clipboard")
                    }
                    .buttonStyle(.bordered)
                }
                Text("Read-only preview. Copy to Notes to edit, then copy your changes and tap Paste to replace this prompt.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("On-device Model") {
                LabeledContent("Model catalog ID", value: settings.localModelID)
                Text("Anima Base v1.0 is downloaded from the Draw Things model catalog on first generation, along with required companion components. This is the same base family named by the SysDeep Chizuru Anima v1 creator.")
                    .font(.caption).foregroundStyle(.secondary)
                if settings.localModelID == OnDeviceImageGenerator.animaModelID {
                    Text("ANIMA-family LoRAs must match the selected ANIMA architecture. An SDXL/Animagine character LoRA such as a Chizuru XL LoRA should be used with Animagine instead.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("Local LoRA") {
                Button {
                    showLoRAImporter = true
                } label: {
                    Label("Choose LoRA file (.safetensors)", systemImage: "folder")
                }
                if !settings.localLoraFile.isEmpty {
                    LabeledContent("Selected file", value: settings.localLoraFile)
                    Button("Choose a different LoRA") { showLoRAImporter = true }
                    Button("Clear selected LoRA", role: .destructive) {
                        settings.localLoraFile = ""
                        selectedLoRATrigger = ""
                        enableCharacterLora = false
                        try? settings.save()
                        saveGenerationDraft()
                        selectedLoRAStatus = "No LoRA selected."
                    }
                }
                if settings.localModelID == OnDeviceImageGenerator.animaModelID
                    && (settings.localLoraFile.isEmpty || !settings.localLoraFile.localizedCaseInsensitiveContains("chizuru")) {
                    Link("Open SysDeep Chizuru Anima LoRA page (access approval required)",
                         destination: URL(string: "https://huggingface.co/oxyllina/SysDeep/tree/main")!)
                    Text("Hugging Face requires accepting the repository's access terms before the file can be downloaded. After downloading the exact Anima v1 .safetensors file, import it here; the app will select it by default.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !selectedLoRAStatus.isEmpty {
                    Text(selectedLoRAStatus).font(.caption).foregroundStyle(.secondary)
                }
                Toggle("Apply selected local LoRA", isOn: $enableCharacterLora)
                    .disabled(settings.localLoraFile.isEmpty)
                HStack {
                    Text("LoRA strength")
                    Spacer()
                    Text(String(format: "%.2f", characterWeight)).monospacedDigit()
                }
                Slider(value: $characterWeight, in: 0...1.5, step: 0.05)
                    .disabled(!enableCharacterLora || settings.localLoraFile.isEmpty)
                Text("Selecting a LoRA copies it into the app's Models folder, reads embedded trigger metadata when available, and enables it for generation. The trigger preview above the positive prompt is read-only; its weight follows LoRA strength.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Generation Settings") {
                Text("Sampler selection is passed to the local engine. DPM++ SDE Karras is the closest available engine sampler to the Civitai GPU sampler, but it is not guaranteed to be numerically identical.")
                    .font(.caption).foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 0) {
                    Text("Sampler")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 6)
                    samplerOptionRow("Euler a", value: "Euler a")
                    samplerOptionRow("DPM++ SDE Karras (closest)", value: "DPM++ SDE Karras (closest to Civitai GPU sampler)")
                    samplerOptionRow("DPM++ 2M Karras", value: "DPM++ 2M Karras")
                    samplerOptionRow("DDIM", value: "DDIM")
                    samplerOptionRow("UniPC", value: "UniPC")
                }
                if settings.localModelID == OnDeviceImageGenerator.animaModelID {
                    HStack { Text("Shift"); Spacer(); Text(String(format: "%.1f", shift)).monospacedDigit() }
                    Slider(value: $shift, in: 0.1...12, step: 0.1)
                        .onChange(of: shift) { _ in saveGenerationDraft() }
                    Text("Anima Base metadata uses Shift 3.0. The selected value is passed into the diffusion configuration.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack { Text("Steps"); Spacer(); Text("\(Int(steps))").monospacedDigit() }
                Slider(value: $steps, in: 4...40, step: 1)
                HStack { Text("CFG scale"); Spacer(); Text(String(format: "%.1f", cfg)).monospacedDigit() }
                Slider(value: $cfg, in: 1...10, step: 0.5)
                HStack { Text("Width"); Spacer(); Text("\(Int(width)) px").monospacedDigit() }
                Slider(value: $width, in: 512...1024, step: 64)
                HStack { Text("Height"); Spacer(); Text("\(Int(height)) px").monospacedDigit() }
                Slider(value: $height, in: 512...1536, step: 64)
                TextField("Seed", text: $seedText).keyboardType(.numberPad)
                Text("This engine accepts a 32-bit seed. Larger Civitai seeds are reduced modulo 2³² and will not reproduce the same image exactly.")
                    .font(.caption).foregroundStyle(.secondary)
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
                    Button {
                        Task { await generateImage(isBenchmark: true) }
                    } label: {
                        Label("Run Full Pipeline Benchmark", systemImage: "speedometer")
                    }
                    .buttonStyle(.bordered)
                    Text("Runs a safe 128×128, 1-step on-device smoke test through model resolution/loading, prompt and input encoding, denoising, decoding, and PNG saving. Export diagnostics for stage markers and memory snapshots. This is a diagnostic run, not a full-resolution speed test.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !generationStatus.isEmpty { Text(generationStatus).font(.caption).foregroundStyle(.secondary) }
                if let journalURL = OnDeviceImageGenerator.memoryJournalURL(),
                   FileManager.default.fileExists(atPath: journalURL.path) {
                    ShareLink(item: journalURL) {
                        Label("Export memory diagnostics", systemImage: "doc.text.magnifyingglass")
                    }
                    Text("Shares the latest on-device generation memory trace, including the last recorded stage before a failure.")
                        .font(.caption).foregroundStyle(.secondary)
                }
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
                StablePromptTextView(text: $trainingCaption, accessibilityLabel: "Shared caption or style description")
                    .frame(minHeight: 72, maxHeight: 160)
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
                    .disabled(trainingItems.count < 3 || settings.localModelID != OnDeviceImageGenerator.defaultModelID)
                    if settings.localModelID != OnDeviceImageGenerator.defaultModelID {
                        Text("The current in-app trainer is validated for SDXL/Animagine LoRAs. Switch to Animagine XL v3.1 before training; ANIMA LoRA training is not enabled in this build.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
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
                Text("The picker exposes Animagine XL v3.1 and the exact Anima Base v1.0 family. ANIMA LoRAs and SDXL/Illustrious LoRAs are not interchangeable. The SysDeep Chizuru Anima v1 file is hosted in a gated Hugging Face repository, so this app cannot silently fetch it without the repository owner's access terms being accepted.")
                    .font(.caption).foregroundStyle(.secondary)
                Link("Embedded local engine source and license", destination: URL(string: "https://github.com/drawthingsai/draw-things-community/tree/4f288803ac898525012c0fe8d998c85bcb920b70")!)
            }
            Section {
                Button("Reset model to Animagine XL v3.1 (8-bit)") {
                    applyModelProfile(OnDeviceImageGenerator.defaultModelID)
                    try? settings.save()
                    saveGenerationDraft()
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

    @ViewBuilder
    private func modelSelectionRow(title: String, subtitle: String, modelID: String) -> some View {
        let isSelected = settings.localModelID == modelID
        Button {
            selectModelSafely(modelID)
        } label: {
            HStack(spacing: 12) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .foregroundStyle(.primary)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                if isSelected {
                    Text("Selected")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Color.accentColor)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private func selectModelSafely(_ modelID: String) {
        guard !isGenerating, !isTraining else {
            errorText = "Wait for the current AI task to finish before changing models."
            return
        }
        guard modelID == OnDeviceImageGenerator.defaultModelID
                || modelID == OnDeviceImageGenerator.animaModelID else {
            errorText = "That model is not in the supported local model catalog."
            return
        }
        guard settings.localModelID != modelID else { return }

        // Save the old selection first. The selection action must never load model
        // weights, construct a pipeline, or touch Metal. Commit the new catalog ID
        // only after its settings snapshot has been persisted.
        let previousSettings = settings
        settings.localModelID = modelID
        do {
            try settings.save()
            applyModelProfile(modelID)
            saveGenerationDraft()
            errorText = ""
            generationStatus = "Selected \(modelProfile). Model weights will load only when Generate is tapped."
        } catch {
            settings = previousSettings
            try? previousSettings.save()
            errorText = "Could not save the model selection. The previous model remains selected. " + error.localizedDescription
        }
    }

    private func applyModelProfile(_ profile: String) {
        let useAnima = profile == OnDeviceImageGenerator.animaModelID
            || profile.localizedCaseInsensitiveContains("Anima Base")
        settings.localModelID = useAnima
            ? OnDeviceImageGenerator.animaModelID
            : OnDeviceImageGenerator.defaultModelID
        modelProfile = useAnima ? "Anima Base v1.0 (Local)" : "Animagine XL v3.1 (Local)"
        UserDefaults.standard.set(modelProfile, forKey: "RIFE60.AIImageStudio.SelectedProfile")
        errorText = ""
        generationStatus = ""
        generatedImage = nil
        generatedURL = nil
        width = useAnima ? 832 : 832
        height = useAnima ? 1216 : 1216
        steps = useAnima ? 30 : 28
        cfg = useAnima ? 4.0 : 5.5
        shift = useAnima ? 3.0 : 1.0
        hiresEnabled = false
        hiresScale = 1.5
        hiresSteps = 12
        hiresDenoise = 0.35
        clipSkip = useAnima ? 1 : 2
        samplerName = useAnima
            ? "DPM++ SDE Karras (closest to Civitai GPU sampler)"
            : "Euler a"
        schedulerName = "Model recommended"
        if useAnima {
            seedText = "943875898392407"
            randomSeed = false
            if !positivePrompt.localizedCaseInsensitiveContains("sysdeep_chizuru") {
                positivePrompt = "masterpiece, best quality, score_7, safe, sysdeep_chizuru, " + positivePrompt
            }
        }
    }

    private func restoreGenerationDraft() {
        guard !didRestoreDraft else { return }
        didRestoreDraft = true

        // Migrate users from the old Aesthetic model selection to the actual Base
        // checkpoint required by the SysDeep Chizuru Anima v1 LoRA.
        if settings.localModelID == "anima_aesthetic_1.1_q8p.ckpt" {
            settings.localModelID = OnDeviceImageGenerator.animaModelID
            try? settings.save()
        }
        // Preserve user-imported character LoRAs, including Chizuru models.
        // Remove obsolete markup, never the character's plain trigger text.
        if settings.localModelID != OnDeviceImageGenerator.defaultModelID
            && settings.localModelID != OnDeviceImageGenerator.animaModelID {
            settings.localModelID = OnDeviceImageGenerator.defaultModelID
        }
        modelProfile = settings.localModelID == OnDeviceImageGenerator.animaModelID
            ? "Anima Base v1.0 (Local)"
            : "Animagine XL v3.1 (Local)"
        positivePrompt = removingLegacyCharacterTrigger(from: positivePrompt)
        if let directory = try? OnDeviceImageGenerator.modelsDirectoryURL() {
            if settings.localModelID == OnDeviceImageGenerator.animaModelID,
               settings.localLoraFile.isEmpty {
                let exactName = "Sysdeep Chizuru - Anima v1.safetensors"
                if FileManager.default.fileExists(atPath: directory.appendingPathComponent(exactName).path) {
                    settings.localLoraFile = exactName
                    enableCharacterLora = true
                    characterWeight = 0.7
                    selectedLoRATrigger = "sysdeep_chizuru"
                    selectedLoRAStatus = "Found the compatible Chizuru Anima LoRA already installed; selected it automatically."
                    try? settings.save()
                }
            }
            if !settings.localLoraFile.isEmpty {
                let loraURL = directory.appendingPathComponent(settings.localLoraFile)
                if FileManager.default.fileExists(atPath: loraURL.path) {
                    let fallback = loraURL.deletingPathExtension().lastPathComponent
                        .replacingOccurrences(of: "_", with: " ")
                        .replacingOccurrences(of: "-", with: " ")
                    let detected = extractLoRATrigger(from: loraURL)?.trigger
                    let isChizuruAnima = settings.localModelID == OnDeviceImageGenerator.animaModelID
                        && loraURL.lastPathComponent.localizedCaseInsensitiveContains("chizuru")
                        && loraURL.lastPathComponent.localizedCaseInsensitiveContains("anima")
                    selectedLoRATrigger = detected ?? (isChizuruAnima ? "sysdeep_chizuru" : fallback)
                    if isChizuruAnima { enableCharacterLora = true; characterWeight = 0.7 }
                }
            }
        }

        guard let data = UserDefaults.standard.data(forKey: AIImageStudioDraft.key),
              let draft = try? JSONDecoder().decode(AIImageStudioDraft.self, from: data) else {
            applyModelProfile(settings.localModelID)
            positivePrompt = removingLegacyCharacterTrigger(from: positivePrompt)
            saveGenerationDraft()
            try? settings.save()
            return
        }
        modelProfile = settings.localModelID == OnDeviceImageGenerator.animaModelID
            ? "Anima Base v1.0 (Local)"
            : "Animagine XL v3.1 (Local)"
        positivePrompt = removingLegacyCharacterTrigger(from: draft.positivePrompt)
        negativePrompt = draft.negativePrompt
        // Keep conservative defaults for each native model family.
        let useAnima = settings.localModelID == OnDeviceImageGenerator.animaModelID
        if useAnima && !positivePrompt.localizedCaseInsensitiveContains("sysdeep_chizuru") {
            positivePrompt = "masterpiece, best quality, score_7, safe, sysdeep_chizuru, " + positivePrompt
        }
        width = useAnima ? 832 : 832
        height = useAnima ? 1216 : 1216
        steps = useAnima ? 30 : 28
        cfg = useAnima ? 4.0 : 5.5
        shift = useAnima ? 3.0 : 1.0
        hiresEnabled = false
        hiresScale = 1.5
        hiresSteps = 12
        hiresDenoise = 0.35
        upscalerName = draft.upscalerName
        clipSkip = 2
        ensd = draft.ensd
        tokenMergingRatio = draft.tokenMergingRatio
        tokenMergingHiresRatio = draft.tokenMergingHiresRatio
        samplerName = useAnima
            ? "DPM++ SDE Karras (closest to Civitai GPU sampler)"
            : (draft.samplerName == "Model recommended" ? "Euler a" : draft.samplerName)
        schedulerName = "Model recommended"
        seedText = useAnima ? "943875898392407" : draft.seedText
        randomSeed = draft.randomSeed
        turboWeight = draft.turboWeight
        characterWeight = draft.characterWeight
        enableCharacterLora = useAnima && selectedLoRATrigger == "sysdeep_chizuru"
            ? true
            : (draft.enableCharacterLora && !settings.localLoraFile.isEmpty)
        UserDefaults.standard.set(modelProfile, forKey: "RIFE60.AIImageStudio.SelectedProfile")
    }

    private func removingLegacyCharacterTrigger(from prompt: String) -> String {
        var cleaned = prompt
        let legacyPatterns = [
            #"(?i)<lora:\s*ichinose[_ -]*chizuru\s*:\s*[-+]?\d*\.?\d+>"#,
            #"(?i)\bmayuzumi\s+fuyuko\b"#
        ]
        for pattern in legacyPatterns {
            cleaned = cleaned.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }
        cleaned = cleaned.replacingOccurrences(of: #",\s*,+"#, with: ",", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: #"^\s*,+|,+\s*$"#, with: "", options: .regularExpression)
        cleaned = cleaned.replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func saveGenerationDraft() {
        positivePrompt = removingLegacyCharacterTrigger(from: positivePrompt)
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
    }

    private var weightedLoRATrigger: String {
        let trigger = selectedLoRATrigger.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trigger.isEmpty else { return "" }
        return "(\(trigger):\(String(format: "%.2f", characterWeight)))"
    }

    /// Keep the trigger as a separate read-only visual row while composing it
    /// into the positive prompt sent to the local generation engine.
    private var effectivePositivePrompt: String {
        let body = promptBodyWithoutSelectedTrigger(positivePrompt)
        guard enableCharacterLora,
              !settings.localLoraFile.isEmpty,
              !selectedLoRATrigger.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return body
        }
        return body.isEmpty ? weightedLoRATrigger : weightedLoRATrigger + ", " + body
    }

    private func promptBodyWithoutSelectedTrigger(_ prompt: String) -> String {
        var body = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let trigger = selectedLoRATrigger.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trigger.isEmpty else { return body }
        let escaped = NSRegularExpression.escapedPattern(for: trigger)
        body = body.replacingOccurrences(
            of: #"(?i)^\s*\(\s*"# + escaped + #":\s*[-+]?\d*\.?\d+\s*\)\s*,?\s*"#,
            with: "", options: .regularExpression
        )
        body = body.replacingOccurrences(of: #"(?i)"# + escaped, with: "", options: .regularExpression)
        body = body.replacingOccurrences(of: #",\s*,+"#, with: ",", options: .regularExpression)
        body = body.replacingOccurrences(of: #"^\s*,+|,+\s*$"#, with: "", options: .regularExpression)
        return body.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func promptBody(fromClipboard clipboard: String) -> String {
        var body = clipboard.trimmingCharacters(in: .whitespacesAndNewlines)
        let weighted = weightedLoRATrigger
        if !weighted.isEmpty && body.hasPrefix(weighted) {
            body = String(body.dropFirst(weighted.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: " ,\n\t"))
        }
        return promptBodyWithoutSelectedTrigger(body)
    }

    /// Read only the small JSON header of a safetensors file, not its tensor data.
    private func extractLoRATrigger(from url: URL) -> (trigger: String, source: String)? {
        guard url.pathExtension.lowercased() == "safetensors",
              let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        do {
            let prefix = try handle.read(upToCount: 8) ?? Data()
            guard prefix.count == 8 else { return nil }
            let headerLength = prefix.enumerated().reduce(UInt64(0)) { partial, item in
                partial | (UInt64(item.element) << (item.offset * 8))
            }
            guard headerLength > 0, headerLength <= 16 * 1024 * 1024 else { return nil }
            let header = try handle.read(upToCount: Int(headerLength)) ?? Data()
            guard header.count == Int(headerLength),
                  let root = try JSONSerialization.jsonObject(with: header) as? [String: Any],
                  let metadata = root["__metadata__"] as? [String: Any] else { return nil }

            let keys = [
                "modelspec.trigger_phrase", "ss_trigger_words", "ss_trigger_word",
                "trigger_words", "trigger_word", "trigger", "trained_words"
            ]
            for key in keys {
                if let raw = metadata[key] as? String, let value = normalizedTrigger(raw) {
                    return (value, "embedded \(key) metadata")
                }
            }
            for (key, rawValue) in metadata where key.localizedCaseInsensitiveContains("trigger") {
                if let raw = rawValue as? String, let value = normalizedTrigger(raw) {
                    return (value, "embedded \(key) metadata")
                }
            }
            if let rawTags = metadata["modelspec.tags"] as? String {
                let firstTag = rawTags.split(whereSeparator: { $0 == "," || $0 == "\n" })
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .first(where: { !$0.isEmpty })
                if let firstTag, firstTag.count <= 120 {
                    return (firstTag, "embedded model tags")
                }
            }
        } catch {
            DiagnosticsLogger.shared.log("LoRA trigger metadata could not be read: \(error.localizedDescription)")
        }
        return nil
    }

    private func normalizedTrigger(_ raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let data = trimmed.data(using: .utf8),
           let array = try? JSONSerialization.jsonObject(with: data) as? [String] {
            let value = array.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: ", ")
            return value.isEmpty ? nil : value
        }
        let firstLine = trimmed.components(separatedBy: .newlines).first?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? trimmed
        return firstLine.isEmpty ? nil : firstLine
    }

    private func importSelectedLoRA(_ result: Result<[URL], Error>) {
        do {
            let urls = try result.get()
            guard let sourceURL = urls.first else { return }
            let hasAccess = sourceURL.startAccessingSecurityScopedResource()
            defer { if hasAccess { sourceURL.stopAccessingSecurityScopedResource() } }
            let ext = sourceURL.pathExtension.lowercased()
            guard ["safetensors", "ckpt", "pt", "bin"].contains(ext) else {
                throw AIImageStudioError("Choose a LoRA checkpoint file (.safetensors, .ckpt, .pt, or .bin).")
            }
            let directory = try OnDeviceImageGenerator.modelsDirectoryURL()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let safeName = sourceURL.lastPathComponent
            let destinationURL = directory.appendingPathComponent(safeName)
            if sourceURL.standardizedFileURL != destinationURL.standardizedFileURL {
                if FileManager.default.fileExists(atPath: destinationURL.path) {
                    try FileManager.default.removeItem(at: destinationURL)
                }
                try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            }
            settings.localLoraFile = safeName
            enableCharacterLora = true
            let stem = sourceURL.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "_", with: " ")
                .replacingOccurrences(of: "-", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let extracted = extractLoRATrigger(from: destinationURL)
            selectedLoRATrigger = extracted?.trigger ?? stem
            try settings.save()
            saveGenerationDraft()
            if let extracted {
                selectedLoRAStatus = "Imported and selected: " + safeName + ". Trigger auto-detected from " + extracted.source + "."
            } else {
                selectedLoRAStatus = "Imported and selected: " + safeName + ". No trigger metadata was embedded; using the filename as a fallback guess."
            }
            errorText = ""
        } catch {
            errorText = "Could not import LoRA: " + error.localizedDescription
            selectedLoRAStatus = "Import failed."
        }
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

    private func generateImage(isBenchmark: Bool = false) async {
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
            let generationPrompt = isBenchmark
                ? "anime landscape, simple composition, clean colors"
                : removingLegacyCharacterTrigger(from: effectivePositivePrompt)
            let generationNegativePrompt = isBenchmark ? "blurry, distorted, low quality" : negativePrompt
            generationStatus = isBenchmark ? "Benchmark: preparing full on-device pipeline…" : "Preparing local GPU/Metal generation…"
            let url = try await OnDeviceImageGenerator.generate(
                settings: settings,
                positivePrompt: generationPrompt,
                negativePrompt: generationNegativePrompt,
                width: isBenchmark ? 128 : Int(width),
                height: isBenchmark ? 128 : Int(height),
                steps: isBenchmark ? 1 : Int(steps),
                cfg: cfg,
                seed: resolvedSeed,
                samplerName: samplerName,
                shift: shift,
                enableLoRA: enableCharacterLora,
                loraWeight: characterWeight,
                hiresEnabled: hiresEnabled,
                hiresScale: hiresScale,
                hiresSteps: Int(hiresSteps),
                hiresDenoise: hiresDenoise,
                clipSkip: Int(clipSkip),
                benchmarkMode: isBenchmark,
                progress: { generationStatus = $0 }
            )
            guard let image = UIImage(contentsOfFile: url.path) else {
                try? FileManager.default.removeItem(at: url)
                throw AIImageStudioError("The local engine saved a PNG that could not be decoded as an image.")
            }
            generatedImage = image
            generatedURL = url
            generationStatus = isBenchmark
                ? "Benchmark complete • 128×128 • 1 step • export diagnostics for stage-by-stage results"
                : "Generation complete • \(image.size.width.rounded()) × \(image.size.height.rounded())"
        } catch {
            OnDeviceImageGenerator.recordGenerationFailure()
            errorText = error.localizedDescription
            generationStatus = isBenchmark ? "Benchmark failed • export diagnostics for the last stage" : "Generation failed"
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
