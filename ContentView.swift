import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import UIKit
import AVKit
import QuickLook

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage("automaticWarmthCorrectionMode") private var automaticWarmthCorrectionModeRawValue = "full"
    @AppStorage("colorPopSkinToneCorrectionMode") private var colorPopSkinToneCorrectionModeRawValue = "processedOnly"
    @StateObject private var vm = VideoProcessorViewModel()
    @StateObject private var pipController = ProcessingPiPController()
    @State private var showingImporter = false
    @State private var showingAIImageStudio = false
    @State private var showingH3VideoEditor = false
    @State private var showingWatermarkEditor = false
    @State private var showingPhotosPicker = false
    @State private var showingPreview = false
    @State private var showingExportFolderPicker = false
    @State private var showingClearRecoveryConfirmation = false
    @State private var photoItems: [PhotosPickerItem] = []
    @State private var showingMaskCopyConfirmation = false
    @State private var lastCompletedVideoURL: URL?
    @State private var previewPlayer: AVPlayer?
    @State private var quickLookURL: URL?
    @State private var postRenderSleepTask: Task<Void, Never>?
    @State private var postRenderScreenDimmed = false
    @State private var postRenderSavedBrightness: CGFloat?
    @State private var postRenderSavedIdleTimerDisabled: Bool?

    var body: some View {
        ZStack {
            ProcessingPiPSourceRepresentable(controller: pipController)
                .ignoresSafeArea()
                .allowsHitTesting(false)
                .accessibilityHidden(true)

            NavigationStack {
                Form {
                    Section("Input") {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack(spacing: 12) {
                                Image(systemName: "wand.and.stars")
                                    .font(.system(size: 24, weight: .semibold))
                                    .foregroundStyle(.cyan)
                                    .frame(width: 48, height: 48)
                                    .background(.cyan.opacity(0.13), in: RoundedRectangle(cornerRadius: 14))
                                VStack(alignment: .leading, spacing: 3) {
                                    Text("Processing Recipe").font(.title3.weight(.bold))
                                    Text(vm.inputURL?.lastPathComponent ?? "Choose a video, GIF, or image")
                                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                }
                                Spacer(minLength: 0)
                            }
                            HStack(spacing: 8) {
                                recipeChip(vm.inputKind == .image ? "Still image" : vm.inputKind == .gif ? "Animated GIF" : "Video", symbol: vm.inputKind == .image ? "photo" : vm.inputKind == .gif ? "livephoto" : "film")
                                recipeChip(vm.inputKind == .image ? (vm.upscaleTo4K ? "2× upscale" : "Enhance") : "HQ RIFE", symbol: "sparkles")
                                if vm.inputKind != .image { recipeChip("60 FPS", symbol: "speedometer") }
                            }
                            Toggle("Real-CUGAN 2× upscale", isOn: $vm.upscaleTo4K)
                                .disabled(vm.queueLocked || vm.isImporting || vm.isBusy || vm.isBenchmarking)
                        }
                        .padding(.vertical, 6)
                        Divider().padding(.vertical, 4)
                        Button { showingPhotosPicker = true } label: { Label("Select media from Photos", systemImage: "photo.on.rectangle") }.disabled(vm.isBusy || vm.isImporting || vm.isBenchmarking)
                        Button { showingImporter = true } label: { Label("Select media from Files", systemImage: "folder") }.disabled(vm.isBusy || vm.isImporting || vm.isBenchmarking)
                        if vm.isImporting {
                            VStack(alignment: .leading, spacing: 6) {
                                if let importProgress = vm.importProgress { ProgressView(value: importProgress) } else { ProgressView() }
                                Text("Importing…").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        if let input = vm.inputURL {
                            Text(input.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                            if vm.inputKind == .image {
                                Text("Image • RIFE interpolation is skipped").font(.caption).foregroundStyle(.secondary)
                            } else if vm.inputKind == .gif {
                                Text("GIF • converted to video → full RIFE/CUGAN pipeline → optimized GIF")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        if vm.inputURL != nil && (vm.inputKind == .video || vm.inputKind == .gif) {
                            HStack {
                                Button {
                                    showingPreview = true
                                    vm.generatePreview()
                                } label: {
                                    Label(vm.isPreviewing ? "Cancel Preview" : "Preview Opening Frames", systemImage: vm.isPreviewing ? "xmark.circle" : "play.rectangle")
                                }
                                // Intentionally remains enabled during preview.
                                .disabled(vm.inputURL == nil || vm.isBusy || vm.isImporting || vm.isBenchmarking)
                                Spacer()
                                if vm.isPreviewing {
                                    ProgressView()
                                }
                            }
                            if !vm.previewStatus.isEmpty {
                                Text(vm.previewStatus).font(.caption).foregroundStyle(.secondary)
                            }
                            Text("Previews up to three opening frames with current still-frame enhancement settings. RIFE motion interpolation is not represented; Process remains available.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }


                    Section("MiniMax H3") {
                        Button { showingH3VideoEditor = true } label: {
                            Label("Open Local MiniMax H3", systemImage: "wand.and.stars")
                        }
                        .disabled(vm.isBusy || vm.isImporting || vm.isBenchmarking)
                        Text("Reference-conditioned video on this iPhone. Tier 1 keeps about 44 GB of published compressed weights and does not use a paid cloud.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Section("AI Image Studio") {
                        Button { showingAIImageStudio = true } label: {
                            Label("Open AI Image Studio", systemImage: "sparkles")
                        }
                        .disabled(vm.isBusy || vm.isImporting || vm.isBenchmarking)
                        Text("Generate anime images locally on the iPhone. To protect memory, AI Image Studio cannot open while video processing, importing, or model benchmarking is active. Generated stills can then go through this app's Real-CUGAN upscale and final Sharpie pipeline.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }


                    mediaQueueSection

                    if vm.recoveryAvailable {
                        Section("Crash Recovery") {
                            Label("Recovery data found", systemImage: "arrow.clockwise.circle.fill").foregroundStyle(.orange)
                            Text(vm.recoveryStatusText).font(.caption)
                            if !vm.isBusy {
                                Button { Task { await vm.start() } } label: { Label(vm.upscaleTo4K ? "Resume 2× 60 FPS Render" : "Resume 60 FPS Render", systemImage: "play.fill") }
                                Button(role: .destructive) { showingClearRecoveryConfirmation = true } label: { Label("Clear Recovery Data", systemImage: "trash") }
                            }
                            Text("The batch and its masks are saved. Completed AI passes are kept in persistent storage. If iOS terminates the app, reopening it reuses every completed checkpoint instead of starting the whole render over.").font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    Section("RIFE 4.26") {
                        if vm.inputKind == .image {
                            Label("Skipped for images", systemImage: "photo")
                            Text("A single image has no neighbouring frame to interpolate, so images go straight through Compression Guard → Real-CUGAN → Final Sharpie.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else if vm.inputKind == .gif {
                            Label("Full video pipeline", systemImage: "film.stack")
                            Text("The GIF is converted to a temporary video, then runs Compression Guard → RIFE HQ → Real-CUGAN → Final Sharpie. The processed video is converted back to an optimized GIF at the end.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            LabeledContent("Interpolation quality", value: "High Quality (HQ)")
                            Toggle("Ghost protection", isOn: $vm.ghostProtection).disabled(vm.queueLocked || vm.isImporting)
                            Toggle("Scene-cut protection", isOn: $vm.sceneCutProtection).disabled(vm.queueLocked || vm.isImporting)
                            LabeledContent("Target", value: "60.00 fps")
                            Text("1080p-class video uses one persistent full-frame HQ stream for maximum speed without dropping RIFE quality.").font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    Section("Anime Watermark Removal") {
                        Toggle("Remove marked watermarks", isOn: $vm.watermarkRemovalEnabled).disabled(vm.queueLocked || vm.isImporting)
                        if vm.watermarkRemovalEnabled {
                            VStack(alignment: .leading, spacing: 0) {
                                Text("Inpainting model")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                    .padding(.bottom, 6)
                                ForEach(WatermarkModel.allCases) { model in
                                    Button {
                                        guard !vm.queueLocked && !vm.isImporting else { return }
                                        vm.watermarkModel = model
                                    } label: {
                                        HStack(spacing: 12) {
                                            Text(model.title).foregroundStyle(.primary)
                                            Spacer(minLength: 8)
                                            Image(systemName: vm.watermarkModel == model ? "checkmark.circle.fill" : "circle")
                                                .foregroundStyle(vm.watermarkModel == model ? Color.accentColor : Color.secondary)
                                        }
                                        .contentShape(Rectangle())
                                        .padding(.vertical, 8)
                                    }
                                    .buttonStyle(.plain)
                                    .disabled(vm.queueLocked || vm.isImporting)
                                    .accessibilityAddTraits(vm.watermarkModel == model ? .isSelected : [])
                                }
                            }
                            if let item = vm.selectedQueueItem, (vm.mediaQueue?.items.count ?? 0) > 1 {
                                Text("Mask for: \(item.displayName)").font(.caption)
                                Toggle("Remove watermark on this item", isOn: Binding(get: { vm.selectedQueueItem?.removeWatermark ?? true }, set: { vm.setSelectedWatermarkRemoval($0) })).disabled(vm.queueLocked || vm.isImporting)
                                Button("Copy this mask to other media…") { showingMaskCopyConfirmation = true }
                                    .disabled(vm.queueLocked || vm.watermarkRegions.isEmpty)
                            }
                            Button { showingWatermarkEditor = true } label: {
                                Label(vm.watermarkRegions.isEmpty ? "Paint watermark mask" : "Edit watermark mask", systemImage: "paintbrush.pointed")
                            }.disabled(vm.inputURL == nil || vm.queueLocked)
                            HStack { Text("Mask padding"); Spacer(); Text("\(Int(vm.watermarkPaddingPixels)) px") }
                            Slider(value: $vm.watermarkPaddingPixels, in: 0...16, step: 1).disabled(vm.queueLocked || vm.isImporting)
                            Text(vm.watermarkModel == .lama
                                 ? "Anime/Manga LaMa reconstructs marked areas before RIFE and upscaling. Painted masks apply throughout a video. Use the brush and eraser to preserve the artwork between letters."
                                 : "DeepCream v2 runs locally on-device using its dedicated bar-censor or mosaic-censor weights. Mark only the area to reconstruct.").font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    Section("Anime Processing") {
                        Toggle("Compression Guard", isOn: $vm.compressionProtection).disabled(vm.queueLocked || vm.isImporting)
                        Toggle("Color Rebuild", isOn: $vm.colorRebuildEnabled).disabled(vm.queueLocked || vm.isImporting)
                        if vm.colorRebuildEnabled {
                            HStack { Text("Color Rebuild strength"); Spacer(); Text(String(format: "%.2f", vm.colorRebuildStrength)) }
                            Slider(value: $vm.colorRebuildStrength, in: 0.1...1.0).disabled(vm.queueLocked || vm.isImporting)
                            Text("Luma-guided chroma rebuild after Real-CUGAN and before Sharpie. Flat cel colors are consolidated and speckle is removed. Line art, skin, and saturation stay.").font(.caption).foregroundStyle(.secondary)
                        }
                        Toggle("Outline Rebuild", isOn: $vm.outlineRebuildEnabled).disabled(vm.queueLocked || vm.isImporting)
                        if vm.outlineRebuildEnabled {
                            HStack { Text("Outline Rebuild strength"); Spacer(); Text(String(format: "%.2f", vm.outlineRebuildStrength)) }
                            Slider(value: $vm.outlineRebuildStrength, in: 0.1...1.0).disabled(vm.queueLocked || vm.isImporting)
                            Text("Closes short gaps, restores faded line darkness, and lifts stray specks before Sharpie. Sharpie itself is unchanged.").font(.caption).foregroundStyle(.secondary)
                        }
                        Toggle("Final Sharpie Outline", isOn: $vm.outlineProtection).disabled(vm.queueLocked || vm.isImporting)
                        Toggle("Color Pop", isOn: $vm.colorPopEnabled).disabled(vm.queueLocked || vm.isImporting)
                        Picker("Automatic warmth", selection: $automaticWarmthCorrectionModeRawValue) {
                            ForEach(AutoWarmthCorrectionMode.allCases) { mode in
                                Text(mode.title).tag(mode.rawValue)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(vm.queueLocked || vm.isImporting)
                        if let mode = AutoWarmthCorrectionMode(rawValue: automaticWarmthCorrectionModeRawValue) {
                            Text(mode.explanation)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Picker("Color Pop skin tone", selection: $colorPopSkinToneCorrectionModeRawValue) {
                            ForEach(ColorPopSkinToneMode.allCases) { mode in
                                Text(mode.title).tag(mode.rawValue)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(vm.queueLocked || vm.isImporting)
                        if let mode = ColorPopSkinToneMode(rawValue: colorPopSkinToneCorrectionModeRawValue) {
                            Text(mode.explanation)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if vm.colorPopEnabled {
                            HStack { Text("Color Pop strength"); Spacer(); Text(String(format: "%.2f", vm.colorPopStrength)) }
                            Slider(value: $vm.colorPopStrength, in: 0.1...1.0).disabled(vm.queueLocked || vm.isImporting)
                            Text("Selective anime color pop after enhancement: richer reds, pinks, blues and teals, with luma-preserving chroma, neutral-gray protection, and skin/highlight safeguards.").font(.caption).foregroundStyle(.secondary)
                        }
                        Text("Compression Guard runs before RIFE. Sharpie now runs as the final visual pass after Real-CUGAN, so CUGAN cannot soften or change the finished outlines. This revision is slightly narrower and closer to the original anime line width while remaining a little thicker/sharper than the original.").font(.caption).foregroundStyle(.secondary)
                    }

                    Section("2× Anime Upscale") {
                        Toggle("Real-CUGAN Native 2×", isOn: $vm.upscaleTo4K).disabled(vm.queueLocked || vm.isImporting)
                        VStack(alignment: .leading, spacing: 0) {
                            Text("Model compute units")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .padding(.bottom, 6)
                            ForEach(ModelComputePreference.allCases) { option in
                                Button {
                                    guard !vm.queueLocked && !vm.isBusy && !vm.isBenchmarking else { return }
                                    vm.computePreference = option
                                } label: {
                                    HStack(spacing: 12) {
                                        Text(option.title).foregroundStyle(.primary)
                                        Spacer(minLength: 8)
                                        Image(systemName: vm.computePreference == option ? "checkmark.circle.fill" : "circle")
                                            .foregroundStyle(vm.computePreference == option ? Color.accentColor : Color.secondary)
                                    }
                                    .contentShape(Rectangle())
                                    .padding(.vertical, 8)
                                }
                                .buttonStyle(.plain)
                                .disabled(vm.queueLocked || vm.isBusy || vm.isBenchmarking)
                                .accessibilityAddTraits(vm.computePreference == option ? .isSelected : [])
                            }
                        }
                        Text("Auto preserves the current Core ML policy. GPU may use more memory. GPU and Neural Engine modes both allow CPU fallback; benchmark first.").font(.caption).foregroundStyle(.secondary)
                        if vm.inputKind == .video {
                            Toggle("Upscale before interpolation (faster)", isOn: $vm.upscaleFirst)
                                .disabled(vm.queueLocked || vm.isBusy || vm.isBenchmarking || !vm.upscaleTo4K)
                            Text("Upscales original frames, then runs RIFE at 2× resolution. The look changes slightly and 4K RIFE is heavier. RIFE reduces band size under memory pressure; total speedup is not guaranteed.").font(.caption).foregroundStyle(.secondary)
                        }
                        LabeledContent("Model", value: "Real-CUGAN – Anime")
                        LabeledContent("Upscale", value: "2× source resolution")
                        LabeledContent("Noise", value: "Level 3")
                        LabeledContent("Intensity", value: "1.30")
                        Text("Runs after RIFE and before the final Sharpie pass. Real-CUGAN processes overlapping tiles and stitches them back to exactly 2× the original source dimensions without converting the source to 1080p first.").font(.caption).foregroundStyle(.secondary)
                    }

                    if vm.inputKind == .video {
                        Section("Ghost Guard") {
                            VStack(alignment: .leading, spacing: 5) {
                                HStack { Text("Sensitivity"); Spacer(); Text(String(format: "%.2f", vm.ghostSensitivity)) }
                                Slider(value: $vm.ghostSensitivity, in: 0.65...1.35, step: 0.05).disabled(vm.queueLocked || vm.isImporting)
                            }
                        }
                    }

                    Section("Render Power") {
                        Toggle("Render power mode", isOn: $vm.renderPowerMode)
                        Text("The processing screen turns completely black immediately. Tap once to wake it; after 20 seconds without a touch it returns to black. While processing, Picture in Picture is armed in the foreground so swiping Home can transition directly into a live progress card without suspending recovery/render work. Returning to the app closes PiP. When a render finishes, the screen gets a 1-minute grace period; if you do not touch the app, it goes black and iOS auto-lock is re-enabled.").font(.caption).foregroundStyle(.secondary)
                    }

                    Section("Export to Files") {
                        if vm.inputKind == .video {
                            Toggle("Convert video to GIF (≤20 MB)", isOn: $vm.videoToGIFEnabled)
                                .disabled(vm.queueLocked || vm.isImporting)
                            Text("When enabled, the normal full-quality video pipeline finishes first, then the completed master is converted to an animated GIF optimized to fit the 20 MB delivery limit. Audio is omitted because GIF has no audio track.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            if vm.videoToGIFEnabled {
                                LabeledContent("GIF resolution / frame-rate cap", value: "Up to 2560 px long edge / 60 FPS")
                            }
                        } else if vm.inputKind == .image {
                            Toggle("Reddit Mode (≤20 MB)", isOn: $vm.redditMode)
                                .disabled(vm.queueLocked || vm.isImporting)
                            Text("When enabled, still images are prepared for Reddit's 20 MB limit with the highest-quality PNG/JPEG option that fits.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            LabeledContent("Delivery format", value: "Animated GIF ≤20 MB • full video pipeline")
                            Text("GIF input is always converted to video, fully processed, then encoded back to GIF. The original GIF is never passed through unchanged.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        if vm.videoToGIFEnabled || vm.inputKind == .gif {
                            Text("GIF export tries the largest processed frame size first and only reduces resolution or FPS when needed to meet the 20 MB target.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        if vm.inputKind == .image {
                            LabeledContent("Delivery format", value: vm.redditMode ? "PNG or high-quality JPEG ≤20 MB" : "PNG (lossless)")
                        } else if vm.inputKind == .gif || vm.videoToGIFEnabled {
                            LabeledContent("Delivery format", value: "Animated GIF ≤20 MB")
                        } else {
                            LabeledContent("Final codec", value: "HEVC Main10")
                            LabeledContent("Pixel format", value: "10-bit P010")
                            LabeledContent("File-size target", value: "< 1 GB")
                            Toggle("Preserve original audio", isOn: $vm.preserveAudio).disabled(vm.queueLocked || vm.isImporting)
                        }
                        LabeledContent("Auto-save location", value: vm.exportFolderName)
                        Button { showingExportFolderPicker = true } label: { Label("Set On My iPhone Save Location", systemImage: "folder.badge.plus") }
                        if let last = lastCompletedVideoURL {
                            ShareLink(item: last) { Label("Export Last Completed Video", systemImage: "square.and.arrow.up") }
                            Text("Last completed: \(last.lastPathComponent)").font(.caption).foregroundStyle(.secondary)
                        }
                        Text("For videos to appear directly in On My iPhone instead of the app's private-looking folder, tap Set On My iPhone Save Location, select the On My iPhone folder, and tap Open once. iOS requires this one-time folder permission. The app remembers it and future completed videos are written there directly. Export Last Completed Video remains available after relaunch so a finished render is not lost if the app is closed.").font(.caption).foregroundStyle(.secondary)
                    }

                    if vm.isBusy || vm.isPaused {
                        Section((vm.mediaQueue?.items.count ?? 0) > 1 ? "Current Item Clock" : "Overall Clock") {
                            ProgressView(value: vm.progress)
                            LabeledContent("Complete", value: String(format: "%.1f%%", vm.progress * 100))
                            LabeledContent("Elapsed", value: formatDuration(vm.elapsedSeconds))
                            LabeledContent("Remaining", value: vm.etaSeconds.map(formatDuration) ?? "Calibrating…")
                            if let eta = vm.etaSeconds, eta > 0 { LabeledContent("Estimated finish", value: finishTime(after: eta)) }
                        }
                        Section("Processing") {
                            Text(vm.inputKind == .gif
                                ? "GIF → temporary video → Compression Guard → RIFE HQ → Real-CUGAN → Final Sharpie → GIF"
                                : vm.inputKind == .image
                                    ? "Compression Guard → Real-CUGAN → Final Sharpie (RIFE skipped)"
                                    : vm.videoToGIFEnabled
                                        ? "Compression Guard → RIFE HQ → Real-CUGAN → Final Sharpie → GIF"
                                        : "Compression Guard → RIFE HQ → Real-CUGAN → Final Sharpie")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            ProgressView(value: vm.progress)
                            Text(vm.statusText).font(.caption)
                            HStack(spacing: 12) {
                                Button("Cancel", role: .destructive) { vm.cancel() }
                                Spacer()
                                Button {
                                    if vm.isPaused {
                                        Task { await vm.start() }
                                    } else {
                                        vm.pause()
                                    }
                                } label: {
                                    Label(vm.isPaused ? "Resume" : "Pause", systemImage: vm.isPaused ? "play.fill" : "pause.fill")
                                }
                                .buttonStyle(.bordered)
                                .disabled((!vm.isBusy && !vm.isPaused) || vm.inputKind == .image)
                            }
                        }
                        if vm.inputKind == .video {
                        Section("Live Performance") {
                            LabeledContent("Thermal / mode", value: vm.telemetry.thermalState)
                            LabeledContent("Memory headroom", value: String(format: "%.0f MB available", vm.telemetry.availableMemoryMB))
                            LabeledContent("Compression", value: String(format: "%.1f ms/source", vm.telemetry.compressionMsPerFrame))
                            LabeledContent("Final outline", value: String(format: "%.1f ms/frame", vm.telemetry.outlineMsPerFrame))
                            LabeledContent("RIFE HQ", value: String(format: "%.1f ms/generated", vm.telemetry.rifeMsPerGeneratedFrame))
                            LabeledContent("RIFE speed", value: String(format: "%.2f gen fps", vm.telemetry.generatedFPS))
                            if vm.upscaleTo4K {
                                LabeledContent("Real-CUGAN", value: String(format: "%.1f ms/frame", vm.telemetry.cuganMsPerFrame))
                                LabeledContent("Upscale speed", value: String(format: "%.2f fps", vm.telemetry.upscaleFPS))
                                LabeledContent("Upscaled", value: "\(vm.telemetry.upscaledFrames)")
                            }
                            LabeledContent("GhostGuard", value: String(format: "%.1f ms/generated", vm.telemetry.ghostMsPerGeneratedFrame))
                            LabeledContent("Encode", value: String(format: "%.1f ms/output", vm.telemetry.encodeMsPerOutputFrame))
                            LabeledContent("Generated", value: "\(vm.telemetry.generatedFrames)")
                            LabeledContent("Rejected", value: "\(vm.telemetry.rejectedFrames)")
                        }
                        }
                    }

                    Section("Diagnostics") {
                        Button { vm.benchmarkModels() } label: { Label("Benchmark Models", systemImage: "speedometer") }
                            .disabled(vm.isBusy || vm.isBenchmarking || vm.isImporting)
                        Button { vm.shareDiagnostics() } label: { Label("Share Diagnostics", systemImage: "square.and.arrow.up") }
                            .disabled(vm.diagnosticsShareItems.isEmpty && DiagnosticsExport.runFiles().isEmpty)
                        Button { vm.copyDiagnosticsJSON() } label: { Label("Copy Diagnostics", systemImage: "doc.on.doc") }
                        if vm.isBenchmarking {
                            ProgressView(value: vm.benchmarkProgress)
                            Button("Cancel Benchmark", role: .destructive) { vm.cancelBenchmark() }
                        }
                        if !vm.benchmarkStatus.isEmpty { Text(vm.benchmarkStatus).font(.caption) }
                        if !vm.benchmarkReport.isEmpty {
                            Text(vm.benchmarkReport)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.disabled)
                            Button {
                                UIPasteboard.general.string = vm.benchmarkReport
                            } label: {
                                Label("Copy Benchmark Report", systemImage: "doc.on.doc")
                            }
                        }
                        Text("Keep the app open and cool. Tests compute policies, then every available Real-CUGAN tile strategy on this phone. The fastest strategy that stays inside memory headroom is saved for this model, tile size, device, iOS version, and app build. The next render uses it. Before a benchmark, Real-CUGAN stays on sequential backed predictions.").font(.caption).foregroundStyle(.secondary)
                        Button { vm.copyErrorLogs() } label: { Label("Copy Error Logs", systemImage: "doc.on.doc") }
                        Button(role: .destructive) { vm.clearErrorLogs() } label: { Label("Clear Error Log", systemImage: "trash") }
                        if !vm.diagnosticsCopyStatus.isEmpty { Text(vm.diagnosticsCopyStatus).font(.caption).foregroundStyle(.secondary) }
                        Text("Copies the persistent render log, last stage/progress, thermal state, frame counters, RIFE speed, Real-CUGAN speed, export folder, and the last error directly to the clipboard.").font(.caption).foregroundStyle(.secondary)
                    }

                    if let out = vm.outputURL {
                        Section("Finished Output") {
                            if !vm.saveStatusText.isEmpty { Text(vm.saveStatusText).font(.caption) }
                            if out.pathExtension.lowercased() == "gif" {
                                Label("Animated GIF ready", systemImage: "livephoto")
                                Button {
                                    quickLookURL = out
                                } label: {
                                    Label("Preview animated GIF", systemImage: "play.rectangle")
                                }
                                Text("GIF exports are encoded to loop continuously and are validated after saving.")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else if let previewPlayer {
                                VideoPlayer(player: previewPlayer)
                                    .frame(height: 230)
                                    .clipShape(RoundedRectangle(cornerRadius: 16))
                            } else {
                                Button {
                                    previewPlayer = AVPlayer(url: out)
                                } label: {
                                    Label("Load playable video preview", systemImage: "play.rectangle")
                                }
                            }
                            ShareLink(item: out) { Label("Share Output", systemImage: "square.and.arrow.up") }
                            Text(out.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if let err = vm.errorText { Section("Error") { Text(err).foregroundStyle(.red) } }
                }
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    if !vm.isBusy && !vm.isPaused && !vm.recoveryAvailable {
                        VStack(spacing: 8) {
                            HStack {
                                Label(vm.inputURL?.lastPathComponent ?? "No media selected",
                                      systemImage: vm.inputKind == .image ? "photo" : vm.inputKind == .gif ? "livephoto" : "film")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                Spacer(minLength: 8)
                                if vm.inputURL != nil {
                                    Text(vm.inputKind == .image ? (vm.upscaleTo4K ? "2× output" : "Enhance") : (vm.videoToGIFEnabled || vm.inputKind == .gif ? "GIF output" : "60 FPS"))
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Button {
                                Task { await vm.start() }
                            } label: {
                                Label(createButtonTitle, systemImage: "play.fill")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 7)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Color(red: 0.04, green: 0.52, blue: 1.0))
                            .disabled(vm.inputURL == nil || vm.isBenchmarking || vm.isImporting || (vm.mediaQueue != nil && !vm.queueHasRemaining))
                        }
                        .padding(.horizontal, 16)
                        .padding(.top, 12)
                        .padding(.bottom, 8)
                        .background(.ultraThinMaterial)
                        .overlay(alignment: .top) { Rectangle().fill(.white.opacity(0.08)).frame(height: 1) }
                    }
                }
                .navigationTitle("RIFE 60")
                .scrollContentBackground(.hidden)
                .background(Color(red: 0.035, green: 0.045, blue: 0.075))
                .preferredColorScheme(.dark)
                .quickLookPreview($quickLookURL)
                .fullScreenCover(isPresented: $showingAIImageStudio) {
                    AIImageStudioView(
                        canUpscale: !vm.queueLocked && !vm.isImporting && !vm.isBenchmarking,
                        onUpscale: { imageURL in
                            Task {
                                await vm.handleImport(.success([imageURL]))
                                if !vm.isBusy && !vm.isImporting {
                                    await vm.start()
                                }
                            }
                        }
                    )
                }
                .photosPicker(isPresented: $showingPhotosPicker, selection: $photoItems, maxSelectionCount: 100, selectionBehavior: .ordered, matching: .any(of: [.videos, .images]), photoLibrary: .shared())
                .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .video, .image], allowsMultipleSelection: true) { result in Task { await vm.handleImport(result) } }
                .fileImporter(isPresented: $showingExportFolderPicker, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
                    vm.handleExportFolderSelection(result)
                    lastCompletedVideoURL = LastCompletedVideoStore.latestCompletedVideo() ?? lastCompletedVideoURL
                }
                .onAppear {
                    // Migrate the old boolean setting once so the segmented picker
                    // and the renderer agree on the same persisted mode.
                    let warmthDefaults = UserDefaults.standard
                    let warmthMigrationKey = "automaticWarmthCorrectionModeMigrationComplete"
                    if !warmthDefaults.bool(forKey: warmthMigrationKey) {
                        if let legacyValue = warmthDefaults.object(forKey: "automaticWarmthCorrectionEnabled") as? Bool {
                            automaticWarmthCorrectionModeRawValue = legacyValue ? AutoWarmthCorrectionMode.full.rawValue : AutoWarmthCorrectionMode.off.rawValue
                        }
                        warmthDefaults.set(true, forKey: warmthMigrationKey)
                    }
                    if lastCompletedVideoURL == nil { lastCompletedVideoURL = LastCompletedVideoStore.latestCompletedVideo() }
                    if vm.isBusy { pipController.arm() }
                    refreshPiPStatus(force: true)
                }
                .onChange(of: photoItems) { items in guard !items.isEmpty else { return }; Task { await vm.handlePhotoSelections(items); photoItems = [] } }
                .onChange(of: scenePhase) { phase in
                    vm.handleScenePhase(phase)
                    if vm.isBusy {
                        refreshPiPStatus(force: true)
                        if phase == .active {
                            pipController.stop()
                        } else {
                            pipController.startIfPossible()
                        }
                    } else {
                        pipController.disarmAndStop()
                    }
                    if phase == .active, lastCompletedVideoURL == nil { lastCompletedVideoURL = LastCompletedVideoStore.latestCompletedVideo() }
                }
                .onChange(of: vm.isBusy) { processing in
                    if processing {
                        cancelPostRenderSleep(restoreDisplay: true)
                        pipController.arm()
                        refreshPiPStatus(force: true)
                        if scenePhase != .active { pipController.startIfPossible() }
                    } else {
                        pipController.disarmAndStop()
                        schedulePostRenderSleepIfNeeded()
                    }
                }
                .onChange(of: vm.progress) { _ in refreshPiPStatus() }
                .onChange(of: vm.statusText) { _ in refreshPiPStatus() }
                .onChange(of: vm.etaSeconds) { _ in refreshPiPStatus() }
                .onChange(of: vm.outputURL) { newURL in
                    if let newURL,
                       ["mov", "mp4", "m4v"].contains(newURL.pathExtension.lowercased()) {
                        lastCompletedVideoURL = newURL
                        previewPlayer = AVPlayer(url: newURL)
                    } else {
                        previewPlayer = nil
                    }
                    schedulePostRenderSleepIfNeeded()
                }
                .confirmationDialog("Copy mask to other media?", isPresented: $showingMaskCopyConfirmation, titleVisibility: .visible) {
                    Button("Copy to other queued media") { vm.copyMaskToOtherMedia() }
                    Button("Cancel", role: .cancel) { }
                } message: {
                    Text("This replaces their existing masks at the same relative image position. Different artwork needs a different mask. Review every item before processing.")
                }
                .sheet(isPresented: $showingWatermarkEditor) {
                    if let source = vm.inputURL {
                        WatermarkMaskEditor(sourceURL: source, isImage: vm.inputKind == .image, regions: $vm.watermarkRegions)
                    }
                }
                .alert("Clear Recovery Data?", isPresented: $showingClearRecoveryConfirmation) {
                    Button("Clear Recovery Data", role: .destructive) { showingClearRecoveryConfirmation = false; vm.clearRecoveryData() }
                    Button("Cancel", role: .cancel) { }
                } message: { Text("This permanently deletes the saved source copy and completed render checkpoints for this interrupted job. It does not delete your last completed exported video.") }
            }

            if vm.isBusy && !vm.processingScreenAwake {
                Color.black.ignoresSafeArea().contentShape(Rectangle()).onTapGesture { vm.wakeProcessingScreen() }.zIndex(999)
            }
            if postRenderScreenDimmed {
                Color.black.ignoresSafeArea().contentShape(Rectangle()).onTapGesture { registerPostRenderInteraction() }.zIndex(1000)
            }
        }
        .simultaneousGesture(DragGesture(minimumDistance: 0).onChanged { _ in registerPostRenderInteraction() })
        .sheet(isPresented: $vm.showingDiagnosticsShare) {
            DiagnosticsActivityView(items: vm.diagnosticsShareItems)
        }
        .sheet(isPresented: $showingH3VideoEditor) {
            H3VideoEditView()
        }
        .sheet(isPresented: $showingPreview) {
            VideoSettingsPreviewSheet(viewModel: vm)
        }
        .onDisappear {
            pipController.disarmAndStop()
            cancelPostRenderSleep(restoreDisplay: true)
        }
    }

    private var mediaQueueSection: some View {
        Group {
                    if let queue = vm.mediaQueue, queue.items.count > 1 {
                        Section("Media Queue") {
                            Text(vm.batchStatusText).font(.caption)
                            ProgressView(value: vm.batchProgress)
                            Text("One item renders and saves before the next starts. Tap an item to edit its watermark mask.")
                                .font(.caption).foregroundStyle(.secondary)
                            ForEach(Array(queue.items.enumerated()), id: \.element.id) { index, item in
                                VStack(alignment: .leading, spacing: 6) {
                                    Button { vm.selectQueueItem(item.id) } label: {
                                        HStack {
                                            Image(systemName: item.kind == .image ? "photo" : "film")
                                            VStack(alignment: .leading) {
                                                Text("\(index + 1). \(item.displayName)").lineLimit(1)
                                                Text("\(item.state.rawValue.capitalized) • \(item.masks.isEmpty ? "No mask" : "Mask saved")").font(.caption)
                                            }
                                            Spacer()
                                            if vm.selectedMediaID == item.id { Image(systemName: "checkmark.circle.fill") }
                                        }
                                    }.buttonStyle(.borderless).disabled(vm.queueLocked || vm.isImporting)
                                    if !vm.queueLocked {
                                        HStack {
                                            Button { vm.moveQueueItem(item.id, by: -1) } label: { Label("Up", systemImage: "arrow.up") }.disabled(index == 0)
                                            Button { vm.moveQueueItem(item.id, by: 1) } label: { Label("Down", systemImage: "arrow.down") }.disabled(index == queue.items.count - 1)
                                        }.font(.caption).buttonStyle(.borderless)
                                    }
                                    if let output = item.outputURL {
                                        ShareLink(item: output) { Label("Share saved output", systemImage: "square.and.arrow.up") }.font(.caption)
                                    }
                                }
                            }
                            Button("Clear media queue", role: .destructive) { vm.clearMediaQueue() }.disabled(vm.isBusy || vm.isImporting)
                        }
                    }

        }
    }


    private func recipeChip(_ title: String, symbol: String) -> some View {
        Label(title, systemImage: symbol)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.white.opacity(0.07), in: Capsule())
            .overlay(Capsule().stroke(.white.opacity(0.08), lineWidth: 1))
            .lineLimit(1)
    }

    private var createButtonTitle: String {
        if let queue = vm.mediaQueue, queue.items.count > 1 {
            return queue.settingsKey == nil ? "Process \(queue.items.count) Media in Order" : "Resume Media Queue"
        }
        if vm.inputKind == .gif {
            return vm.redditMode ? "Prepare Reddit GIF" : "Process GIF • Full Pipeline"
        }
        if vm.inputKind == .image { return vm.upscaleTo4K ? "Create 2× Image" : "Create Enhanced Image" }
        return vm.upscaleTo4K ? "Create 2× 60 FPS Video" : "Create 60 FPS Video"
    }

    @MainActor
    private func refreshPiPStatus(force: Bool = false) {
        guard vm.isBusy else { return }
        pipController.enqueueStatusFrame(
            progress: vm.progress,
            status: vm.statusText,
            elapsed: vm.elapsedSeconds,
            eta: vm.etaSeconds,
            force: force
        )
    }

    @MainActor
    private func schedulePostRenderSleepIfNeeded() {
        guard !vm.isBusy, vm.outputURL != nil, vm.statusText == "Finished" else { return }
        postRenderSleepTask?.cancel()
        postRenderScreenDimmed = false
        if postRenderSavedBrightness == nil { postRenderSavedBrightness = UIScreen.main.brightness }
        if postRenderSavedIdleTimerDisabled == nil { postRenderSavedIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled }

        UIApplication.shared.isIdleTimerDisabled = true
        postRenderSleepTask = Task { @MainActor in
            do {
                try await Task.sleep(nanoseconds: 60_000_000_000)
            } catch {
                return
            }
            guard !Task.isCancelled, !vm.isBusy, vm.outputURL != nil else { return }
            postRenderScreenDimmed = true
            UIScreen.main.brightness = 0.01
            UIApplication.shared.isIdleTimerDisabled = false
            postRenderSleepTask = nil
        }
    }

    @MainActor
    private func registerPostRenderInteraction() {
        guard !vm.isBusy, vm.outputURL != nil else { return }
        cancelPostRenderSleep(restoreDisplay: true)
    }

    @MainActor
    private func cancelPostRenderSleep(restoreDisplay: Bool) {
        postRenderSleepTask?.cancel()
        postRenderSleepTask = nil
        postRenderScreenDimmed = false
        guard restoreDisplay else { return }
        if let brightness = postRenderSavedBrightness { UIScreen.main.brightness = brightness }
        if let idle = postRenderSavedIdleTimerDisabled { UIApplication.shared.isIdleTimerDisabled = idle }
        postRenderSavedBrightness = nil
        postRenderSavedIdleTimerDisabled = nil
    }

    private func finishTime(after seconds: Double) -> String {
        let formatter = DateFormatter(); formatter.timeStyle = .short
        return formatter.string(from: Date().addingTimeInterval(seconds))
    }
}


private struct VideoSettingsPreviewSheet: View {
    @ObservedObject var viewModel: VideoProcessorViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(viewModel.previewStatus.isEmpty ? "Preview samples the opening frames using your current settings." : viewModel.previewStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if viewModel.isPreviewing {
                        ProgressView("Rendering preview frames…")
                        Button("Cancel Preview") { viewModel.cancelPreview() }
                            .buttonStyle(.bordered)
                    }
                    if viewModel.previewURLs.isEmpty && !viewModel.isPreviewing {
                        Text("No preview frames yet. Tap Preview Opening Frames after selecting a video or GIF.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(Array(viewModel.previewURLs.enumerated()), id: \.offset) { index, url in
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Sample frame \(index + 1)").font(.headline)
                            if let image = UIImage(contentsOfFile: url.path) {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxWidth: .infinity)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                            } else {
                                Text("This preview frame could not be displayed.")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    Text("This is a visual settings check, not a 60 FPS motion preview. Change settings on the main screen and run Preview again.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Video / GIF Preview")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}
