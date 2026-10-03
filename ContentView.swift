import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import UIKit

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var vm = VideoProcessorViewModel()
    @StateObject private var pipController = ProcessingPiPController()
    @State private var showingImporter = false
    @State private var showingGenerativeModelImporter = false
    @State private var generativeEditExpanded = false
    @State private var showingWatermarkEditor = false
    @State private var showingPhotosPicker = false
    @State private var showingExportFolderPicker = false
    @State private var showingClearRecoveryConfirmation = false
    @State private var photoItems: [PhotosPickerItem] = []
    @State private var showingMaskCopyConfirmation = false
    @State private var lastCompletedVideoURL: URL?
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
                            if vm.inputKind == .image { Text("Image • RIFE interpolation is skipped").font(.caption).foregroundStyle(.secondary) }
                        }
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
                            Text("A single image has no neighbouring frame to interpolate, so images go straight through Compression Guard → Real-CUGAN → Final Sharpie and are saved as a PNG.").font(.caption).foregroundStyle(.secondary)
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
                            Text("Anime/Manga LaMa reconstructs marked areas before RIFE and upscaling. Painted masks apply throughout a video. Use the brush and eraser to preserve the artwork between letters.").font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    Section {
                        DisclosureGroup(isExpanded: $generativeEditExpanded) {
                            if vm.inputKind == .video {
                                TextField("Edit prompt", text: $vm.generativeEdit.prompt, axis: .vertical)
                                    .lineLimit(3...6)
                                    .disabled(vm.isBusy || vm.isImporting)
                                TextField("Negative prompt", text: $vm.generativeEdit.negativePrompt, axis: .vertical)
                                    .lineLimit(2...5)
                                    .disabled(vm.isBusy || vm.isImporting)
                                HStack { Text("Edit strength"); Spacer(); Text(String(format: "%.2f", vm.generativeEdit.strength)) }
                                Slider(value: $vm.generativeEdit.strength, in: 0.45...1.0, step: 0.01)
                                    .disabled(vm.isBusy || vm.isImporting)
                                Stepper("Steps: \(vm.generativeEdit.steps)", value: $vm.generativeEdit.steps, in: 4...20)
                                    .disabled(vm.isBusy || vm.isImporting)
                                HStack {
                                    Text("Seed")
                                    Spacer()
                                    TextField("Random", value: $vm.generativeEdit.seed, format: .number)
                                        .keyboardType(.numbersAndPunctuation)
                                        .multilineTextAlignment(.trailing)
                                        .frame(width: 120)
                                }
                                Toggle("Preserve original audio", isOn: $vm.preserveAudio)
                                    .disabled(vm.isBusy || vm.isImporting)
                                Button { showingGenerativeModelImporter = true } label: {
                                    Label("Install VACE model files…", systemImage: "arrow.down.doc")
                                }
                                .buttonStyle(.borderless)
                                .disabled(vm.isBusy || vm.isImporting)
                                Text(vm.generativeModelStatus).font(.caption).foregroundStyle(.secondary)
                                Text("True temporal reconstruction: the model regenerates the video from a multi-frame source window instead of digitally placing a new object on top. The 1.3B profile uses up to 832×480 and 33-frame windows.")
                                    .font(.caption).foregroundStyle(.secondary)
                                Button { vm.startGenerativeEdit() } label: {
                                    Label("Run Anime Generative Edit", systemImage: "wand.and.stars")
                                }
                                .buttonStyle(.borderless)
                                .disabled(vm.isBusy || vm.isImporting || vm.isBenchmarking || vm.inputURL == nil)
                                if !vm.generativeEditStatus.isEmpty { Text(vm.generativeEditStatus).font(.caption).foregroundStyle(.secondary) }
                            } else {
                                Text("Select a video to use generative video editing.").font(.caption).foregroundStyle(.secondary)
                            }
                        } label: {
                            Label("Anime Generative Edit", systemImage: "wand.and.stars")
                        }
                    }

                    Section("Anime Processing") {
                        Toggle("Compression Guard", isOn: $vm.compressionProtection).disabled(vm.queueLocked || vm.isImporting)
                        Toggle("Final Sharpie Outline", isOn: $vm.outlineProtection).disabled(vm.queueLocked || vm.isImporting)
                        Toggle("Color Pop", isOn: $vm.colorPopEnabled).disabled(vm.queueLocked || vm.isImporting)
                        if vm.colorPopEnabled {
                            HStack { Text("Color Pop strength"); Spacer(); Text(String(format: "%.2f", vm.colorPopStrength)) }
                            Slider(value: $vm.colorPopStrength, in: 0.1...1.0).disabled(vm.queueLocked || vm.isImporting)
                            Text("Selective vibrance after upscale and sharpening. Protects grays and highlights; gently cleans orange skin tones.").font(.caption).foregroundStyle(.secondary)
                        }
                        Text("Compression Guard runs before RIFE. Sharpie now runs as the final visual pass after Real-CUGAN, so CUGAN cannot soften or change the finished outlines. This revision is slightly narrower and closer to the original anime line width while remaining a little thicker/sharper than the original.").font(.caption).foregroundStyle(.secondary)
                    }

                    Section("2× Anime Upscale") {
                        Toggle("Real-CUGAN Native 2×", isOn: $vm.upscaleTo4K).disabled(vm.queueLocked || vm.isImporting)
                        Picker("Model compute units", selection: $vm.computePreference) {
                            ForEach(ModelComputePreference.allCases) { option in Text(option.title).tag(option) }
                        }.disabled(vm.queueLocked || vm.isBusy || vm.isBenchmarking)
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
                        if vm.inputKind == .image {
                            LabeledContent("Image format", value: "PNG (lossless)")
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

                    if vm.isBusy {
                        Section((vm.mediaQueue?.items.count ?? 0) > 1 ? "Current Item Clock" : "Overall Clock") {
                            ProgressView(value: vm.progress)
                            LabeledContent("Complete", value: String(format: "%.1f%%", vm.progress * 100))
                            LabeledContent("Elapsed", value: formatDuration(vm.elapsedSeconds))
                            LabeledContent("Remaining", value: vm.etaSeconds.map(formatDuration) ?? "Calibrating…")
                            if let eta = vm.etaSeconds, eta > 0 { LabeledContent("Estimated finish", value: finishTime(after: eta)) }
                        }
                        Section("Processing") {
                            Text(vm.inputKind == .image ? "Compression Guard → Real-CUGAN → Final Sharpie (RIFE skipped)" : "Compression Guard → RIFE HQ → Real-CUGAN → Final Sharpie").font(.caption).foregroundStyle(.secondary)
                            ProgressView(value: vm.progress)
                            Text(vm.statusText).font(.caption)
                            Button("Cancel", role: .destructive) { vm.cancel() }
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
                    } else if !vm.recoveryAvailable {
                        Section {
                            Button { Task { await vm.start() } } label: { Label(createButtonTitle, systemImage: "wand.and.stars") }.disabled(vm.inputURL == nil || vm.isBenchmarking || vm.isImporting || (vm.mediaQueue != nil && !vm.queueHasRemaining))
                        }
                    }

                    Section("Diagnostics") {
                        Button { vm.benchmarkModels() } label: { Label("Benchmark Models", systemImage: "speedometer") }
                            .disabled(vm.isBusy || vm.isBenchmarking || vm.isImporting)
                        if vm.isBenchmarking {
                            ProgressView(value: vm.benchmarkProgress)
                            Button("Cancel Benchmark", role: .destructive) { vm.cancelBenchmark() }
                        }
                        if !vm.benchmarkStatus.isEmpty { Text(vm.benchmarkStatus).font(.caption) }
                        if !vm.benchmarkReport.isEmpty { Text(vm.benchmarkReport).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
                        Text("Keep the app open and cool. Tests 2 warm-ups and 5 timed predictions per model/policy. Results are device measurements; no preference is changed automatically.").font(.caption).foregroundStyle(.secondary)
                        Button { vm.copyErrorLogs() } label: { Label("Copy Error Logs", systemImage: "doc.on.doc") }
                        if !vm.diagnosticsCopyStatus.isEmpty { Text(vm.diagnosticsCopyStatus).font(.caption).foregroundStyle(.secondary) }
                        Text("Copies the persistent render log, last stage/progress, thermal state, frame counters, RIFE speed, Real-CUGAN speed, export folder, and the last error directly to the clipboard.").font(.caption).foregroundStyle(.secondary)
                    }

                    if let out = vm.outputURL {
                        Section("Finished") {
                            if !vm.saveStatusText.isEmpty { Text(vm.saveStatusText).font(.caption) }
                            ShareLink(item: out) { Label("Share Output", systemImage: "square.and.arrow.up") }
                            Text(out.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if let err = vm.errorText { Section("Error") { Text(err).foregroundStyle(.red) } }
                }
                .navigationTitle("RIFE 60")
                .photosPicker(isPresented: $showingPhotosPicker, selection: $photoItems, maxSelectionCount: 100, selectionBehavior: .ordered, matching: .any(of: [.videos, .images]), photoLibrary: .shared())
                .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .video, .image], allowsMultipleSelection: true) { result in Task { await vm.handleImport(result) } }
                .fileImporter(isPresented: $showingGenerativeModelImporter, allowedContentTypes: [.data, .item], allowsMultipleSelection: true) { result in
                    switch result {
                    case .success(let urls):
                        vm.installGenerativeModels(urls)
                    case .failure(let error):
                        vm.errorText = error.localizedDescription
                    }
                }
                .fileImporter(isPresented: $showingExportFolderPicker, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
                    vm.handleExportFolderSelection(result)
                    lastCompletedVideoURL = LastCompletedVideoStore.latestCompletedVideo() ?? lastCompletedVideoURL
                }
                .onAppear {
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
                    if let newURL { lastCompletedVideoURL = newURL }
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

    private var createButtonTitle: String {
        if let queue = vm.mediaQueue, queue.items.count > 1 {
            return queue.settingsKey == nil ? "Process \(queue.items.count) Media in Order" : "Resume Media Queue"
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

