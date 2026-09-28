import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var vm = VideoProcessorViewModel()
    @State private var showingImporter = false
    @State private var showingPhotosPicker = false
    @State private var showingExportFolderPicker = false
    @State private var showingClearRecoveryConfirmation = false
    @State private var photoItem: PhotosPickerItem?

    var body: some View {
        ZStack {
            NavigationStack {
                Form {
                    Section("Input") {
                        Button { showingPhotosPicker = true } label: { Label("Select from Photos", systemImage: "photo.on.rectangle") }
                        Button { showingImporter = true } label: { Label("Select from Files", systemImage: "folder") }
                        if vm.isImporting {
                            VStack(alignment: .leading, spacing: 6) {
                                if let importProgress = vm.importProgress { ProgressView(value: importProgress) } else { ProgressView() }
                                Text("Importing video…").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        if let input = vm.inputURL { Text(input.lastPathComponent).font(.caption).foregroundStyle(.secondary) }
                    }

                    if vm.recoveryAvailable {
                        Section("Crash Recovery") {
                            Label("Recovery data found", systemImage: "arrow.clockwise.circle.fill").foregroundStyle(.orange)
                            Text(vm.recoveryStatusText).font(.caption)
                            if !vm.isProcessing {
                                Button { Task { await vm.start() } } label: { Label(vm.upscaleTo4K ? "Resume 2× 60 FPS Render" : "Resume 60 FPS Render", systemImage: "play.fill") }
                                Button(role: .destructive) { showingClearRecoveryConfirmation = true } label: { Label("Clear Recovery Data", systemImage: "trash") }
                            }
                            Text("Completed AI passes are kept in persistent storage. If iOS terminates the app, reopening it reuses every completed checkpoint instead of starting the whole render over.").font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    Section("RIFE 4.26") {
                        LabeledContent("Interpolation quality", value: "High Quality (HQ)")
                        Toggle("Ghost protection", isOn: $vm.ghostProtection).disabled(vm.recoveryAvailable)
                        Toggle("Scene-cut protection", isOn: $vm.sceneCutProtection).disabled(vm.recoveryAvailable)
                        LabeledContent("Target", value: "60.00 fps")
                        Text("1080p-class video uses one persistent full-frame HQ stream for maximum speed without dropping RIFE quality.").font(.caption).foregroundStyle(.secondary)
                    }

                    Section("Anime Restoration") {
                        Toggle("Compression Guard", isOn: $vm.compressionProtection).disabled(vm.recoveryAvailable)
                        Toggle("Sharpie Outline Subtle", isOn: $vm.outlineProtection).disabled(vm.recoveryAvailable)
                        Text("Uses the final slightly-thinner Sharpie revision on source frames before RIFE.").font(.caption).foregroundStyle(.secondary)
                    }

                    Section("2× Anime Upscale") {
                        Toggle("Real-CUGAN Native 2×", isOn: $vm.upscaleTo4K).disabled(vm.recoveryAvailable)
                        LabeledContent("Model", value: "Real-CUGAN – Anime")
                        LabeledContent("Upscale", value: "2× source resolution")
                        LabeledContent("Noise", value: "Level 3")
                        LabeledContent("Intensity", value: "1.30")
                        Text("Runs as a separate final AI pass after RIFE. Real-CUGAN processes overlapping tiles and stitches them back to exactly 2× the original source dimensions without converting the source to 1080p first.").font(.caption).foregroundStyle(.secondary)
                    }

                    Section("Ghost Guard") {
                        VStack(alignment: .leading, spacing: 5) {
                            HStack { Text("Sensitivity"); Spacer(); Text(String(format: "%.2f", vm.ghostSensitivity)) }
                            Slider(value: $vm.ghostSensitivity, in: 0.65...1.35, step: 0.05).disabled(vm.recoveryAvailable)
                        }
                    }

                    Section("Render Power") {
                        Toggle("Render power mode", isOn: $vm.renderPowerMode)
                        Text("The processing screen turns completely black immediately. Tap once to wake it; after 20 seconds without a touch it returns to black. OLED black + 1% brightness minimizes display heat while processing continues at high priority.").font(.caption).foregroundStyle(.secondary)
                    }

                    Section("Export to Files") {
                        LabeledContent("Final codec", value: "HEVC Main10")
                        LabeledContent("Pixel format", value: "10-bit P010")
                        LabeledContent("File-size target", value: "< 1 GB")
                        Toggle("Preserve original audio", isOn: $vm.preserveAudio).disabled(vm.recoveryAvailable)
                        LabeledContent("Auto-save folder", value: vm.exportFolderName)
                        Button { showingExportFolderPicker = true } label: { Label("Choose Files Export Folder", systemImage: "folder.badge.plus") }
                        Button("Use Default App Export Folder") { vm.useDefaultExportFolder() }
                        Text("Finished videos are automatically copied into this Files folder. Choose a folder once and the app remembers it for future renders. This avoids the Photos import issue and means you no longer need to use Copy/Paste to get the finished video into Files.").font(.caption).foregroundStyle(.secondary)
                    }

                    if vm.isProcessing {
                        Section("Overall Clock") {
                            ProgressView(value: vm.progress)
                            LabeledContent("Complete", value: String(format: "%.1f%%", vm.progress * 100))
                            LabeledContent("Elapsed", value: formatDuration(vm.elapsedSeconds))
                            LabeledContent("Remaining", value: vm.etaSeconds.map(formatDuration) ?? "Calibrating…")
                            if let eta = vm.etaSeconds, eta > 0 { LabeledContent("Estimated finish", value: finishTime(after: eta)) }
                        }
                        Section("Processing") {
                            Text("Restoration / outline").font(.caption).foregroundStyle(.secondary)
                            ProgressView(value: vm.restorationProgress)
                            if vm.upscaleTo4K { Text("Real-CUGAN native 2× upscale").font(.caption).foregroundStyle(.secondary); ProgressView(value: vm.upscaleProgress) }
                            Text(vm.statusText).font(.caption)
                            Button("Cancel", role: .destructive) { vm.cancel() }
                        }
                        Section("Live Performance") {
                            LabeledContent("Thermal", value: vm.telemetry.thermalState)
                            LabeledContent("Compression", value: String(format: "%.1f ms/source", vm.telemetry.compressionMsPerFrame))
                            LabeledContent("Outline", value: String(format: "%.1f ms/source", vm.telemetry.outlineMsPerFrame))
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
                    } else if !vm.recoveryAvailable {
                        Section {
                            Button { Task { await vm.start() } } label: { Label(vm.upscaleTo4K ? "Create 2× 60 FPS Video" : "Create 60 FPS Video", systemImage: "wand.and.stars") }.disabled(vm.inputURL == nil)
                        }
                    }

                    Section("Diagnostics") {
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
                .photosPicker(isPresented: $showingPhotosPicker, selection: $photoItem, matching: .videos, photoLibrary: .shared())
                .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .video], allowsMultipleSelection: false) { result in vm.handleImport(result) }
                .fileImporter(isPresented: $showingExportFolderPicker, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in vm.handleExportFolderSelection(result) }
                .onChange(of: photoItem) { item in guard let item else { return }; Task { await vm.handlePhotoSelection(item); photoItem = nil } }
                .onChange(of: scenePhase) { phase in vm.handleScenePhase(phase) }
                .alert("Clear Recovery Data?", isPresented: $showingClearRecoveryConfirmation) {
                    Button("Clear Recovery Data", role: .destructive) { vm.clearRecoveryData() }
                    Button("Cancel", role: .cancel) { }
                } message: { Text("This permanently deletes the saved source copy and completed render checkpoints for this interrupted job. You can then select a new video and start fresh.") }
            }
            if vm.isProcessing && !vm.processingScreenAwake { Color.black.ignoresSafeArea().contentShape(Rectangle()).onTapGesture { vm.wakeProcessingScreen() }.zIndex(999) }
        }
    }

    private func finishTime(after seconds: Double) -> String {
        let formatter = DateFormatter(); formatter.timeStyle = .short
        return formatter.string(from: Date().addingTimeInterval(seconds))
    }
}
