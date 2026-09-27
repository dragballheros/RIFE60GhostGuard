import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var vm = VideoProcessorViewModel()
    @State private var showingImporter = false
    @State private var photoItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            Form {
                Section("Input") {
                    PhotosPicker(selection: $photoItem, matching: .videos) {
                        Label("Select from Photos", systemImage: "photo.on.rectangle")
                    }

                    Button {
                        showingImporter = true
                    } label: {
                        Label("Select from Files", systemImage: "folder")
                    }

                    if vm.isImporting {
                        VStack(alignment: .leading, spacing: 6) {
                            if let importProgress = vm.importProgress {
                                ProgressView(value: importProgress)
                            } else {
                                ProgressView()
                            }
                            Text("Importing video…")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if let input = vm.inputURL {
                        Text(input.lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("RIFE 4.26") {
                    LabeledContent("Interpolation quality", value: "High Quality (HQ)")
                    Toggle("Ghost protection", isOn: $vm.ghostProtection)
                    Toggle("Scene-cut protection", isOn: $vm.sceneCutProtection)
                    LabeledContent("Target", value: "60.00 fps")
                    Text("Uses persistent streaming tiled RIFE HQ for lower memory usage and cached previous-frame features.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Anime Restoration") {
                    Toggle("Compression Guard", isOn: $vm.compressionProtection)
                    Toggle("Sharpie Outline Subtle", isOn: $vm.outlineProtection)
                    Text("The outline is the final slightly-thinner revision. Source frames are cleaned and outlined before RIFE; generated frames inherit the processed line style.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Ghost Guard") {
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text("Sensitivity")
                            Spacer()
                            Text(String(format: "%.2f", vm.ghostSensitivity))
                        }
                        Slider(value: $vm.ghostSensitivity, in: 0.65...1.35, step: 0.05)
                    }
                }

                Section("Render Power") {
                    Toggle("Render power mode", isOn: $vm.renderPowerMode)
                    Text("Dims the display to 5%, prevents auto-lock, and keeps the render task high-priority. It does not enable iOS Low Power Mode, which would throttle processing.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Export") {
                    LabeledContent("Codec", value: "HEVC Main10")
                    LabeledContent("Pixel format", value: "10-bit P010")
                    LabeledContent("Quality", value: "Maximum")
                    Toggle("Preserve original audio", isOn: $vm.preserveAudio)
                }

                if vm.isProcessing {
                    Section("Processing") {
                        Text("Overall")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ProgressView(value: vm.progress)

                        Text("Restoration / outline")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ProgressView(value: vm.restorationProgress)

                        Text(vm.statusText)
                            .font(.caption)

                        Button("Cancel", role: .destructive) { vm.cancel() }
                    }

                    Section("Live Performance") {
                        LabeledContent("Thermal", value: vm.telemetry.thermalState)
                        LabeledContent("Compression", value: String(format: "%.1f ms/source", vm.telemetry.compressionMsPerFrame))
                        LabeledContent("Outline", value: String(format: "%.1f ms/source", vm.telemetry.outlineMsPerFrame))
                        LabeledContent("RIFE HQ", value: String(format: "%.1f ms/generated", vm.telemetry.rifeMsPerGeneratedFrame))
                        LabeledContent("GhostGuard", value: String(format: "%.1f ms/generated", vm.telemetry.ghostMsPerGeneratedFrame))
                        LabeledContent("Encode", value: String(format: "%.1f ms/output", vm.telemetry.encodeMsPerOutputFrame))
                        LabeledContent("RIFE speed", value: String(format: "%.2f gen fps", vm.telemetry.generatedFPS))
                        LabeledContent("Generated", value: "\(vm.telemetry.generatedFrames)")
                        LabeledContent("Rejected", value: "\(vm.telemetry.rejectedFrames)")
                        Text("If RIFE ms/generated rises while Thermal changes from Nominal → Fair → Serious, that confirms thermal throttling is the main slowdown.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Section {
                        Button {
                            Task { await vm.start() }
                        } label: {
                            Label("Create 60 FPS Video", systemImage: "wand.and.stars")
                        }
                        .disabled(vm.inputURL == nil)
                    }
                }

                if let out = vm.outputURL {
                    Section("Finished") {
                        ShareLink(item: out) {
                            Label("Share / Save Output", systemImage: "square.and.arrow.up")
                        }
                        Text(out.lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                if let err = vm.errorText {
                    Section("Error") {
                        Text(err).foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("RIFE 60")
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.video],
                allowsMultipleSelection: false
            ) { result in
                vm.handleImport(result)
            }
            .onChange(of: photoItem) { item in
                guard let item else { return }
                Task {
                    await vm.handlePhotoSelection(item)
                    photoItem = nil
                }
            }
            .onChange(of: scenePhase) { phase in
                vm.handleScenePhase(phase)
            }
        }
    }
}
