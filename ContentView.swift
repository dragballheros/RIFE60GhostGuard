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
                    .disabled(vm.isImporting || vm.isProcessing)

                    Button {
                        showingImporter = true
                    } label: {
                        Label("Select from Files", systemImage: "folder")
                    }
                    .disabled(vm.isImporting || vm.isProcessing)

                    if vm.isImporting {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Importing video…")
                                .font(.caption)
                            if let p = vm.importProgress {
                                ProgressView(value: p)
                            } else {
                                ProgressView()
                            }
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
                    Text("Uses adaptive tiled HQ with persistent per-band RIFE streams.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Anime Restoration") {
                    Toggle("Compression Guard", isOn: $vm.compressionProtection)
                    Toggle("Sharpie Outline Subtle", isOn: $vm.outlineProtection)
                    Text("The outline is the final slightly-reduced Subtle version: a tiny amount of the cleaned source is mixed back in so the added ink reads a little narrower without losing the trained line placement.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Render Power Mode") {
                    Toggle("Dim screen while processing", isOn: $vm.renderPowerMode)
                    Text("Dims the display to about 5%, prevents auto-lock, keeps the render task high priority, and restores your previous brightness afterward. iOS Low Power Mode is not enabled because it would also throttle this app.")
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
                    Text("Rejected synthetic frames are replaced by the nearest processed real source frame instead of another generated frame.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Export") {
                    LabeledContent("Codec", value: "HEVC Main10")
                    LabeledContent("Pixel format", value: "10-bit P010")
                    LabeledContent("Quality", value: "Maximum")
                    Toggle("Preserve original audio", isOn: $vm.preserveAudio)
                    Text("Export is always high-bitrate H.265/HEVC Main10. There is no lower-quality codec mode.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if vm.isProcessing {
                    Section("Processing") {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("Overall")
                                Spacer()
                                Text("\(Int(vm.progress * 100))%")
                                    .monospacedDigit()
                            }
                            .font(.caption)
                            ProgressView(value: vm.progress)
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("Cleanup + outline")
                                Spacer()
                                Text("\(Int(vm.restorationProgress * 100))%")
                                    .monospacedDigit()
                            }
                            .font(.caption)
                            ProgressView(value: vm.restorationProgress)
                        }

                        Text(vm.statusText).font(.caption)
                        Button("Cancel", role: .destructive) { vm.cancel() }
                    }
                } else {
                    Section {
                        Button {
                            Task { await vm.start() }
                        } label: {
                            Label("Create 60 FPS Video", systemImage: "wand.and.stars")
                        }
                        .disabled(vm.inputURL == nil || vm.isImporting)
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
