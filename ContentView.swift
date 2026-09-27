import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ContentView: View {
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
                    Text("Uses RIFE HQ with the lower-memory streaming path.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Anime Restoration") {
                    Toggle("Compression Guard", isOn: $vm.compressionProtection)
                    Toggle("Sharpie Outline Subtle", isOn: $vm.outlineProtection)
                    Text("Source frames are cleaned first, then processed by our 1× Anime Sharpie Outline Subtle Core ML model before RIFE. Generated 60 fps frames inherit the processed line style without running the outline model again.")
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
                        ProgressView(value: vm.progress)
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
        }
    }
}
