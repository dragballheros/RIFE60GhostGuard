import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var vm = VideoProcessorViewModel()
    @State private var showingImporter = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Input") {
                    Button {
                        showingImporter = true
                    } label: {
                        Label(vm.inputURL?.lastPathComponent ?? "Select Video", systemImage: "film")
                    }
                    if let input = vm.inputURL {
                        Text(input.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                }

                Section("RIFE") {
                    Picker("Quality", selection: $vm.quality) {
                        ForEach(RIFEQualityChoice.allCases) { q in
                            Text(q.title).tag(q)
                        }
                    }
                    Toggle("Ghost protection", isOn: $vm.ghostProtection)
                    Toggle("Scene-cut protection", isOn: $vm.sceneCutProtection)
                    LabeledContent("Target", value: "60.00 fps")
                }

                Section("Ghost Guard") {
                    VStack(alignment: .leading, spacing: 5) {
                        HStack { Text("Sensitivity"); Spacer(); Text(String(format: "%.2f", vm.ghostSensitivity)) }
                        Slider(value: $vm.ghostSensitivity, in: 0.65...1.35, step: 0.05)
                    }
                    Text("Rejected synthetic frames are replaced by the nearest real source frame instead of being repaired by another generated frame.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Export") {
                    Picker("Codec", selection: $vm.codec) {
                        ForEach(OutputCodec.allCases) { c in Text(c.title).tag(c) }
                    }
                    Toggle("Preserve original audio", isOn: $vm.preserveAudio)
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
                        Text(out.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                }

                if let err = vm.errorText {
                    Section("Error") { Text(err).foregroundStyle(.red) }
                }
            }
            .navigationTitle("RIFE 60")
            .fileImporter(isPresented: $showingImporter,
                          allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie],
                          allowsMultipleSelection: false) { result in
                vm.handleImport(result)
            }
        }
    }
}
