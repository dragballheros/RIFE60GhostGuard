import AVFoundation
import CoreTransferable
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

private struct H3PickedVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent("MiniMaxH3-Source-\(UUID().uuidString).\(ext)")
            try FileManager.default.copyItem(at: received.file, to: destination)
            return H3PickedVideo(url: destination)
        }
    }
}

private struct H3DownloadStatus: View {
    @ObservedObject var downloader: H3ModelDownloader

    var body: some View {
        switch downloader.state {
        case .idle:
            if !downloader.tapNote.isEmpty {
                Text(downloader.tapNote).font(.caption)
            }
        case .listing:
            ProgressView()
            Text(downloader.tapNote.isEmpty ? "Checking the H3 manifest…" : downloader.tapNote)
                .font(.caption)
        case .ready:
            Text(downloader.totalFiles > 0
                 ? "Official file list is ready: \(downloader.totalFiles) files, about \(String(format: "%.1f", downloader.totalGiB)) GiB. Tier 1 download uses the manifest URL above, not this full tree."
                 : "Ready.")
                .font(.caption)
        case .downloading:
            ProgressView(value: downloader.fileTotalBytes > 0 ? min(1, Double(downloader.fileBytes) / Double(downloader.fileTotalBytes)) : downloader.progress)
            Text("Shard \(max(downloader.shardIndex, 1)) of \(max(downloader.totalFiles, 1)): \(downloader.currentFile.isEmpty ? "starting" : downloader.currentFile)")
                .font(.caption)
            Text("\(downloader.fileBytes) / \(downloader.fileTotalBytes) bytes in this file · \(downloader.downloadedBytes) / \(downloader.totalBytes) bytes overall")
                .font(.caption).foregroundStyle(.secondary)
            if !downloader.tapNote.isEmpty {
                Text(downloader.tapNote).font(.caption)
            }
        case .completed:
            Text(downloader.tapNote.isEmpty ? "Tier 1 download finished." : downloader.tapNote)
                .font(.caption)
        case .failed(let message):
            Text(message)
                .foregroundStyle(.red)
                .copyPasteBar(text: message) { _ in }
        }
    }
}

private enum LocalVideoBackend: String, CaseIterable, Hashable {
    case minimaxH3 = "MiniMax H3"
}

/// Reference-conditioned regeneration. This mode does not use the VACE mask and does not claim pixel-mask preservation.
struct H3VideoEditView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var videoItem: PhotosPickerItem?
    @State private var sourceURL: URL?
    @State private var prompt = "Describe the desired change while keeping the overall scene coherent."
    @State private var isRunning = false
    @State private var progress = 0.0
    @State private var status = "Choose a reference video."
    @State private var errorText: String?
    @State private var outputURL: URL?
    @State private var selectedBackend: LocalVideoBackend = .minimaxH3
    @State private var h3ModelDirectoryURL: URL?
    @State private var isChoosingH3Model = false
    @State private var acceptedH3License = false
    @AppStorage(H3Tier1OnDeviceConverter.manifestDefaultsKey) private var tier1ManifestURL = ""
    @State private var h3CancellationToken: H3CancellationToken?
    @StateObject private var h3ModelDownloader = H3ModelDownloader()

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    ForEach(LocalVideoBackend.allCases, id: \.self) { backend in
                        Button {
                            selectedBackend = backend
                        } label: {
                            HStack {
                                Image(systemName: selectedBackend == backend ? "checkmark.circle.fill" : "circle")
                                Text(backend.rawValue)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                    Text("The selected video is supplied as an H3 video reference. H3 may regenerate content outside any intended region; this mode does not use the VACE mask.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button {
                        isChoosingH3Model = true
                    } label: {
                        Label(h3ModelDirectoryURL == nil ? "Choose local H3 model folder" : "Change local H3 model folder", systemImage: "folder")
                    }
                    if let h3ModelDirectoryURL {
                        Text(h3ModelDirectoryURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                    Link("Review the official MiniMax H3 model license", destination: H3ModelDownloader.licenseURL)
                    Text("Territory restriction: the published Community License excludes the United States, European Union, United Kingdom, and South Korea from its standard applicable territory. If you are in an excluded territory, do not download or use these weights unless MiniMax has separately authorized you. Check the current license before proceeding.")
                        .font(.caption).foregroundStyle(.red)
                    Toggle("I have verified that my use is authorized by the license, including separate authorization if my territory is excluded", isOn: $acceptedH3License)
                    Text("Tier 1 stays on this phone: one published compressed shard at a time, about 30 GB kept, scratch deleted before the next file. No paid cloud, no encoder-student distillation, and no DMD training.")
                        .font(.caption).foregroundStyle(.secondary)
                    TextField("Tier 1 manifest URL (https)", text: $tier1ManifestURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .copyPasteBar(text: tier1ManifestURL) { tier1ManifestURL = $0 }
                    Button("Check official H3 model files") { h3ModelDownloader.prepare() }
                        .disabled(h3ModelDownloader.state == .listing || h3ModelDownloader.state == .downloading)
                    Button("Download / resume local H3 model") {
                        guard acceptedH3License else {
                            h3ModelDownloader.reportBlocked("Accept the license statement above before this phone downloads the local H3 model.")
                            return
                        }
                        h3ModelDownloader.startDownload()
                    }
                    .disabled(isRunning)
                    H3DownloadStatus(downloader: h3ModelDownloader)
                }
                Section("Reference video") {
                    PhotosPicker(selection: $videoItem, matching: .videos) {
                        Label(sourceURL == nil ? "Choose reference video" : "Change reference video", systemImage: "film")
                    }
                    Text("Official Ref2VA accepts reference clips of 2–15 seconds.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Generation prompt") {
                    TextField("Positive prompt", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                        .copyPasteBar(text: prompt) { prompt = $0 }
                    Text("The current native H3 API does not expose a separate negative-prompt input. Put exclusions directly into the main prompt.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section {
                    if isRunning {
                        ProgressView(value: progress)
                        Button("Cancel H3 generation") { h3CancellationToken?.cancel() }
                    }
                    Button { Task { await runH3ReferenceEdit() } } label: {
                        Label("Generate H3 reference video", systemImage: "wand.and.stars")
                    }
                    .disabled(!acceptedH3License || isRunning || sourceURL == nil || h3ModelDirectoryURL == nil)
                    if let errorText {
                        Text(errorText).foregroundStyle(.red)
                            .copyPasteBar(text: errorText) { _ in }
                    }
                    Text(status).font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Local MiniMax H3")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
            .fileImporter(isPresented: $isChoosingH3Model, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
                if case .success(let urls) = result { h3ModelDirectoryURL = urls.first }
            }
            .onChange(of: videoItem) { item in
                guard let item else { return }
                Task {
                    if let picked = try? await item.loadTransferable(type: H3PickedVideo.self) {
                        await MainActor.run { sourceURL = picked.url }
                    }
                }
            }
        }
    }

    @MainActor
    private func runH3ReferenceEdit() async {
        guard let sourceURL else { errorText = "Choose a source video first."; return }
        guard let modelURL = h3ModelDirectoryURL else {
            errorText = "Choose a local MiniMax H3 model folder first."
            return
        }
        guard #available(iOS 18.0, *) else {
            errorText = "MiniMax H3 requires iOS 18.0 or newer."
            return
        }
        isRunning = true
        progress = 0
        outputURL = nil
        errorText = nil
        let token = H3CancellationToken()
        h3CancellationToken = token
        var h3OutputCandidate: URL?
        defer { isRunning = false; h3CancellationToken = nil }
        let hasSecurityScope = modelURL.startAccessingSecurityScopedResource()
        defer { if hasSecurityScope { modelURL.stopAccessingSecurityScopedResource() } }
        do {
            let referenceAsset = AVURLAsset(url: sourceURL)
            let referenceDuration = try await referenceAsset.load(.duration)
            let referenceTracks = try await referenceAsset.loadTracks(withMediaType: .video)
            guard !referenceTracks.isEmpty,
                  referenceDuration.seconds.isFinite,
                  (2.0...15.0).contains(referenceDuration.seconds) else {
                throw H3NativeRuntime.RuntimeError.invalidRequest(
                    "H3 Ref2VA requires a playable reference video between 2 and 15 seconds. Trim the clip and retry."
                )
            }
            let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("MiniMaxH3Exports", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent("MiniMaxH3-\(UUID().uuidString).mov")
            h3OutputCandidate = destination
            let requestPrompt = prompt
            let result = try await Task.detached(priority: .userInitiated) {
                let runtime = try H3NativeRuntime.load(modelDirectory: modelURL)
                return try runtime.generateReferenceConditionedVideo(
                    prompt: requestPrompt,
                    referenceVideoURL: sourceURL,
                    outputURL: destination,
                    width: 448,
                    height: 256,
                    frames: 56,
                    steps: 20,
                    seed: 42,
                    cancellationToken: token,
                    onProgress: { phase, fraction in
                        Task { @MainActor in
                            progress = max(0, min(1, fraction))
                            status = "H3 \(phase): \(Int(fraction * 100))%"
                        }
                    }
                )
            }.value
            let outputAttributes = try? FileManager.default.attributesOfItem(atPath: result.outputURL.path)
            let outputSize = (outputAttributes?[.size] as? NSNumber)?.intValue ?? 0
            guard FileManager.default.fileExists(atPath: result.outputURL.path), outputSize > 0 else {
                throw H3NativeRuntime.RuntimeError.generationFailed("H3 returned without creating a non-empty output video.")
            }
            let exportedAsset = AVURLAsset(url: result.outputURL)
            let exportedVideoTracks = try await exportedAsset.loadTracks(withMediaType: .video)
            let exportedDuration = try await exportedAsset.load(.duration)
            guard !exportedVideoTracks.isEmpty, exportedDuration.seconds.isFinite, exportedDuration.seconds > 0 else {
                throw H3NativeRuntime.RuntimeError.generationFailed("H3 output exists but does not contain a playable video track with a valid duration.")
            }
            outputURL = result.outputURL
            progress = 1
            status = "H3 reference-conditioned MOV created."
        } catch {
            if let h3OutputCandidate { try? FileManager.default.removeItem(at: h3OutputCandidate) }
            errorText = error.localizedDescription
            status = token.isCancelled
                ? "H3 generation cancelled. Partial output was discarded."
                : "H3 stopped without exporting a substitute video."
        }
    }
}
