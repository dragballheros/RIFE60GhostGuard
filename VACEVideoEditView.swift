import AVFoundation
import CoreTransferable
import ImageIO
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// Copies a selected movie out of the Photos item provider's temporary transfer location.
private struct VACEPickedVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent("WanVACE-Source-\\(UUID().uuidString).\\(ext)")
            try FileManager.default.copyItem(at: received.file, to: destination)
            return VACEPickedVideo(url: destination)
        }
    }
}

/// Copies a selected mask image to a stable local URL without relying on a Photos-only URL.
private struct VACEPickedMaskImage: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .image) { received in
            let ext = received.file.pathExtension.isEmpty ? "png" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent("WanVACE-Mask-\\(UUID().uuidString).\\(ext)")
            try FileManager.default.copyItem(at: received.file, to: destination)
            return VACEPickedMaskImage(url: destination)
        }
    }
}

private enum VACEVideoEditInputError: LocalizedError {
    case missingVideo
    case missingMask
    case invalidVideoDimensions
    case noEditableMaskPixels
    case unreadableMask

    var errorDescription: String? {
        switch self {
        case .missingVideo: return "Choose a source video first."
        case .missingMask: return "Choose a black-and-white edit mask. White means regenerate; black means preserve."
        case .invalidVideoDimensions: return "The source video dimensions could not be read."
        case .noEditableMaskPixels: return "The selected mask contains no white/editable pixels."
        case .unreadableMask: return "The selected mask image could not be decoded."
        }
    }
}

private enum LocalVideoBackend: String, CaseIterable, Hashable {
    case wanVACE = "Wan VACE"
    case minimaxH3 = "MiniMax H3"
}

/// Local video workflows. Wan VACE performs mask-guided edits; MiniMax H3 performs
/// reference-conditioned regeneration and does not claim pixel-mask preservation.
struct VACEVideoEditView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var videoItem: PhotosPickerItem?
    @State private var maskItem: PhotosPickerItem?
    @State private var sourceURL: URL?
    @State private var maskURL: URL?
    @State private var prompt = "Replace the white masked area with a coherent natural-looking alternative that matches the scene."
    @State private var negativePrompt = "flicker, inconsistent subject, warped geometry, duplicate objects, text, watermark"
    @State private var isRunning = false
    @State private var progress = 0.0
    @State private var status = "Choose a video and a matching white-on-black edit mask."
    @State private var errorText: String?
    @State private var outputURL: URL?
    @State private var selectedBackend: LocalVideoBackend = .wanVACE
    @State private var h3ModelDirectoryURL: URL?
    @State private var isChoosingH3Model = false
    @State private var h3CancellationToken: H3CancellationToken?

    var body: some View {
        NavigationStack {
            Form {
                Section("Local model") {
                    ForEach(LocalVideoBackend.allCases, id: \.self) { backend in
                        Button {
                            selectedBackend = backend
                            outputURL = nil
                            errorText = nil
                            status = backend == .wanVACE
                                ? "Wan VACE uses the selected white-on-black mask."
                                : "MiniMax H3 regenerates from the reference video. It does not preserve black-mask pixels."
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: selectedBackend == backend ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(selectedBackend == backend ? Color.accentColor : Color.secondary)
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(backend.rawValue).foregroundStyle(.primary)
                                    Text(backend == .wanVACE
                                         ? "Mask-guided local editing"
                                         : "Local reference-conditioned video regeneration")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(isRunning)
                    }
                    if selectedBackend == .minimaxH3 {
                        Button {
                            isChoosingH3Model = true
                        } label: {
                            Label(h3ModelDirectoryURL == nil ? "Choose local H3 model folder" : "Change local H3 model folder", systemImage: "folder")
                        }
                        .disabled(isRunning)
                        if let h3ModelDirectoryURL {
                            Text(h3ModelDirectoryURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                        }
                        Text("The model must already be installed locally in the native H3 runtime's expected folder layout. Model files are not included in the IPA. H3 requires iOS 18 or newer; successful loading does not guarantee the phone has enough working memory to finish inference.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                Section("Source video") {
                    PhotosPicker(selection: $videoItem, matching: .videos) {
                        Label(sourceURL == nil ? "Choose video from Photos" : "Change source video",
                              systemImage: "film")
                    }
                    .disabled(isRunning)
                    if let sourceURL {
                        Text(sourceURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                    if selectedBackend == .wanVACE {
                        PhotosPicker(selection: $maskItem, matching: .images) {
                            Label(maskURL == nil ? "Choose edit mask image" : "Change edit mask",
                                  systemImage: "paintbrush.pointed")
                        }
                        .disabled(isRunning)
                        if let maskURL {
                            Text(maskURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                        }
                        Text("Use a black-and-white mask with white over the area to change and black everywhere to preserve. The mask is tracked locally with optical flow; hard scene cuts stop propagation.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("The selected video is supplied as an H3 video reference. H3 may regenerate content outside any intended region; this mode does not use the VACE mask.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                Section("Generation prompt") {
                    TextField("Positive prompt", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                    TextField("Negative prompt", text: $negativePrompt, axis: .vertical)
                        .lineLimit(2...4)
                }

                Section("Explicit model / resolution") {
                    if selectedBackend == .wanVACE {
                        LabeledContent("Local model", value: "Wan 2.1 VACE 1.3B Q8")
                        LabeledContent("Inference crop", value: "512 × 512")
                        LabeledContent("Denoising steps", value: "20")
                        Text("This is an experimental 1.3B profile using Draw Things' native Metal runtime. It is not the requested 14B / 720p model and does not claim equivalent quality.")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        LabeledContent("Local model", value: "MiniMax H3 native Metal runtime")
                        LabeledContent("Output canvas", value: "448 × 256")
                        LabeledContent("Frames / steps", value: "56 / 20")
                        Text("The H3 workflow uses the source video as a reference and exports a newly generated MOV. It is not exact masked inpainting. The native engine and checkpoint must be compatible; memory and thermal feasibility still require testing on this iPhone.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Generation runs locally. There is no remote inference or fallback to another backend.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("Run") {
                    Button {
                        Task {
                            if selectedBackend == .wanVACE {
                                await runEdit()
                            } else {
                                await runH3ReferenceEdit()
                            }
                        }
                    } label: {
                        if isRunning {
                            HStack { ProgressView(); Text(selectedBackend == .wanVACE ? "Running local Wan VACE…" : "Running local MiniMax H3…") }
                        } else {
                            Label(selectedBackend == .wanVACE ? "Generate masked video edit" : "Generate H3 reference video", systemImage: "wand.and.stars")
                        }
                    }
                    .disabled(isRunning || sourceURL == nil ||
                              (selectedBackend == .wanVACE && maskURL == nil) ||
                              (selectedBackend == .minimaxH3 && h3ModelDirectoryURL == nil) ||
                              prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if isRunning {
                        ProgressView(value: progress)
                        Text(status).font(.caption).foregroundStyle(.secondary)
                        if selectedBackend == .minimaxH3 {
                            Button {
                                h3CancellationToken?.cancel()
                                status = "Cancellation requested. Waiting for the native H3 engine to reach its next progress checkpoint…"
                            } label: {
                                Label("Cancel H3 generation", systemImage: "xmark.circle")
                            }
                            .disabled(h3CancellationToken == nil)
                        }
                    } else {
                        Text(status).font(.caption).foregroundStyle(.secondary)
                    }
                    if let errorText {
                        Text(errorText).font(.caption).foregroundStyle(.red)
                    }
                    if let outputURL {
                        LabeledContent("Export", value: outputURL.lastPathComponent)
                        ShareLink(item: outputURL) {
                            Label("Share generated MOV", systemImage: "square.and.arrow.up")
                        }
                    }
                }
            }
            .navigationTitle(selectedBackend == .wanVACE ? "Local Wan VACE Edit" : "Local MiniMax H3")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.disabled(isRunning)
                }
            }
            .fileImporter(isPresented: $isChoosingH3Model, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
                switch result {
                case .success(let urls):
                    h3ModelDirectoryURL = urls.first
                    outputURL = nil
                    errorText = nil
                    status = urls.first == nil ? "No H3 model folder selected." : "Selected local H3 model folder. Native loading will validate its contents."
                case .failure(let error):
                    errorText = error.localizedDescription
                }
            }
            .onChange(of: videoItem) { item in
                guard let item else { return }
                Task {
                    do {
                        let picked = try await item.loadTransferable(type: VACEPickedVideo.self)
                        await MainActor.run {
                            sourceURL = picked?.url
                            outputURL = nil
                            errorText = nil
                        }
                    } catch {
                        await MainActor.run { errorText = error.localizedDescription }
                    }
                }
            }
            .onChange(of: maskItem) { item in
                guard let item else { return }
                Task {
                    do {
                        let picked = try await item.loadTransferable(type: VACEPickedMaskImage.self)
                        await MainActor.run {
                            maskURL = picked?.url
                            outputURL = nil
                            errorText = nil
                        }
                    } catch {
                        await MainActor.run { errorText = error.localizedDescription }
                    }
                }
            }
        }
    }

    @MainActor
    private func runEdit() async {
        guard let sourceURL else { errorText = VACEVideoEditInputError.missingVideo.localizedDescription; return }
        guard let maskURL else { errorText = VACEVideoEditInputError.missingMask.localizedDescription; return }
        isRunning = true
        progress = 0
        outputURL = nil
        errorText = nil
        status = "Preparing the local Wan VACE model…"
        defer { isRunning = false }

        do {
            let asset = AVURLAsset(url: sourceURL)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let videoTrack = tracks.first else { throw VACEVideoPipelineError.noVideoTrack }
            let naturalSize = try await videoTrack.load(.naturalSize)
            let width = Int(abs(naturalSize.width).rounded())
            let height = Int(abs(naturalSize.height).rounded())
            guard width > 0, height > 0 else { throw VACEVideoEditInputError.invalidVideoDimensions }
            let seedMask = try Self.loadMask(url: maskURL, width: width, height: height)
            guard seedMask.activePixelCount > 0 else { throw VACEVideoEditInputError.noEditableMaskPixels }

            let engine = DrawThingsLocalVACEReconstructionEngine()
            try await engine.prepare { file, written, expected in
                let amount = expected > 0 ? "\(Int(Double(written) * 100 / Double(expected)))%" : "preparing"
                Task { @MainActor in status = "Preparing local model assets: \(file) (\(amount))" }
            }

            var options = VACESelectiveVideoOptions()
            options.temporalWindowLength = 5
            options.temporalOverlap = 2
            options.maximumCropWidth = 832
            options.maximumCropHeight = 480
            options.tileOverlapPixels = 64

            let provider = VACEOpticalFlowMaskProvider(firstFrameMask: seedMask)
            let processor = VACESelectiveVideoPipeline(
                maskProvider: provider, reconstructionEngine: engine, options: options
            )
            let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("WanVACEExports", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent("WanVACE-\(UUID().uuidString).mov")
            status = "Running real local VACE inference on temporal windows…"
            try await processor.process(
                sourceURL: sourceURL,
                destinationURL: destination,
                prompt: prompt,
                negativePrompt: negativePrompt
            ) { fraction, message in
                Task { @MainActor in
                    progress = max(progress, min(1, max(0, fraction)))
                    status = message
                }
            }
            outputURL = destination
            progress = 1
            status = "Generated MOV saved. Output uses the source video's dimensions; model crops were inferred at 512 × 512."
        } catch {
            errorText = error.localizedDescription
            status = "Stopped without exporting a substitute video."
        }
    }

    @MainActor
    private func runH3ReferenceEdit() async {
        guard let sourceURL else { errorText = VACEVideoEditInputError.missingVideo.localizedDescription; return }
        guard let modelURL = h3ModelDirectoryURL else {
            errorText = "Choose a local MiniMax H3 model folder first."
            return
        }
        guard #available(iOS 18.0, *) else {
            errorText = "MiniMax H3 requires iOS 18.0 or newer. Wan VACE remains available."
            return
        }

        isRunning = true
        progress = 0
        outputURL = nil
        errorText = nil
        status = "Checking local H3 model and reference video…"
        let token = H3CancellationToken()
        h3CancellationToken = token
        var h3OutputCandidate: URL?
        defer {
            isRunning = false
            h3CancellationToken = nil
        }

        let hasSecurityScope = modelURL.startAccessingSecurityScopedResource()
        defer { if hasSecurityScope { modelURL.stopAccessingSecurityScopedResource() } }

        do {
            let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("MiniMaxH3Exports", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent("MiniMaxH3-\\(UUID().uuidString).mov")
            h3OutputCandidate = destination
            status = "Loading the native H3 runtime locally. Large allocations may fail on this device."
            let requestPrompt = prompt
            let modelDirectory = modelURL
            let referenceVideo = sourceURL
            let result = try await Task.detached(priority: .userInitiated) {
                let runtime = try H3NativeRuntime.load(modelDirectory: modelDirectory)
                return try runtime.generateReferenceConditionedVideo(
                    prompt: requestPrompt,
                    referenceVideoURL: referenceVideo,
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
                            status = "H3 \\(phase): \\(Int(fraction * 100))%"
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
            status = "H3 reference-conditioned MOV created (\\(result.width) × \\(result.height), \\(result.frames) frames, seed \\(result.seed)). Verify playback and quality on-device."
        } catch {
            if let h3OutputCandidate {
                try? FileManager.default.removeItem(at: h3OutputCandidate)
            }
            errorText = error.localizedDescription
            status = token.isCancelled
                ? "H3 generation cancelled. Partial output was discarded; Wan VACE remains available."
                : "H3 stopped without exporting a substitute video. Wan VACE remains available."
        }
    }

    private static func loadMask(url: URL, width: Int, height: Int) throws -> VACEMaskPlane {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height)
        ]
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw VACEVideoEditInputError.unreadableMask
        }
        var pixels = [UInt8](repeating: 0, count: width * height)
        let success = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard success else { throw VACEVideoEditInputError.unreadableMask }
        return try VACEMaskPlane(width: width, height: height, pixels: pixels)
    }
}
