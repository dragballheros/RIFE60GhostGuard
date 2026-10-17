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

/// A real local video-edit path. A still mask is propagated across the clip using
/// on-device Vision optical flow; each 5-frame crop is processed by native Wan VACE.
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

    var body: some View {
        NavigationStack {
            Form {
                Section("Source video") {
                    PhotosPicker(selection: $videoItem, matching: .videos) {
                        Label(sourceURL == nil ? "Choose video from Photos" : "Change source video",
                              systemImage: "film")
                    }
                    if let sourceURL {
                        Text(sourceURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                    PhotosPicker(selection: $maskItem, matching: .images) {
                        Label(maskURL == nil ? "Choose edit mask image" : "Change edit mask",
                              systemImage: "paintbrush.pointed")
                    }
                    if let maskURL {
                        Text(maskURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Use a black-and-white mask with white over the area to change and black everywhere to preserve. The mask is tracked locally with optical flow; hard scene cuts stop propagation.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("Generation prompt") {
                    TextField("Positive prompt", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                    TextField("Negative prompt", text: $negativePrompt, axis: .vertical)
                        .lineLimit(2...4)
                }

                Section("Explicit model / resolution") {
                    LabeledContent("Local model", value: "Wan 2.1 VACE 1.3B Q8")
                    LabeledContent("Inference crop", value: "512 × 512")
                    LabeledContent("Denoising steps", value: "20")
                    Text("This is an experimental 1.3B profile using Draw Things' native Metal runtime. It is not the requested 14B / 720p model and does not claim equivalent quality. The 14B checkpoint cannot fit in this phone's unified memory.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("Weights are downloaded as local model files when needed. Generation itself always uses the local backend; the sequence API rejects remote inference.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("Run") {
                    Button {
                        Task { await runEdit() }
                    } label: {
                        if isRunning {
                            HStack { ProgressView(); Text("Running local Wan VACE…") }
                        } else {
                            Label("Generate masked video edit", systemImage: "wand.and.stars")
                        }
                    }
                    .disabled(isRunning || sourceURL == nil || maskURL == nil ||
                              prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if isRunning {
                        ProgressView(value: progress)
                        Text(status).font(.caption).foregroundStyle(.secondary)
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
            .navigationTitle("Local Wan VACE Edit")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.disabled(isRunning)
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
