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
                .appendingPathComponent("WanVACE-Source-\(UUID().uuidString).\(ext)")
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
                .appendingPathComponent("WanVACE-Mask-\(UUID().uuidString).\(ext)")
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

/// Mask-guided local editing with Wan VACE. MiniMax H3 stays in H3VideoEditView.
struct VACEVideoEditView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var videoItem: PhotosPickerItem?
    @State private var maskItem: PhotosPickerItem?
    @State private var sourceURL: URL?
    @State private var maskURL: URL?
    @State private var maskMode: VACEMaskMode = .autoSubject
    @State private var maskPreview: UIImage?
    @State private var maskCoverageText: String?
    @State private var maskPreviewNote: String?
    @State private var prompt = "Describe the desired change while keeping the overall scene coherent."
    @State private var negativePrompt = "flicker, inconsistent subject, warped geometry, duplicate objects, text, watermark"
    @State private var isRunning = false
    @State private var progress = 0.0
    @State private var status = "Choose a video. Auto (subject) builds the mask on this iPhone."
    @State private var errorText: String?
    @State private var outputURL: URL?
    @State private var previewTime: Double = 0
    @State private var previewDuration: Double = 1
    @State private var brushErase = false
    @State private var manualMask: VACEMaskPlane?
    @State private var previewFrame: CGImage?

    private var generateDisableReasons: [String] {
        var reasons: [String] = []
        if sourceURL == nil { reasons.append("source video") }
        if maskMode.needsHandDrawnMask && maskURL == nil && manualMask == nil { reasons.append("a tap or brush stroke") }
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { reasons.append("prompt text") }
        return reasons
    }
    var body: some View {
        NavigationStack {
            Form {
                Section("Source video") {
                    PhotosPicker(selection: $videoItem, matching: .videos) {
                        Label(sourceURL == nil ? "Choose video from Photos" : "Change source video",
                              systemImage: "film")
                    }
                    .disabled(isRunning)
                    if let sourceURL {
                        Text(sourceURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    }
                    Picker("Mask", selection: $maskMode) {
                        ForEach(VACEMaskMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .disabled(isRunning)
                    Text(maskMode.caption)
                        .font(.caption).foregroundStyle(.secondary)
                    if maskMode == .drawManually {
                        Text("Scrub to a frame, tap the character, or brush what should change.")
                            .font(.caption).foregroundStyle(.secondary)
                        Picker("Tool", selection: $brushErase) {
                            Text("Brush").tag(false)
                            Text("Erase").tag(true)
                        }
                        .pickerStyle(.segmented)
                        .disabled(isRunning)
                        PhotosPicker(selection: $maskItem, matching: .images) {
                            Label(maskURL == nil ? "Import a mask image" : "Change imported mask",
                                  systemImage: "photo")
                        }
                        .disabled(isRunning)
                    }
                    if sourceURL != nil {
                        Slider(value: $previewTime, in: 0...max(previewDuration, 0.1)) { editing in
                            if !editing { Task { await refreshMaskPreview() } }
                        }
                        .disabled(isRunning)
                        Text(String(format: "Preview %.1f s", previewTime))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let maskPreview {
                        maskPreviewImage(maskPreview)
                        if let maskCoverageText {
                            Text(maskCoverageText)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    if let maskPreviewNote {
                        Text(maskPreviewNote).font(.caption).foregroundStyle(.orange)
                    }
                }

                Section("Generation prompt") {
                    TextField("Positive prompt", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                        .copyPasteBar(text: prompt) { prompt = $0 }
                    TextField("Negative prompt", text: $negativePrompt, axis: .vertical)
                        .lineLimit(2...4)
                        .copyPasteBar(text: negativePrompt) { negativePrompt = $0 }
                }

                Section("Explicit model / resolution") {
                    LabeledContent("Local model", value: "Wan 2.1 VACE 1.3B Q8")
                    LabeledContent("Inference crop", value: "512 × 512")
                    LabeledContent("Denoising steps", value: "20")
                    Text("This is an experimental 1.3B profile using Draw Things' native Metal runtime. It is not the requested 14B / 720p model and does not claim equivalent quality.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("Generation runs locally. There is no remote inference or fallback to another backend.")
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
                    .disabled(isRunning || !generateDisableReasons.isEmpty)
                    if !generateDisableReasons.isEmpty && !isRunning {
                        Text("Needs: \(generateDisableReasons.joined(separator: ", "))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if isRunning {
                        ProgressView(value: progress)
                    }
                    Text(status).font(.caption).foregroundStyle(.secondary)
                    if let errorText {
                        Text(errorText).font(.caption).foregroundStyle(.red)
                            .copyPasteBar(text: errorText) { _ in }
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
            .onChange(of: maskMode) { _ in
                maskPreview = nil
                maskCoverageText = nil
                maskPreviewNote = nil
                Task { await refreshMaskPreview() }
            }
            .onChange(of: videoItem) { item in
                guard let item else { return }
                Task {
                    do {
                        let picked = try await item.loadTransferable(type: VACEPickedVideo.self)
                        let url = picked?.url
                        await MainActor.run {
                            sourceURL = url
                            outputURL = nil
                            errorText = nil
                            maskPreview = nil
                            maskCoverageText = nil
                            maskPreviewNote = nil
                        }
                        await refreshMaskPreview()
                    } catch {
                        CrashCapture.step("VACE source import failed: \(error.localizedDescription)")
                        await MainActor.run { errorText = error.localizedDescription }
                    }
                }
            }
            .onChange(of: maskItem) { item in
                guard let item else { return }
                Task {
                    do {
                        let picked = try await item.loadTransferable(type: VACEPickedMaskImage.self)
                        let url = picked?.url
                        await MainActor.run {
                            maskURL = url
                            outputURL = nil
                            errorText = nil
                        }
                        await refreshMaskPreview()
                    } catch {
                        CrashCapture.step("VACE mask import failed: \(error.localizedDescription)")
                        await MainActor.run { errorText = error.localizedDescription }
                    }
                }
            }
        }
    }

    @MainActor
    private func runEdit() async {
        guard let sourceURL else { errorText = VACEVideoEditInputError.missingVideo.localizedDescription; return }
        if maskMode.needsHandDrawnMask {
            guard maskURL != nil || manualMask != nil else { errorText = VACEVideoEditInputError.missingMask.localizedDescription; return }
        }
        isRunning = true
        progress = 0
        outputURL = nil
        errorText = nil
        CrashCapture.step("VACE mask mode: \(maskMode.rawValue)")
        status = "Preparing the local Wan VACE model…"
        defer { isRunning = false }

        let autoProvider = VACEAutoMaskProvider(mode: maskMode)
        do {
            let asset = AVURLAsset(url: sourceURL)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let videoTrack = tracks.first else { throw VACEVideoPipelineError.noVideoTrack }
            let naturalSize = try await videoTrack.load(.naturalSize)
            let width = Int(abs(naturalSize.width).rounded())
            let height = Int(abs(naturalSize.height).rounded())
            guard width > 0, height > 0 else { throw VACEVideoEditInputError.invalidVideoDimensions }
            let provider: VACEFrameMaskProvider
            if maskMode == .drawManually {
                let seedMask: VACEMaskPlane
                if let manualMask {
                    seedMask = manualMask
                } else if let maskURL {
                    seedMask = try Self.loadMask(url: maskURL, width: width, height: height)
                } else {
                    throw VACEVideoEditInputError.missingMask
                }
                guard seedMask.activePixelCount > 0 else { throw VACEVideoEditInputError.noEditableMaskPixels }
                provider = VACEManualTrackMaskProvider(seed: seedMask)
            } else {
                provider = autoProvider
            }

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
            if autoProvider.fellBackToWholeFrame {
                status = "No subject found, so the whole frame was edited. Generated MOV saved at the source dimensions; model crops were inferred at 512 × 512."
                maskPreviewNote = "No subject found. Using the whole frame."
            } else {
                status = "Generated MOV saved. Output uses the source video's dimensions; model crops were inferred at 512 × 512."
            }
        } catch {
            CrashCapture.step("VACE edit failed: \(error.localizedDescription)")
            errorText = error.localizedDescription
            status = "Stopped without exporting a substitute video."
        }
    }

    @MainActor
    private func refreshMaskPreview() async {
        guard let sourceURL else {
            maskPreview = nil
            maskCoverageText = nil
            maskPreviewNote = nil
            previewFrame = nil
            return
        }
        let mode = maskMode
        let manualURL = maskURL
        let seconds = previewTime
        let painted = manualMask
        do {
            let asset = AVURLAsset(url: sourceURL)
            let duration = try await asset.load(.duration).seconds
            if duration.isFinite, duration > 0 {
                previewDuration = duration
            }
            let frame = try await VACESubjectSegmenter.frameImage(url: sourceURL, at: seconds)
            previewFrame = frame
            let built: VACESegmentationDecision
            if mode == .drawManually, let painted {
                let point = Self.centroid(of: painted)
                if abs(seconds) < 0.05 {
                    let fraction = VACESubjectSegmenter.coverage(of: painted.pixels)
                    built = VACESegmentationDecision(mask: painted, foundSubject: fraction > 0, method: "tap-select", coverage: fraction)
                } else if let tracked = try? VACESubjectSegmenter.tapSelect(cgImage: frame, x: point.x * frame.width / max(painted.width, 1), y: point.y * frame.height / max(painted.height, 1)), tracked.foundSubject {
                    built = VACESegmentationDecision(mask: tracked.mask, foundSubject: true, method: "manual-tracked", coverage: tracked.coverage)
                } else {
                    let fraction = VACESubjectSegmenter.coverage(of: painted.pixels)
                    built = VACESegmentationDecision(mask: painted, foundSubject: fraction > 0, method: "manual-tracked", coverage: fraction)
                }
            } else if mode == .drawManually, let manualURL {
                let plane = try Self.loadMask(url: manualURL, width: frame.width, height: frame.height)
                let fraction = VACESubjectSegmenter.coverage(of: plane.pixels)
                built = VACESegmentationDecision(mask: plane, foundSubject: fraction > 0, method: "manual-tracked", coverage: fraction)
            } else if mode == .wholeFrame {
                let plane = try VACEMaskPlane(width: frame.width, height: frame.height, pixels: [UInt8](repeating: 255, count: frame.width * frame.height))
                built = VACESegmentationDecision(mask: plane, foundSubject: true, method: "whole-frame", coverage: 1)
            } else {
                built = try VACESubjectSegmenter.editMask(cgImage: frame, mode: mode)
            }
            var note: String?
            var mask = built.mask
            var method = built.method
            var fraction = built.coverage
            if mode != .wholeFrame && mode != .drawManually && !built.foundSubject {
                CrashCapture.step("VACE segmentation found nothing; using whole frame")
                mask = try VACEMaskPlane(width: frame.width, height: frame.height, pixels: [UInt8](repeating: 255, count: frame.width * frame.height))
                method = "whole-frame"
                fraction = 1
                note = "No subject found. Using the whole frame."
            }
            maskPreview = VACESubjectSegmenter.previewImage(base: frame, mask: mask)
            maskCoverageText = String(format: "Coverage %.0f%% · %@", fraction * 100, VACEMaskPreviewLabel.displayName(for: method))
            maskPreviewNote = note
        } catch {
            CrashCapture.step("VACE segmentation failed: \(error.localizedDescription)")
            maskPreview = nil
            maskCoverageText = nil
            maskPreviewNote = "Mask preview failed: \(error.localizedDescription)"
        }
    }

    @ViewBuilder
    private func maskPreviewImage(_ image: UIImage) -> some View {
        GeometryReader { geo in
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(width: geo.size.width, height: geo.size.height)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            guard maskMode == .drawManually, let frame = previewFrame else { return }
                            let travel = hypot(value.translation.width, value.translation.height)
                            if travel >= 8 {
                                applyBrush(at: value.location, frame: frame, viewSize: geo.size)
                            }
                        }
                        .onEnded { value in
                            guard maskMode == .drawManually, let frame = previewFrame else { return }
                            let travel = hypot(value.translation.width, value.translation.height)
                            if travel < 12 {
                                applyTap(at: value.startLocation, frame: frame, viewSize: geo.size)
                            }
                        }
                )
                .accessibilityLabel("Mask preview. Tap the character or drag to brush.")
        }
        .frame(height: 220)
    }

    private func applyTap(at location: CGPoint, frame: CGImage, viewSize: CGSize) {
        guard let pixel = pixel(at: location, frame: frame, viewSize: viewSize) else { return }
        Task {
            do {
                let decision = try VACESubjectSegmenter.tapSelect(cgImage: frame, x: pixel.x, y: pixel.y)
                await MainActor.run {
                    if decision.foundSubject {
                        manualMask = decision.mask
                        maskURL = nil
                    }
                }
                await refreshMaskPreview()
            } catch {
                await MainActor.run { maskPreviewNote = error.localizedDescription }
            }
        }
    }

    private func applyBrush(at location: CGPoint, frame: CGImage, viewSize: CGSize) {
        guard let pixel = pixel(at: location, frame: frame, viewSize: viewSize) else { return }
        var mask = manualMask ?? (try? VACEMaskPlane(width: frame.width, height: frame.height, pixels: [UInt8](repeating: 0, count: frame.width * frame.height)))
        guard var mask else { return }
        if mask.width != frame.width || mask.height != frame.height {
            mask = (try? VACEMaskPlane(width: frame.width, height: frame.height, pixels: [UInt8](repeating: 0, count: frame.width * frame.height))) ?? mask
        }
        let radius = max(8, min(frame.width, frame.height) / 40)
        manualMask = VACESubjectSegmenter.paint(mask, x: pixel.x, y: pixel.y, radius: radius, erase: brushErase)
        if let preview = VACESubjectSegmenter.previewImage(base: frame, mask: manualMask!) {
            maskPreview = preview
            let fraction = VACESubjectSegmenter.coverage(of: manualMask!.pixels)
            maskCoverageText = String(format: "Coverage %.0f%% · %@", fraction * 100, VACEMaskPreviewLabel.displayName(for: "manual-tracked"))
        }
    }

    private func pixel(at location: CGPoint, frame: CGImage, viewSize: CGSize) -> (x: Int, y: Int)? {
        let imageSize = CGSize(width: frame.width, height: frame.height)
        let scale = min(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        guard scale > 0 else { return nil }
        let drawn = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let origin = CGPoint(x: (viewSize.width - drawn.width) / 2, y: (viewSize.height - drawn.height) / 2)
        let local = CGPoint(x: location.x - origin.x, y: location.y - origin.y)
        guard local.x >= 0, local.y >= 0, local.x <= drawn.width, local.y <= drawn.height else { return nil }
        let x = Int(local.x / scale)
        let y = Int(local.y / scale)
        return (min(max(0, x), frame.width - 1), min(max(0, y), frame.height - 1))
    }

    private static func centroid(of mask: VACEMaskPlane) -> (x: Int, y: Int) {
        var sumX = 0
        var sumY = 0
        var count = 0
        for y in 0..<mask.height {
            for x in 0..<mask.width where mask.pixels[y * mask.width + x] > 127 {
                sumX += x
                sumY += y
                count += 1
            }
        }
        guard count > 0 else { return (mask.width / 2, mask.height / 2) }
        return (sumX / count, sumY / count)
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
