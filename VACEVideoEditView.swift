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
    @State private var brushPercent = 0.8
    @State private var brushDraft: WatermarkBrushStroke?
    @State private var draftFrameIndex = 0
    @State private var keyframeStrokes: [Int: WatermarkBrushMask] = [:]
    @State private var keyframeMasks: [Int: VACEMaskKeyframe] = [:]
    @State private var previewFrameIndex = 0
    @State private var previewFPS = 24.0
    @State private var previewUIImage: UIImage?
    @State private var trackLabel: String?
    @State private var paintingThisFrame = true
    @State private var previewFrame: CGImage?

    private var canvasMask: WatermarkBrushMask {
        var mask = keyframeStrokes[previewFrameIndex] ?? WatermarkBrushMask()
        if let brushDraft, draftFrameIndex == previewFrameIndex {
            mask.strokes.append(brushDraft)
        }
        return mask
    }

    private var generateDisableReasons: [String] {
        var reasons: [String] = []
        if sourceURL == nil { reasons.append("source video") }
        if maskMode.needsHandDrawnMask && maskURL == nil && keyframeMasks.isEmpty { reasons.append("a tap or brush stroke") }
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
                        Text("Paint the character on one frame. Every other frame is tracked on device.")
                            .font(.caption).foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 0) {
                            Text("Tool").font(.subheadline).foregroundStyle(.secondary)
                            ForEach([false, true], id: \.self) { isEraser in
                                Button {
                                    brushErase = isEraser
                                } label: {
                                    HStack {
                                        Text(isEraser ? "Eraser" : "Brush").foregroundStyle(.primary)
                                        Spacer()
                                        Image(systemName: brushErase == isEraser ? "checkmark.circle.fill" : "circle")
                                            .foregroundStyle(brushErase == isEraser ? Color.accentColor : Color.secondary)
                                    }
                                    .padding(.vertical, 6)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        HStack {
                            Text("Brush size")
                            Slider(value: $brushPercent, in: 0.2...4)
                            Text(String(format: "%.1f%%", brushPercent)).monospacedDigit()
                        }
                        HStack {
                            Button("Undo") {
                                var strokes = keyframeStrokes[previewFrameIndex] ?? WatermarkBrushMask()
                                if !strokes.strokes.isEmpty { strokes.strokes.removeLast() }
                                keyframeStrokes[previewFrameIndex] = strokes
                                commitBrush(on: previewFrameIndex)
                            }
                            .disabled((keyframeStrokes[previewFrameIndex]?.strokes ?? []).isEmpty)
                            Button("Clear frame") {
                                keyframeStrokes[previewFrameIndex] = WatermarkBrushMask()
                                keyframeMasks[previewFrameIndex] = nil
                                brushDraft = nil
                            }
                        }
                    }
                    if sourceURL != nil {
                        Slider(value: $previewTime, in: 0...max(previewDuration, 0.1)) { editing in
                            if !editing { Task { await refreshMaskPreview() } }
                        }
                        .disabled(isRunning)
                        Text(String(format: "Preview %.1f s", previewTime))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if maskMode == .drawManually, let previewUIImage, paintingThisFrame || keyframeStrokes[previewFrameIndex] != nil {
                        WatermarkBrushCanvas(image: previewUIImage, mask: canvasMask, showMask: true) { point, ended in
                            let frameIndex = previewFrameIndex
                            if brushDraft == nil, (keyframeStrokes[frameIndex]?.strokes.count ?? 0) < 256 {
                                draftFrameIndex = frameIndex
                                let radius = brushPercent / 200
                                brushDraft = WatermarkBrushStroke(
                                    points: [],
                                    radiusX: radius,
                                    radiusY: radius * Double(previewUIImage.size.width / max(previewUIImage.size.height, 1)),
                                    erase: brushErase
                                )
                            }
                            if let point, var stroke = brushDraft, draftFrameIndex == frameIndex, stroke.points.count < 4096 {
                                if stroke.points.last != point { stroke.points.append(point) }
                                brushDraft = stroke
                            }
                            if ended {
                                if let brushDraft, brushDraft.isValid, draftFrameIndex == frameIndex {
                                    var stored = keyframeStrokes[frameIndex] ?? WatermarkBrushMask()
                                    stored.strokes.append(brushDraft)
                                    keyframeStrokes[frameIndex] = stored
                                }
                                self.brushDraft = nil
                                commitBrush(on: frameIndex)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 280)
                    } else if maskMode == .drawManually, maskPreview != nil, !paintingThisFrame {
                        Button("Touch up this frame") { paintingThisFrame = true }
                            .font(.caption)
                    }
                    if maskMode != .drawManually || (!paintingThisFrame && keyframeStrokes[previewFrameIndex] == nil), let maskPreview {
                        Image(uiImage: maskPreview)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 220)
                    }
                    if let maskCoverageText {
                        Text(maskCoverageText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if let trackLabel {
                        Text(trackLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
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
            guard maskURL != nil || !keyframeMasks.isEmpty else { errorText = VACEVideoEditInputError.missingMask.localizedDescription; return }
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
                var frames = Array(keyframeMasks.values)
                if frames.isEmpty, let maskURL {
                    let seedMask = try Self.loadMask(url: maskURL, width: width, height: height)
                    frames = [VACEMaskKeyframe(frameIndex: 0, mask: seedMask, appearance: [])]
                }
                guard frames.contains(where: { $0.mask.activePixelCount > 0 }) else { throw VACEVideoEditInputError.noEditableMaskPixels }
                provider = VACEManualTrackMaskProvider(keyframes: frames)
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
        let masks = keyframeMasks
        do {
            let asset = AVURLAsset(url: sourceURL)
            let duration = try await asset.load(.duration).seconds
            if duration.isFinite, duration > 0 {
                previewDuration = duration
            }
            if let track = try await asset.loadTracks(withMediaType: .video).first {
                let rate = try await track.load(.nominalFrameRate)
                if rate > 1 { previewFPS = Double(rate) }
            }
            let frameIndex = max(0, Int((seconds * previewFPS).rounded()))
            previewFrameIndex = frameIndex
            if keyframeStrokes[frameIndex] == nil {
                paintingThisFrame = false
            }
            let frame = try await VACESubjectSegmenter.frameImage(url: sourceURL, at: seconds)
            previewFrame = frame
            previewUIImage = UIImage(cgImage: frame)
            let built: VACESegmentationDecision
            if mode == .drawManually, !masks.isEmpty {
                let tracked = try await Self.propagatedMask(url: sourceURL, fps: previewFPS, target: frameIndex, keyframes: masks, targetImage: frame)
                let fraction = VACESubjectSegmenter.coverage(of: tracked.mask.pixels)
                built = VACESegmentationDecision(mask: tracked.mask, foundSubject: fraction > 0, method: "manual-tracked", coverage: fraction)
                trackLabel = frameIndex == tracked.anchorFrame
                    ? "Drawn on this frame"
                    : "Tracked from frame \(tracked.anchorFrame + 1)"
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
            if mode != .drawManually { trackLabel = nil }
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

    private func commitBrush(on frameIndex: Int) {
        guard let frame = previewFrame, let strokes = keyframeStrokes[frameIndex], strokes.strokes.contains(where: { !$0.erase }) else {
            keyframeMasks[frameIndex] = nil
            return
        }
        let raster = strokes.raster(width: frame.width, height: frame.height, padding: 0)
        guard let plane = try? VACEMaskPlane(width: frame.width, height: frame.height, pixels: raster), plane.activePixelCount > 0 else {
            keyframeMasks[frameIndex] = nil
            return
        }
        let appearance = VACESubjectSegmenter.appearanceHistogram(cgImage: frame, mask: raster)
        keyframeMasks[frameIndex] = VACEMaskKeyframe(frameIndex: frameIndex, mask: plane, appearance: appearance)
        paintingThisFrame = true
        Task { await refreshMaskPreview() }
    }

    /// Walks from the nearest earlier keyframe to the scrubbed frame and caches each step.
    private static func propagatedMask(url: URL, fps: Double, target: Int, keyframes: [Int: VACEMaskKeyframe], targetImage: CGImage) async throws -> VACETrackedFrame {
        let start = keyframes.keys.filter { $0 <= target }.max() ?? target
        let tracker = VACEManualTrackMaskProvider(keyframes: Array(keyframes.values))
        var last: VACETrackedFrame?
        for index in start...target {
            let image: CGImage
            if index == target {
                image = targetImage
            } else {
                image = try await VACESubjectSegmenter.frameImage(url: url, at: Double(index) / max(fps, 1))
            }
            let rgba = try VACESubjectSegmenter.rgbaFrame(from: image)
            let time = CMTime(seconds: Double(index) / max(fps, 1), preferredTimescale: 600)
            last = try tracker.track(frameIndex: index, presentationTime: time, image: image, rgba: rgba)
        }
        if let last { return last }
        let empty = try VACEMaskPlane(width: targetImage.width, height: targetImage.height, pixels: [UInt8](repeating: 0, count: targetImage.width * targetImage.height))
        return VACETrackedFrame(mask: empty, confidence: 0, anchorFrame: start, usedFallback: true)
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
