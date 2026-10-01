import SwiftUI
import UIKit
import AVFoundation
import CoreImage

/// A precise rectangle selector for stationary watermarks in an image or clip.
/// The preview is upright; WatermarkConfiguration maps video masks to raw frames.
struct WatermarkMaskEditor: View {
    let sourceURL: URL
    let isImage: Bool
    @Binding var regions: [WatermarkRegion]
    @Environment(\.dismiss) private var dismiss
    @State private var preview: UIImage?
    @State private var errorText: String?
    @State private var dragStart: CGPoint?
    @State private var draft: WatermarkRegion?

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Text("Drag a box tightly around the watermark. Add another box for a second watermark.")
                    .font(.callout).padding(.horizontal)
                if let preview {
                    GeometryReader { geometry in
                        let imageSize = preview.size
                        let scale = min(geometry.size.width / max(imageSize.width, 1), geometry.size.height / max(imageSize.height, 1))
                        let display = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
                        let rect = CGRect(x: (geometry.size.width - display.width) / 2,
                                          y: (geometry.size.height - display.height) / 2, width: display.width, height: display.height)
                        ZStack(alignment: .topLeading) {
                            Image(uiImage: preview).resizable().frame(width: display.width, height: display.height)
                                .position(x: rect.midX, y: rect.midY)
                            ForEach(regions.indices, id: \.self) { index in
                                selection(regions[index], inside: rect, color: .red)
                            }
                            if let draft { selection(draft, inside: rect, color: .yellow) }
                        }
                        .contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                guard regions.count < 8, rect.width > 0, rect.height > 0 else { return }
                                if dragStart == nil {
                                    guard rect.contains(value.startLocation) else { return }
                                    dragStart = value.startLocation
                                }
                                guard let start = dragStart else { return }
                                let end = CGPoint(x: min(max(value.location.x, rect.minX), rect.maxX),
                                                  y: min(max(value.location.y, rect.minY), rect.maxY))
                                draft = WatermarkRegion(x: Double((min(start.x, end.x) - rect.minX) / rect.width),
                                                        y: Double((min(start.y, end.y) - rect.minY) / rect.height),
                                                        width: Double(abs(end.x - start.x) / rect.width),
                                                        height: Double(abs(end.y - start.y) / rect.height))
                            }
                            .onEnded { _ in dragStart = nil })
                    }
                    .frame(height: 360)
                    Button("Add Selected Box") {
                        if let draft, draft.isValid, regions.count < 8 { regions.append(draft); self.draft = nil }
                    }
                    .disabled(draft == nil || regions.count >= 8)
                    HStack {
                        Text("\(regions.count) of 8 boxes").foregroundStyle(.secondary)
                        Spacer()
                        Button("Undo") { if !regions.isEmpty { regions.removeLast() }; draft = nil }.disabled(regions.isEmpty)
                        Button("Clear") { regions.removeAll(); draft = nil }.disabled(regions.isEmpty && draft == nil)
                    }.padding(.horizontal)
                } else if let errorText {
                    Text(errorText).foregroundStyle(.red).padding()
                } else {
                    ProgressView("Loading preview…").frame(height: 360)
                }
                Text(isImage ? "Only marked regions are reconstructed. The rest of the artwork is preserved." : "These boxes stay at the same position throughout the clip. Mark every position used by a moving watermark, or process shorter clips.")
                    .font(.caption).foregroundStyle(.secondary).padding(.horizontal)
                Spacer()
            }
            .padding(.top)
            .navigationTitle("Mark Watermark")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        if let draft, draft.isValid, regions.count < 8 { regions.append(draft) }
                        dismiss()
                    }
                }
            }
            .task(id: sourceURL) {
                do { preview = try await Self.loadPreview(sourceURL: sourceURL, isImage: isImage) }
                catch { errorText = error.localizedDescription }
            }
        }
    }

    private func selection(_ region: WatermarkRegion, inside rect: CGRect, color: Color) -> some View {
        Rectangle().fill(color.opacity(0.22)).overlay(Rectangle().stroke(color, lineWidth: 2))
            .frame(width: region.width * rect.width, height: region.height * rect.height)
            .position(x: rect.minX + (region.x + region.width / 2) * rect.width,
                      y: rect.minY + (region.y + region.height / 2) * rect.height)
            .allowsHitTesting(false)
    }

    private static func loadPreview(sourceURL: URL, isImage: Bool) async throws -> UIImage {
        try await Task.detached(priority: .userInitiated) {
            let secured = sourceURL.startAccessingSecurityScopedResource()
            defer { if secured { sourceURL.stopAccessingSecurityScopedResource() } }
            if isImage {
                guard let decoded = CIImage(contentsOf: sourceURL, options: [.applyOrientationProperty: true]) else {
                    throw ProcessorError.conversionFailed("could not load the image preview")
                }
                let scale = min(1, 1600 / max(decoded.extent.width, decoded.extent.height))
                let smaller = decoded.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                let context = CIContext(options: [.cacheIntermediates: false])
                guard let image = context.createCGImage(smaller, from: smaller.extent) else {
                    throw ProcessorError.conversionFailed("could not render the image preview")
                }
                return UIImage(cgImage: image)
            }
            let generator = AVAssetImageGenerator(asset: AVURLAsset(url: sourceURL))
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 1600, height: 1600)
            let image = try generator.copyCGImage(at: .zero, actualTime: nil)
            return UIImage(cgImage: image)
        }.value
    }
}
