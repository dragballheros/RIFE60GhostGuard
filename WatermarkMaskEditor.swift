import SwiftUI
import UIKit
import AVFoundation
import CoreImage

/// One finger paints; pinch zoom and two-finger pan allow accurate selections.
struct WatermarkMaskEditor: View {
    let sourceURL: URL
    let isImage: Bool
    @Binding var regions: [WatermarkRegion]
    @Environment(\.dismiss) private var dismiss
    @State private var preview: UIImage?
    @State private var errorText: String?
    @State private var mask = WatermarkBrushMask()
    @State private var draft: WatermarkBrushStroke?
    @State private var erase = false
    @State private var brushPercent = 0.8
    @State private var showMask = true

    private var displayedMask: WatermarkBrushMask {
        var result = mask
        if let draft { result.strokes.append(draft) }
        return result
    }
    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                Text("Paint only the watermark lettering. Pinch to zoom; use two fingers to move the image.")
                    .font(.callout).padding(.horizontal)
                if let preview {
                    WatermarkBrushCanvas(image: preview, mask: displayedMask, showMask: showMask) { point, ended in
                        if draft == nil, mask.strokes.count < 256 {
                            let radius = brushPercent / 200
                            draft = WatermarkBrushStroke(points: [], radiusX: radius,
                                radiusY: radius * Double(preview.size.width / max(preview.size.height, 1)), erase: erase)
                        }
                        if let point, var stroke = draft, stroke.points.count < 4096,
                           mask.strokes.reduce(0, { $0 + $1.points.count }) + stroke.points.count < 32768 {
                            if stroke.points.last != point { stroke.points.append(point) }
                            draft = stroke
                        }
                        if ended {
                            if let draft, draft.isValid { mask.strokes.append(draft) }
                            draft = nil
                        }
                    }
                    .frame(maxWidth: .infinity).frame(minHeight: 180, maxHeight: .infinity).layoutPriority(1)
                    Picker("Tool", selection: $erase) {
                        Text("Brush").tag(false)
                        Text("Eraser").tag(true)
                    }.pickerStyle(.segmented).padding(.horizontal)
                    HStack {
                        Text("Brush size")
                        Slider(value: $brushPercent, in: 0.2...4)
                        Text(String(format: "%.1f%%", brushPercent)).monospacedDigit()
                    }.padding(.horizontal)
                    HStack {
                        Toggle("Show mask", isOn: $showMask)
                        Button("Undo") {
                            if !mask.strokes.isEmpty { mask.strokes.removeLast() }
                            else if !mask.rectangles.isEmpty { mask.rectangles.removeLast() }
                        }.disabled(mask.strokes.isEmpty && mask.rectangles.isEmpty)
                        Button("Clear") { mask = WatermarkBrushMask(); draft = nil }
                    }.padding(.horizontal)
                    Text("Red marks show the area to reconstruct before mask padding. Unpainted pixels are preserved by the remover. Set padding to 0 for the exact painted shape.")
                        .font(.caption).foregroundStyle(.secondary).padding(.horizontal)
                } else if let errorText {
                    Text(errorText).foregroundStyle(.red).padding()
                } else { ProgressView("Loading preview…").frame(height: 420) }
                if !isImage {
                    Text("The mask stays fixed throughout the clip. A moving watermark needs a mask covering its movement or a shorter clip.")
                        .font(.caption).foregroundStyle(.secondary).padding(.horizontal)
                }
                Spacer(minLength: 0)
            }
            .padding(.top)
            .navigationTitle("Paint Watermark")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        regions = displayedMask.region.map { [$0] } ?? []
                        dismiss()
                    }.disabled(preview == nil)
                }
            }
            .task(id: sourceURL) {
                // Editing old rectangle masks keeps their coverage until erased.
                mask = WatermarkBrushMask()
                for region in regions {
                    if let brush = region.brush {
                        mask.rectangles.append(contentsOf: brush.rectangles)
                        mask.strokes.append(contentsOf: brush.strokes)
                    } else { mask.rectangles.append(region) }
                }
                do { preview = try await Self.loadPreview(sourceURL: sourceURL, isImage: isImage) }
                catch { errorText = error.localizedDescription }
            }
        }
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

private struct WatermarkBrushCanvas: UIViewRepresentable {
    let image: UIImage
    let mask: WatermarkBrushMask
    let showMask: Bool
    let changed: (WatermarkBrushPoint?, Bool) -> Void

    func makeUIView(context: Context) -> BrushScrollView {
        let view = BrushScrollView()
        view.minimumZoomScale = 1
        view.maximumZoomScale = 8
        view.delegate = context.coordinator
        view.panGestureRecognizer.minimumNumberOfTouches = 2
        view.canvas.image = image
        view.canvas.changed = changed
        view.addSubview(view.canvas)
        view.backgroundColor = .black
        return view
    }
    func updateUIView(_ view: BrushScrollView, context: Context) {
        view.canvas.image = image
        view.canvas.brushMask = mask
        view.canvas.showMask = showMask
        view.canvas.changed = changed
        view.canvas.setNeedsDisplay()
    }
    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator: NSObject, UIScrollViewDelegate {
        func viewForZooming(in scrollView: UIScrollView) -> UIView? { (scrollView as? BrushScrollView)?.canvas }
    }
}

private final class BrushScrollView: UIScrollView {
    let canvas = BrushDrawingView()
    private var fittedSize = CGSize.zero
    override func layoutSubviews() {
        super.layoutSubviews()
        guard let image = canvas.image, bounds.width > 0, bounds.height > 0 else { return }
        if fittedSize != bounds.size {
            fittedSize = bounds.size
            zoomScale = 1
            let scale = min(bounds.width / image.size.width, bounds.height / image.size.height)
            canvas.frame = CGRect(origin: .zero, size: CGSize(width: image.size.width * scale, height: image.size.height * scale))
            contentSize = canvas.frame.size
        }
        contentInset = UIEdgeInsets(top: max(0, (bounds.height - contentSize.height) / 2), left: max(0, (bounds.width - contentSize.width) / 2), bottom: 0, right: 0)
    }
}

private final class BrushDrawingView: UIView {
    var image: UIImage?
    var brushMask = WatermarkBrushMask()
    var showMask = true
    var changed: ((WatermarkBrushPoint?, Bool) -> Void)?
    override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = false
        isMultipleTouchEnabled = true
        let paint = UIPanGestureRecognizer(target: self, action: #selector(paint(_:)))
        paint.minimumNumberOfTouches = 1
        paint.maximumNumberOfTouches = 1
        addGestureRecognizer(paint)
        let tap = UITapGestureRecognizer(target: self, action: #selector(tap(_:)))
        tap.require(toFail: paint)
        addGestureRecognizer(tap)
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }
    private func point(_ location: CGPoint) -> WatermarkBrushPoint? {
        guard bounds.width > 0, bounds.height > 0 else { return nil }
        return WatermarkBrushPoint(x: Double(min(max(location.x / bounds.width, 0), 1)),
                                   y: Double(min(max(location.y / bounds.height, 0), 1)))
    }
    @objc private func paint(_ gesture: UIPanGestureRecognizer) {
        switch gesture.state {
        case .began:
            let location = gesture.location(in: self), translation = gesture.translation(in: self)
            changed?(point(CGPoint(x: location.x - translation.x, y: location.y - translation.y)), false)
            changed?(point(location), false)
        case .changed: changed?(point(gesture.location(in: self)), false)
        case .ended: changed?(point(gesture.location(in: self)), true)
        case .cancelled, .failed: changed?(nil, true)
        default: break
        }
    }
    @objc private func tap(_ gesture: UITapGestureRecognizer) { changed?(point(gesture.location(in: self)), true) }
    override func draw(_ rect: CGRect) {
        image?.draw(in: bounds)
        guard showMask, let context = UIGraphicsGetCurrentContext() else { return }
        context.saveGState()
        context.setAlpha(0.45)
        context.beginTransparencyLayer(auxiliaryInfo: nil)
        context.setFillColor(UIColor.red.cgColor)
        for box in brushMask.rectangles {
            context.fill(CGRect(x: box.x * bounds.width, y: box.y * bounds.height, width: box.width * bounds.width, height: box.height * bounds.height))
        }
        for stroke in brushMask.strokes {
            context.saveGState()
            context.setBlendMode(stroke.erase ? .clear : .normal)
            context.setFillColor(UIColor.red.cgColor)
            context.setStrokeColor(UIColor.red.cgColor)
            context.scaleBy(x: stroke.radiusX * bounds.width, y: stroke.radiusY * bounds.height)
            context.setLineWidth(2)
            context.setLineCap(.round)
            context.setLineJoin(.round)
            if let first = stroke.points.first {
                let start = CGPoint(x: first.x / stroke.radiusX, y: first.y / stroke.radiusY)
                if stroke.points.count == 1 { context.fillEllipse(in: CGRect(x: start.x - 1, y: start.y - 1, width: 2, height: 2)) }
                else {
                    context.beginPath(); context.move(to: start)
                    for p in stroke.points.dropFirst() { context.addLine(to: CGPoint(x: p.x / stroke.radiusX, y: p.y / stroke.radiusY)) }
                    context.strokePath()
                }
            }
            context.restoreGState()
        }
        context.endTransparencyLayer()
        context.restoreGState()
    }
}
