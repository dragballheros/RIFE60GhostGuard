import Foundation
import AVKit
import AVFoundation
import CoreMedia
import CoreVideo
import CoreImage
import UIKit
import SwiftUI

@MainActor
final class ProcessingPiPController: NSObject, ObservableObject, AVPictureInPictureControllerDelegate, AVPictureInPictureSampleBufferPlaybackDelegate {
    private let displayLayer = AVSampleBufferDisplayLayer()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var pictureInPictureController: AVPictureInPictureController?
    private weak var sourceView: UIView?
    private var frameIndex: Int64 = 0
    private var lastFrameDate = Date.distantPast
    private var pendingStartTask: Task<Void, Never>?
    @Published private(set) var isActive = false

    func attach(to view: UIView) {
        sourceView = view
        displayLayer.videoGravity = .resizeAspect
        displayLayer.backgroundColor = UIColor.black.cgColor
        displayLayer.frame = view.bounds

        if displayLayer.superlayer !== view.layer {
            displayLayer.removeFromSuperlayer()
            view.layer.addSublayer(displayLayer)
        }

        guard pictureInPictureController == nil,
              AVPictureInPictureController.isPictureInPictureSupported() else { return }

        let source = AVPictureInPictureController.ContentSource(
            sampleBufferDisplayLayer: displayLayer,
            playbackDelegate: self
        )
        let controller = AVPictureInPictureController(contentSource: source)
        controller.delegate = self
        controller.canStartPictureInPictureAutomaticallyFromInline = true
        controller.requiresLinearPlayback = true
        pictureInPictureController = controller

        enqueueStatusFrame(progress: 0, status: "Ready", elapsed: 0, eta: nil, force: true)
        DiagnosticsLogger.shared.log("Processing PiP prepared • sample-buffer content source")
    }

    func sourceViewDidLayout() {
        guard let sourceView else { return }
        displayLayer.frame = sourceView.bounds
    }

    func enqueueStatusFrame(progress: Double, status: String, elapsed: Double, eta: Double?, force: Bool = false) {
        let now = Date()
        if !force && now.timeIntervalSince(lastFrameDate) < 0.5 { return }
        lastFrameDate = now

        guard let pixelBuffer = makeStatusPixelBuffer(progress: progress, status: status, elapsed: elapsed, eta: eta),
              let sampleBuffer = makeSampleBuffer(pixelBuffer: pixelBuffer) else { return }

        if displayLayer.status == .failed {
            displayLayer.flushAndRemoveImage()
        }
        displayLayer.enqueue(sampleBuffer)
        frameIndex += 1
    }

    func startIfPossible() {
        guard !isActive,
              let controller = pictureInPictureController,
              AVPictureInPictureController.isPictureInPictureSupported() else { return }

        prepareAudioSessionForPiP()
        pendingStartTask?.cancel()

        if controller.isPictureInPicturePossible {
            DiagnosticsLogger.shared.log("Processing PiP start requested.")
            controller.startPictureInPicture()
            return
        }

        pendingStartTask = Task { @MainActor [weak self] in
            for _ in 0..<6 {
                try? await Task.sleep(nanoseconds: 120_000_000)
                guard let self, !Task.isCancelled, !self.isActive else { return }
                guard let controller = self.pictureInPictureController else { return }
                if controller.isPictureInPicturePossible {
                    DiagnosticsLogger.shared.log("Processing PiP start requested after transition retry.")
                    controller.startPictureInPicture()
                    return
                }
            }
            DiagnosticsLogger.shared.log("Processing PiP was not possible during this background transition.")
        }
    }

    func stop() {
        pendingStartTask?.cancel()
        pendingStartTask = nil
        guard let controller = pictureInPictureController else { return }
        if controller.isPictureInPictureActive {
            controller.stopPictureInPicture()
        } else {
            isActive = false
            deactivatePiPAudioSession()
        }
    }

    private func prepareAudioSessionForPiP() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .moviePlayback, options: [.mixWithOthers])
            try session.setActive(true)
        } catch {
            DiagnosticsLogger.shared.log("Processing PiP audio-session preparation failed: \(error.localizedDescription)")
        }
    }

    private func deactivatePiPAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        } catch {
            DiagnosticsLogger.shared.log("Processing PiP audio-session release failed: \(error.localizedDescription)")
        }
    }

    private func makeStatusPixelBuffer(progress: Double, status: String, elapsed: Double, eta: Double?) -> CVPixelBuffer? {
        let width = 640
        let height = 360
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        ) == kCVReturnSuccess, let pixelBuffer else { return nil }

        let size = CGSize(width: width, height: height)
        let renderer = UIGraphicsImageRenderer(size: size)
        let image = renderer.image { context in
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: size))

            let titleAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 38, weight: .bold),
                .foregroundColor: UIColor.white
            ]
            let statusAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 22, weight: .semibold),
                .foregroundColor: UIColor.white
            ]
            let detailAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.monospacedDigitSystemFont(ofSize: 19, weight: .medium),
                .foregroundColor: UIColor(white: 0.78, alpha: 1)
            ]

            NSString(string: "RIFE 60 Processing").draw(at: CGPoint(x: 34, y: 34), withAttributes: titleAttributes)

            let clamped = min(max(progress, 0), 1)
            let percent = String(format: "%.1f%%", clamped * 100)
            NSString(string: percent).draw(at: CGPoint(x: 34, y: 100), withAttributes: statusAttributes)

            let bar = CGRect(x: 34, y: 145, width: 572, height: 18)
            UIColor(white: 0.18, alpha: 1).setFill()
            UIBezierPath(roundedRect: bar, cornerRadius: 9).fill()
            if clamped > 0 {
                UIColor.white.setFill()
                let filled = CGRect(x: bar.minX, y: bar.minY, width: max(8, bar.width * clamped), height: bar.height)
                UIBezierPath(roundedRect: filled, cornerRadius: 9).fill()
            }

            let trimmed = status.replacingOccurrences(of: "\n", with: " ")
            let visibleStatus = String(trimmed.prefix(58))
            NSString(string: visibleStatus).draw(in: CGRect(x: 34, y: 188, width: 572, height: 58), withAttributes: statusAttributes)

            let elapsedText = "Elapsed  \(Self.formatDuration(elapsed))"
            let remainingText = eta.map { "Remaining  \(Self.formatDuration($0))" } ?? "Remaining  Calibrating"
            NSString(string: elapsedText).draw(at: CGPoint(x: 34, y: 284), withAttributes: detailAttributes)
            NSString(string: remainingText).draw(at: CGPoint(x: 330, y: 284), withAttributes: detailAttributes)
        }

        guard let ciImage = CIImage(image: image) else { return nil }
        ciContext.render(ciImage, to: pixelBuffer)
        return pixelBuffer
    }

    private func makeSampleBuffer(pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
        var formatDescription: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDescription
        ) == noErr, let formatDescription else { return nil }

        let pts = CMTime(value: frameIndex, timescale: 2)
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: pts, decodeTimeStamp: .invalid)
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        ) == noErr, let sampleBuffer else { return nil }

        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: true),
           CFArrayGetCount(attachments) > 0 {
            let raw = CFArrayGetValueAtIndex(attachments, 0)
            let dictionary = unsafeBitCast(raw, to: CFMutableDictionary.self)
            CFDictionarySetValue(
                dictionary,
                Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                Unmanaged.passUnretained(kCFBooleanTrue).toOpaque()
            )
        }
        return sampleBuffer
    }

    private static func formatDuration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "--:--" }
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        return hours > 0 ? String(format: "%d:%02d:%02d", hours, minutes, secs) : String(format: "%02d:%02d", minutes, secs)
    }

    func pictureInPictureControllerDidStartPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) {
        isActive = true
        DiagnosticsLogger.shared.log("Processing PiP started.")
    }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        failedToStartPictureInPictureWithError error: Error
    ) {
        isActive = false
        DiagnosticsLogger.shared.log("Processing PiP failed to start: \(error.localizedDescription)")
        deactivatePiPAudioSession()
    }

    func pictureInPictureControllerDidStopPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) {
        isActive = false
        DiagnosticsLogger.shared.log("Processing PiP stopped.")
        deactivatePiPAudioSession()
    }

    func pictureInPictureController(_ pictureInPictureController: AVPictureInPictureController, setPlaying playing: Bool) {
        pictureInPictureController.invalidatePlaybackState()
    }

    func pictureInPictureControllerTimeRangeForPlayback(_ pictureInPictureController: AVPictureInPictureController) -> CMTimeRange {
        CMTimeRange(start: .zero, duration: .positiveInfinity)
    }

    func pictureInPictureControllerIsPlaybackPaused(_ pictureInPictureController: AVPictureInPictureController) -> Bool {
        false
    }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        didTransitionToRenderSize newRenderSize: CMVideoDimensions
    ) { }

    func pictureInPictureController(
        _ pictureInPictureController: AVPictureInPictureController,
        skipByInterval skipInterval: CMTime,
        completion: @escaping () -> Void
    ) {
        completion()
    }

    func pictureInPictureControllerShouldProhibitBackgroundAudioPlayback(_ pictureInPictureController: AVPictureInPictureController) -> Bool {
        true
    }
}

@MainActor
final class ProcessingPiPSourceView: UIView {
    weak var controller: ProcessingPiPController?

    override func layoutSubviews() {
        super.layoutSubviews()
        controller?.sourceViewDidLayout()
    }
}

struct ProcessingPiPSourceRepresentable: UIViewRepresentable {
    let controller: ProcessingPiPController

    func makeUIView(context: Context) -> ProcessingPiPSourceView {
        let view = ProcessingPiPSourceView(frame: CGRect(x: 0, y: 0, width: 16, height: 9))
        view.backgroundColor = .black
        view.controller = controller
        controller.attach(to: view)
        return view
    }

    func updateUIView(_ uiView: ProcessingPiPSourceView, context: Context) {
        uiView.controller = controller
        controller.attach(to: uiView)
    }
}
