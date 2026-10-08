import Foundation
import CoreImage
import CoreVideo
import UIKit
import avif

struct ImageStillResult: Sendable {
    let url: URL
    let width: Int
    let height: Int
    let notes: [String]
}

/// Still-image path. Images have no neighbouring frame, so RIFE is skipped entirely and the
/// image goes through the same per-frame stages the video pipeline uses:
///
///     Compression Guard -> Real-CUGAN native 2x -> Final Sharpie
///
/// The result is written as a lossless PNG.
final class ImageStillProcessor {
    static let maxInputPixels = 64_000_000
    static let maxUpscaledPixels = 40_000_000

    private let compressionProtection: Bool
    private let outlineProtection: Bool
    private let upscale2x: Bool
    private let colorPopStrength: Double
    private let watermark: WatermarkConfiguration
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    init(compressionProtection: Bool, outlineProtection: Bool, upscale2x: Bool, colorPopEnabled: Bool = false, colorPopStrength: Double = 0.5, watermark: WatermarkConfiguration = WatermarkConfiguration()) {
        self.watermark = watermark
        self.colorPopStrength = colorPopEnabled && colorPopStrength.isFinite ? min(max(colorPopStrength, 0), 1) : 0
        self.compressionProtection = compressionProtection
        self.outlineProtection = outlineProtection
        self.upscale2x = upscale2x
    }

    func process(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void
    ) async throws -> ImageStillResult {
        var notes: [String] = []

        progress(0.02, "Image • Loading…")
        // ImageIO exposes APNG as a multi-frame image source. Never send an
        // animated APNG through the single-frame CIImage(contentsOf:) path.
        // That path only gives us one frame and was the reason APNG exports
        // could stall at the single-image AVIF stage.
        if let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil), CGImageSourceGetCount(source) > 1 {
            progress(0.04, "Image • Animated APNG detected…")
            let animated = AnimatedAVIFEncoder(
                compressionProtection: compressionProtection,
                outlineProtection: outlineProtection,
                upscale2x: upscale2x,
                colorPopStrength: colorPopStrength,
                watermark: watermark
            )
            let result = try await animated.encode(sourceURL: sourceURL, quality: 95.0, speed: 6, progress: progress)
            DiagnosticsLogger.shared.log("Animated image pipeline complete • APNG frames=\\(result.frameCount) • output=\\(result.width)x\\(result.height) AVIF")
            return ImageStillResult(url: result.url, width: result.width, height: result.height, notes: ["Animated APNG encoded as animated AVIF"])
        }
        var buffer = try loadBGRABuffer(from: sourceURL)
        let sourceWidth = CVPixelBufferGetWidth(buffer)
        let sourceHeight = CVPixelBufferGetHeight(buffer)
        DiagnosticsLogger.shared.log("Image pipeline entered • RIFE skipped • source=\(sourceWidth)x\(sourceHeight) • compression=\(compressionProtection) • cugan2x=\(upscale2x) • outline=\(outlineProtection)")
        try Task.checkCancellation()

        if watermark.enabled {
            progress(0.05, "Image • Anime watermark removal…")
            let remover = try AnimeWatermarkRemover(configuration: watermark)
            buffer = try remover.apply(buffer)
            try Task.checkCancellation()
        }

        if compressionProtection {
            progress(0.10, "Image • Compression Guard…")
            if let cleaned = try? CompressionGuard().clean(buffer) { buffer = cleaned }
            try Task.checkCancellation()
        }

        // Automatic quality-first scaling:
        //   <1080p long edge + 4x fits memory budget -> two native 2x passes.
        //   otherwise, if 2x fits -> one native 2x pass.
        //   otherwise preserve the decoded source rather than forcing an unsafe allocation.
        let automaticPasses: Int = {
            let fourX = (sourceWidth * 4) * (sourceHeight * 4)
            let twoX = (sourceWidth * 2) * (sourceHeight * 2)
            if max(sourceWidth, sourceHeight) < 1080 && fourX <= Self.maxUpscaledPixels { return 2 }
            if twoX <= Self.maxUpscaledPixels { return 1 }
            return 0
        }()
        let requestedPasses = upscale2x ? automaticPasses : 0

        if requestedPasses > 0 {
            for pass in 1...requestedPasses {
                try Task.checkCancellation()
                let fraction = 0.18 + (Double(pass) / Double(requestedPasses)) * 0.48
                progress(fraction, "Image • Real-CUGAN native 2× pass \(pass)/\(requestedPasses)…")
                let cugan = RealCUGANPass(intensity: 1.30)
                buffer = try cugan.upscaleStill(buffer)
                try Task.checkCancellation()
            }
            let finalWidth = CVPixelBufferGetWidth(buffer)
            let finalHeight = CVPixelBufferGetHeight(buffer)
            DiagnosticsLogger.shared.log("Image automatic upscale complete • passes=\(requestedPasses) • output=\(finalWidth)x\(finalHeight)")
        } else if upscale2x {
            let note = "Real-CUGAN automatic upscale skipped • 2× output would exceed 40 megapixels"
            notes.append(note)
            DiagnosticsLogger.shared.log("Image pipeline: \(note) (\(sourceWidth)x\(sourceHeight))")
        }

        if outlineProtection {
            progress(0.65, "Image • Final Sharpie…")
            let polish = FinalOutlinePass()
            buffer = try polish.polishStill(buffer)
            try Task.checkCancellation()
        }

        if colorPopStrength > 0 {
            DiagnosticsLogger.shared.log("Color Pop active • strength=\(String(format: "%.2f", colorPopStrength)) • image grade after upscale and final polish")
            progress(0.88, "Image • Color Pop…")
            buffer = try ColorPopGrade(strength: colorPopStrength).apply(buffer)
            try Task.checkCancellation()
        }

        // Regular image mode uses AVIF directly. Unlike Reddit delivery mode,
        // this is a single loss-controlled encode at the enhanced dimensions.
        // There is no 20 MB search, no Reddit-specific resizing, and no delivery
        // optimizer involved. The enhancement result itself is what gets exported.
        progress(0.92, "Image • Encoding AVIF…")
        let outputWidth = CVPixelBufferGetWidth(buffer)
        let outputHeight = CVPixelBufferGetHeight(buffer)
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-IMAGE-\(UUID().uuidString).avif")
        try? FileManager.default.removeItem(at: outputURL)
        do {
            let image = CIImage(cvPixelBuffer: buffer)
            guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
                throw ProcessorError.writer("could not create the AVIF source image")
            }
            let data = try AVIFEncoderGate.shared.encode(cgImage, quality: 95.0)
            try data.write(to: outputURL, options: .atomic)

            let writtenBytes = try FileManager.default.attributesOfItem(atPath: outputURL.path)[.size] as? NSNumber
            guard let writtenBytes, writtenBytes.int64Value > 0 else {
                throw ProcessorError.writer("AVIF output was empty after writing")
            }
            DiagnosticsLogger.shared.log("AVIF validation passed • bytes=\(writtenBytes.int64Value) • output=\(outputWidth)x\(outputHeight)")
        } catch {
            throw ProcessorError.writer("could not encode the AVIF • \(error.localizedDescription)")
        }
        ciContext.clearCaches()
        DiagnosticsLogger.shared.log("Image pipeline complete • output=\(outputWidth)x\(outputHeight) AVIF • quality=95")
        return ImageStillResult(url: outputURL, width: outputWidth, height: outputHeight, notes: notes)
    }

    private func loadBGRABuffer(from url: URL) throws -> CVPixelBuffer {
        guard let decoded = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else {
            throw ProcessorError.conversionFailed("the selected image could not be decoded")
        }
        let extent = decoded.extent
        guard extent.width.isFinite, extent.height.isFinite, extent.width >= 1, extent.height >= 1 else {
            throw ProcessorError.conversionFailed("the selected image has no readable pixels")
        }
        let width = Int(extent.width.rounded())
        let height = Int(extent.height.rounded())
        guard width * height <= Self.maxInputPixels else {
            throw ProcessorError.conversionFailed("the image is \(width)×\(height); the largest supported input is 64 megapixels")
        }

        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        let normalized = decoded.transformed(by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y))
        // Flatten any transparency onto white; the models expect opaque RGB.
        let white = CIImage(color: CIColor(red: 1, green: 1, blue: 1, alpha: 1)).cropped(to: bounds)
        let flattened = normalized.composited(over: white).cropped(to: bounds)

        let buffer = try makeBGRABuffer(width: width, height: height)
        ciContext.render(flattened, to: buffer, bounds: bounds, colorSpace: colorSpace)
        return buffer
    }

    private func makeBGRABuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var output: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &output)
        guard status == kCVReturnSuccess, let output else {
            throw ProcessorError.conversionFailed("could not allocate a \(width)×\(height) image buffer")
        }
        return output
    }
}
