import Foundation
import CoreImage
import CoreVideo

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
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    init(compressionProtection: Bool, outlineProtection: Bool, upscale2x: Bool) {
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
        var buffer = try loadBGRABuffer(from: sourceURL)
        let sourceWidth = CVPixelBufferGetWidth(buffer)
        let sourceHeight = CVPixelBufferGetHeight(buffer)
        DiagnosticsLogger.shared.log("Image pipeline entered • RIFE skipped • source=\(sourceWidth)x\(sourceHeight) • compression=\(compressionProtection) • cugan2x=\(upscale2x) • outline=\(outlineProtection)")
        try Task.checkCancellation()

        if compressionProtection {
            progress(0.10, "Image • Compression Guard…")
            if let cleaned = try? CompressionGuard().clean(buffer) { buffer = cleaned }
            try Task.checkCancellation()
        }

        if upscale2x {
            let targetPixels = (sourceWidth * 2) * (sourceHeight * 2)
            if targetPixels > Self.maxUpscaledPixels {
                let note = "Real-CUGAN 2× skipped • output would exceed 40 megapixels"
                notes.append(note)
                DiagnosticsLogger.shared.log("Image pipeline: \(note) (\(sourceWidth)x\(sourceHeight))")
            } else {
                progress(0.25, "Image • Real-CUGAN native 2×…")
                let cugan = RealCUGANPass(intensity: 1.30)
                buffer = try cugan.upscaleStill(buffer)
                try Task.checkCancellation()
            }
        }

        if outlineProtection {
            progress(0.65, "Image • Final Sharpie…")
            let polish = FinalOutlinePass()
            buffer = try polish.polishStill(buffer)
            try Task.checkCancellation()
        }

        progress(0.92, "Image • Encoding PNG…")
        let outputWidth = CVPixelBufferGetWidth(buffer)
        let outputHeight = CVPixelBufferGetHeight(buffer)
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-IMAGE-\(UUID().uuidString).png")
        try? FileManager.default.removeItem(at: outputURL)
        do {
            try ciContext.writePNGRepresentation(
                of: CIImage(cvPixelBuffer: buffer),
                to: outputURL,
                format: .RGBA8,
                colorSpace: colorSpace
            )
        } catch {
            throw ProcessorError.writer("could not encode the PNG • \(error.localizedDescription)")
        }
        ciContext.clearCaches()
        DiagnosticsLogger.shared.log("Image pipeline complete • output=\(outputWidth)x\(outputHeight) PNG")
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
