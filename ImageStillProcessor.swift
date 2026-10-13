import Foundation
import CoreImage
import CoreVideo
import UIKit
import Metal
import ImageIO
import UniformTypeIdentifiers

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
    private let ciContext: CIContext
    private let colorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    init(compressionProtection: Bool, outlineProtection: Bool, upscale2x: Bool, colorPopEnabled: Bool = false, colorPopStrength: Double = 0.5, watermark: WatermarkConfiguration = WatermarkConfiguration()) {
        if let device = MTLCreateSystemDefaultDevice() {
            self.ciContext = CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
        } else {
            self.ciContext = CIContext(options: [.cacheIntermediates: false])
        }
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

        // Export high-precision PNG (16-bit integer RGB; at least 10-bit precision)
        // and enforce a strict 20 MiB cap. PNG is lossless; if the encoded image
        // exceeds the cap, reduce dimensions in small steps and re-encode losslessly.
        progress(0.92, "Image • Encoding high-bit-depth PNG…")
        let outputWidth = CVPixelBufferGetWidth(buffer)
        let outputHeight = CVPixelBufferGetHeight(buffer)
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-IMAGE-\(UUID().uuidString).png")
        try? FileManager.default.removeItem(at: outputURL)
        do {
            let image = CIImage(cvPixelBuffer: buffer)
            guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
                throw ProcessorError.writer("could not create the PNG source image")
            }
            let writtenBytes = try encodeHighBitDepthPNG(cgImage, to: outputURL, maxBytes: 20 * 1024 * 1024)
            DiagnosticsLogger.shared.log("PNG validation passed • bytes=\(writtenBytes) • output=\(outputWidth)x\(outputHeight) • 16-bit integer RGB • max=20MiB")
        } catch {
            try? FileManager.default.removeItem(at: outputURL)
            throw ProcessorError.writer("could not encode a high-bit-depth PNG under 20 MiB • \(error.localizedDescription)")
        }
        ciContext.clearCaches()
        DiagnosticsLogger.shared.log("Image pipeline complete • output=\(outputWidth)x\(outputHeight) high-bit-depth PNG • max=20MiB")
        return ImageStillResult(url: outputURL, width: outputWidth, height: outputHeight, notes: notes)
    }

    private func encodeHighBitDepthPNG(_ source: CGImage, to url: URL, maxBytes: Int) throws -> Int64 {
        var width = source.width
        var height = source.height
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

        for attempt in 0..<12 {
            try Task.checkCancellation()
            guard let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 16,
                bytesPerRow: width * 8,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo.byteOrder16Little.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue
            ) else {
                throw ProcessorError.writer("could not allocate 16-bit PNG staging buffer")
            }
            context.interpolationQuality = .high
            context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
            guard let highBitDepthImage = context.makeImage() else {
                throw ProcessorError.writer("could not create 16-bit PNG image")
            }
            try? FileManager.default.removeItem(at: url)
            guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                throw ProcessorError.writer("could not initialize PNG encoder")
            }
            CGImageDestinationAddImage(destination, highBitDepthImage, nil)
            guard CGImageDestinationFinalize(destination) else {
                throw ProcessorError.writer("PNG encoder could not finalize the image")
            }
            let bytes = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber
            guard let bytes, bytes.int64Value > 0 else {
                throw ProcessorError.writer("PNG output was empty")
            }
            if bytes.intValue <= maxBytes { return bytes.int64Value }
            guard attempt < 11 else {
                throw ProcessorError.writer("image cannot fit below 20 MiB without excessive downscaling")
            }
            let scale = min(0.94, max(0.72, sqrt(Double(maxBytes) / Double(bytes.intValue)) * 0.94))
            let nextWidth = max(1, Int(Double(width) * scale))
            let nextHeight = max(1, Int(Double(height) * scale))
            guard nextWidth < width, nextHeight < height else {
                throw ProcessorError.writer("could not reduce image dimensions to meet the 20 MiB cap")
            }
            width = nextWidth
            height = nextHeight
            DiagnosticsLogger.shared.log("High-bit-depth PNG exceeds 20 MiB; lossless re-encode at reduced dimensions • attempt=\(attempt + 1) • size=\(width)x\(height)")
        }
        throw ProcessorError.writer("could not satisfy the 20 MiB PNG limit")
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
