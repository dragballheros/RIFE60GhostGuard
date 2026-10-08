import Foundation
import ImageIO
import CoreImage
import CoreVideo
import UIKit
import Metal
import avif
import avifc

/// Animated AVIF exporter for APNG and other ImageIO animations.
///
/// The expensive pixel preparation remains on Apple's GPU-backed Core Image path.
/// The AV1 bitstream itself is produced by libaom through avif.swift because that is
/// the AVIF encoder exposed by the current package. If VideoToolbox ever exposes a
/// usable hardware AV1 encoder for the target device, that is a separate codec path
/// and cannot be substituted into AVIFAnimatedEncoder without an AVIF muxing layer.
final class AnimatedAVIFEncoder: @unchecked Sendable {
    private let ciContext: CIContext
    private let colorSpace: CGColorSpace
    private let compressionProtection: Bool
    private let outlineProtection: Bool
    private let upscale2x: Bool
    private let colorPopStrength: Double
    private let watermark: WatermarkConfiguration

    init(compressionProtection: Bool = false, outlineProtection: Bool = false, upscale2x: Bool = true, colorPopStrength: Double = 0, watermark: WatermarkConfiguration = WatermarkConfiguration()) {
        self.compressionProtection = compressionProtection
        self.outlineProtection = outlineProtection
        self.upscale2x = upscale2x
        self.colorPopStrength = colorPopStrength.isFinite ? min(max(colorPopStrength, 0), 1) : 0
        self.watermark = watermark
        self.colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        if let device = MTLCreateSystemDefaultDevice() {
            self.ciContext = CIContext(
                mtlDevice: device,
                options: [
                    .cacheIntermediates: false,
                    .priorityRequestLow: false
                ]
            )
        } else {
            self.ciContext = CIContext(options: [.cacheIntermediates: false])
        }
    }

    func encode(
        sourceURL: URL,
        quality: Double = 95.0,
        speed: Int = 6,
        progress: @escaping @Sendable (Double, String) -> Void = { _, _ in }
    ) async throws -> (url: URL, width: Int, height: Int, frameCount: Int) {
        try await Task.detached(priority: .userInitiated) {
            try self.encodeSync(
                sourceURL: sourceURL,
                quality: quality,
                speed: speed,
                progress: progress
            )
        }.value
    }

    private func encodeSync(
        sourceURL: URL,
        quality: Double,
        speed: Int,
        progress: @Sendable (Double, String) -> Void
    ) throws -> (url: URL, width: Int, height: Int, frameCount: Int) {
        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil) else {
            throw AVIFEncodingError.encoderFailed("the animated image could not be decoded")
        }

        let frameCount = CGImageSourceGetCount(source)
        guard frameCount > 1 else {
            throw AVIFEncodingError.encoderFailed("animated AVIF export requires more than one frame")
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-ANIMATED-\(UUID().uuidString).avif")
        try? FileManager.default.removeItem(at: outputURL)

        let encoder = AVIFAnimatedEncoder()
        var nativeError: Error?
        guard encoder.create(.AOM, error: &nativeError) != nil else {
            throw AVIFEncodingError.encoderFailed(
                nativeError?.localizedDescription ?? "could not initialize the animated AVIF encoder"
            )
        }
        encoder.setSpeed(Int64(max(0, min(speed, 10))))
        encoder.setCompressionQuality(quality)
        encoder.setLoopsCount(readLoopCount(source))

        var outputWidth = 0
        var outputHeight = 0

        do {
            for index in 0..<frameCount {
                try Task.checkCancellation()
                guard let cgImage = CGImageSourceCreateImageAtIndex(source, index, nil) else {
                    throw AVIFEncodingError.encoderFailed("APNG frame \(index + 1) could not be decoded")
                }

                var buffer = try makeBGRABuffer(from: cgImage)
                if watermark.enabled {
                    buffer = try AnimeWatermarkRemover(configuration: watermark).apply(buffer)
                }
                if compressionProtection, let cleaned = try? CompressionGuard().clean(buffer) {
                    buffer = cleaned
                }

                let sourceWidth = CVPixelBufferGetWidth(buffer)
                let sourceHeight = CVPixelBufferGetHeight(buffer)
                let fourX = (sourceWidth * 4) * (sourceHeight * 4)
                let twoX = (sourceWidth * 2) * (sourceHeight * 2)
                let passes: Int = {
                    guard upscale2x else { return 0 }
                    if max(sourceWidth, sourceHeight) < 1080 && fourX <= 40_000_000 { return 2 }
                    if twoX <= 40_000_000 { return 1 }
                    return 0
                }()
                if passes > 0 {
                    for _ in 0..<passes {
                        buffer = try RealCUGANPass(intensity: 1.30).upscaleStill(buffer)
                        try Task.checkCancellation()
                    }
                }
                if outlineProtection {
                    buffer = try FinalOutlinePass().polishStill(buffer)
                }
                if colorPopStrength > 0 {
                    buffer = try ColorPopGrade(strength: colorPopStrength).apply(buffer)
                }
                guard let image = cgImage(from: buffer) else {
                    throw AVIFEncodingError.encoderFailed("could not render processed APNG frame")
                }
                outputWidth = image.width
                outputHeight = image.height

                let duration = frameDurationMilliseconds(source: source, index: index)
                let platformImage = UIImage(cgImage: image)

                var addError: Error?
                guard encoder.addImage(platformImage, duration: UInt(duration), error: &addError) != nil else {
                    throw AVIFEncodingError.encoderFailed(
                        addError?.localizedDescription ?? "animated AVIF rejected frame \(index + 1)"
                    )
                }

                progress(
                    0.05 + (Double(index + 1) / Double(frameCount)) * 0.90,
                    "AVIF • encoding frame \(index + 1)/\(frameCount)…"
                )
            }

            var encodeError: Error?
            guard let data = encoder.encode(&encodeError) else {
                throw AVIFEncodingError.encoderFailed(
                    encodeError?.localizedDescription ?? "animated AVIF encoder returned no data"
                )
            }

            guard !data.isEmpty, AVIFEncoderGate.isAVIFData(data) else {
                throw AVIFEncodingError.invalidContainer
            }

            try data.write(to: outputURL, options: .atomic)
            DiagnosticsLogger.shared.log(
                "Animated AVIF complete • frames=\(frameCount) • output=\(outputWidth)x\(outputHeight) • bytes=\(data.count) • quality=\(quality) • speed=\(speed) • GPU-prepared=true"
            )
            progress(1, "Animated AVIF complete • \(frameCount) frames")
            encoder.cleanUp()

            return (outputURL, outputWidth, outputHeight, frameCount)
        } catch {
            encoder.cleanUp()
            try? FileManager.default.removeItem(at: outputURL)
            throw error
        }
    }

    private func makeBGRABuffer(from cgImage: CGImage) throws -> CVPixelBuffer {
        let width = cgImage.width
        let height = cgImage.height
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var output: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &output) == kCVReturnSuccess, let output else {
            throw AVIFEncodingError.encoderFailed("could not allocate APNG frame buffer")
        }
        CVPixelBufferLockBaseAddress(output, [])
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        guard let base = CVPixelBufferGetBaseAddress(output), let context = CGContext(
            data: base,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(output),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw AVIFEncodingError.encoderFailed("could not create APNG frame graphics context")
        }
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return output
    }

    private func cgImage(from buffer: CVPixelBuffer) -> CGImage? {
        let image = CIImage(cvPixelBuffer: buffer)
        return ciContext.createCGImage(image, from: image.extent, format: .RGBA8, colorSpace: colorSpace)
    }

    private func normalizedImage(_ cgImage: CGImage) -> CGImage {
        let extent = CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height)
        let image = CIImage(cgImage: cgImage)
        guard let rendered = ciContext.createCGImage(image, from: extent, format: .RGBA8, colorSpace: colorSpace) else {
            return cgImage
        }
        return rendered
    }

    private func frameDurationMilliseconds(source: CGImageSource, index: Int) -> Int {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any]
        let png = properties?[kCGImagePropertyPNGDictionary] as? [CFString: Any]
        let unclamped = png?[kCGImagePropertyAPNGUnclampedDelayTime] as? Double
        let clamped = png?[kCGImagePropertyAPNGDelayTime] as? Double
        let delay = unclamped ?? clamped ?? 0.1
        let safeDelay = delay.isFinite && delay >= 0 ? min(delay, 655.35) : 0.1
        return max(1, Int((safeDelay * 1000.0).rounded()))
    }

    private func readLoopCount(_ source: CGImageSource) -> Int {
        guard
            let properties = CGImageSourceCopyProperties(source, nil) as? [CFString: Any],
            let png = properties[kCGImagePropertyPNGDictionary] as? [CFString: Any],
            let count = png[kCGImagePropertyAPNGLoopCount] as? NSNumber
        else {
            return 0
        }
        return max(0, count.intValue)
    }
}
