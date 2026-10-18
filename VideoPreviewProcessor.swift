import Foundation
import AVFoundation
import CoreGraphics
import CoreImage
import ImageIO
import UIKit

/// Produces a lightweight, disposable preview of the opening video/GIF frames.
/// Each sample is sent through the existing still-frame enhancement stages; RIFE
/// is intentionally omitted because a few isolated samples cannot preview motion interpolation.
enum VideoPreviewProcessor {
    static func generate(
        sourceURL: URL,
        isGIF: Bool,
        compressionProtection: Bool,
        outlineProtection: Bool,
        upscale2x: Bool,
        colorPopEnabled: Bool,
        colorPopStrength: Double,
        watermark: WatermarkConfiguration
    ) async throws -> [URL] {
        try await Task.detached(priority: .userInitiated) {
            try generateSync(
                sourceURL: sourceURL,
                isGIF: isGIF,
                compressionProtection: compressionProtection,
                outlineProtection: outlineProtection,
                upscale2x: upscale2x,
                colorPopEnabled: colorPopEnabled,
                colorPopStrength: colorPopStrength,
                watermark: watermark
            )
        }.value
    }

    private static func generateSync(
        sourceURL: URL,
        isGIF: Bool,
        compressionProtection: Bool,
        outlineProtection: Bool,
        upscale2x: Bool,
        colorPopEnabled: Bool,
        colorPopStrength: Double,
        watermark: WatermarkConfiguration
    ) throws -> [URL] {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-Preview-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var completed: [URL] = []
        do {
            let samples: [CGImage]
            if isGIF {
                guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil) else {
                    throw ProcessorError.conversionFailed("The selected GIF could not be opened for preview.")
                }
                let count = min(3, CGImageSourceGetCount(source))
                samples = try (0..<count).map { index in
                    guard let image = CGImageSourceCreateImageAtIndex(source, index, nil) else {
                        throw ProcessorError.conversionFailed("GIF preview frame \(index + 1) could not be decoded.")
                    }
                    return image
                }
            } else {
                let asset = AVURLAsset(url: sourceURL)
                let duration = CMTimeGetSeconds(asset.duration)
                guard duration.isFinite, duration > 0 else {
                    throw ProcessorError.conversionFailed("The selected video has no readable duration.")
                }
                let generator = AVAssetImageGenerator(asset: asset)
                generator.appliesPreferredTrackTransform = true
                generator.maximumSize = CGSize(width: 720, height: 720)
                let times = [0.0, min(0.15, max(0, duration - 0.01)), min(0.30, max(0, duration - 0.01))]
                    .reduce(into: [Double]()) { values, time in
                        if !values.contains(where: { abs($0 - time) < 0.001 }) { values.append(time) }
                    }
                samples = try times.map { seconds in
                    try generator.copyCGImage(at: CMTime(seconds: seconds, preferredTimescale: 600), actualTime: nil)
                }
            }

            guard !samples.isEmpty else {
                throw ProcessorError.conversionFailed("No frames were available to preview.")
            }

            let context = CIContext(options: [.cacheIntermediates: false])
            for (index, cgImage) in samples.enumerated() {
                try Task.checkCancellation()
                let sourceURL = root.appendingPathComponent("source-\(index).png")
                guard let destination = CGImageDestinationCreateWithURL(sourceURL as CFURL, "public.png" as CFString, 1, nil) else {
                    throw ProcessorError.conversionFailed("Could not create a temporary preview frame.")
                }
                CGImageDestinationAddImage(destination, cgImage, nil)
                guard CGImageDestinationFinalize(destination) else {
                    throw ProcessorError.conversionFailed("Could not write a temporary preview frame.")
                }

                let processor = ImageStillProcessor(
                    compressionProtection: compressionProtection,
                    outlineProtection: outlineProtection,
                    upscale2x: upscale2x,
                    colorPopEnabled: colorPopEnabled,
                    colorPopStrength: colorPopStrength,
                    watermark: watermark
                )
                let processed = try await processor.process(sourceURL: sourceURL) { _, _ in }
                defer { try? FileManager.default.removeItem(at: sourceURL) }
                guard let ciImage = CIImage(contentsOf: processed.url),
                      let outputCG = context.createCGImage(ciImage, from: ciImage.extent),
                      let png = UIImage(cgImage: outputCG).pngData() else {
                    throw ProcessorError.conversionFailed("The enhanced preview frame could not be decoded.")
                }
                let outputURL = root.appendingPathComponent("preview-\(index + 1).png")
                try png.write(to: outputURL, options: .atomic)
                try? FileManager.default.removeItem(at: processed.url)
                completed.append(outputURL)
            }
            return completed
        } catch {
            for url in completed { try? FileManager.default.removeItem(at: url) }
            try? FileManager.default.removeItem(at: root)
            throw error
        }
    }
}
