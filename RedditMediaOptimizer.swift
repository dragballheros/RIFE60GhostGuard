import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import UIKit

struct RedditMediaResult: Sendable {
    enum Kind: String, Sendable {
        case image
        case gif
    }

    enum Processing: String, Sendable {
        case unchangedPassThrough
        case reencoded
    }

    let url: URL
    let bytes: Int64
    let originalBytes: Int64
    let kind: Kind
    let processing: Processing
    let summary: String
}

enum RedditMediaOptimizerError: LocalizedError {
    case unsupportedImage
    case invalidVideo
    case couldNotFitImage
    case couldNotFitGIF

    var errorDescription: String? {
        switch self {
        case .unsupportedImage:
            return "Reddit mode could not read the selected image."
        case .invalidVideo:
            return "Reddit mode could not read the completed video."
        case .couldNotFitImage:
            return "The image could not be reduced to 20 MB without producing a usable Reddit image."
        case .couldNotFitGIF:
            return "The GIF could not be reduced to 20 MB while preserving a usable animation. Try a shorter clip."
        }
    }
}

/// Delivery-only optimizer. The AI rendering pipeline is untouched.
/// Reddit mode is applied after the high-quality master is complete.
struct RedditMediaOptimizer: Sendable {
    static let hardLimitBytes: Int64 = 20_000_000
    static let targetBytes: Int64 = 19_800_000

    func optimizeImage(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void = { _, _ in }
    ) async throws -> RedditMediaResult {
        try await Task.detached(priority: .userInitiated) {
            try Self.optimizeImageSync(sourceURL: sourceURL, progress: progress)
        }.value
    }

    func optimizeGIF(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void = { _, _ in }
    ) async throws -> RedditMediaResult {
        try await Task.detached(priority: .userInitiated) {
            try Self.optimizeGIFSync(sourceURL: sourceURL, progress: progress)
        }.value
    }

    func optimizeVideoAsGIF(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void = { _, _ in }
    ) async throws -> RedditMediaResult {
        try await Task.detached(priority: .userInitiated) {
            try Self.optimizeVideoAsGIFSync(sourceURL: sourceURL, progress: progress)
        }.value
    }

    private static func optimizeImageSync(
        sourceURL: URL,
        progress: @Sendable (Double, String) -> Void
    ) throws -> RedditMediaResult {
        let originalBytes = fileSize(sourceURL)

        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
              let sourceImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        let originalWidth = sourceImage.width
        let originalHeight = sourceImage.height
        let originalMaxDimension = max(originalWidth, originalHeight)

        // Reddit mode is quality-first. Never preserve PNG/JPEG merely because it
        // already fits. Encode the enhanced master as AVIF and keep the largest
        // dimensions and highest encoder quality that fit below the hard ceiling.
        let dimensions = qualityImageDimensions(width: originalWidth, height: originalHeight)
        let qualities = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]

        var attempt = 0
        let totalAttempts = max(1, dimensions.count * qualities.count)

        for dimension in dimensions {
            try Task.checkCancellation()
            let workingImage: CGImage
            if dimension >= originalMaxDimension {
                workingImage = sourceImage
            } else {
                workingImage = try resize(sourceImage, maxDimension: dimension)
            }

            for quality in qualities {
                try Task.checkCancellation()
                attempt += 1
                progress(
                    min(0.96, Double(attempt - 1) / Double(totalAttempts)),
                    "Reddit image • AVIF \(dimension)px • quality \(quality)% • optimizing…"
                )

                let candidate = try encodeAVIF(
                    image: workingImage,
                    quality: quality,
                    outputURL: FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-Reddit-AVIF-\(UUID().uuidString).avif")
                )

                if candidate.bytes <= targetBytes {
                    progress(1, "Reddit image • AVIF • \(formatMB(candidate.bytes)) • \(candidate.width)x\(candidate.height) • quality \(quality)%")
                    return RedditMediaResult(
                        url: candidate.url,
                        bytes: candidate.bytes,
                        originalBytes: originalBytes,
                        kind: .image,
                        processing: .reencoded,
                        summary: "RE-ENCODED AVIF • Reddit image • \(formatMB(candidate.bytes)) • \(candidate.width)x\(candidate.height) • quality \(quality)% • source \(formatMB(originalBytes))"
                    )
                }

                try? FileManager.default.removeItem(at: candidate.url)
            }
        }

        throw RedditMediaOptimizerError.couldNotFitImage
    }

    private static func optimizeGIFSync(
        sourceURL: URL,
        progress: @Sendable (Double, String) -> Void
    ) throws -> RedditMediaResult {
        let originalBytes = fileSize(sourceURL)

        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        let frameCount = CGImageSourceGetCount(source)
        guard frameCount > 0 else { throw RedditMediaOptimizerError.unsupportedImage }

        var frameTimes: [Double] = []
        frameTimes.reserveCapacity(frameCount)
        var duration = 0.0

        for index in 0..<frameCount {
            frameTimes.append(duration)
            duration += gifFrameDelay(source: source, index: index)
        }

        if duration <= 0 {
            duration = max(0.1, Double(frameCount) * 0.1)
        }

        guard let sourceImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }
        let baseDimension = max(sourceImage.width, sourceImage.height)
        let sourceFPS = Double(frameCount) / duration

        // Preserve already-compliant GIFs byte-for-byte. Only re-encode when size,
        // resolution, or frame rate exceeds the Reddit delivery target.
        let width = sourceImage.width
        let height = sourceImage.height
        let fitsSize = originalBytes > 0 && originalBytes <= hardLimitBytes
        let fitsResolution = baseDimension <= 2560
        let fitsFrameRate = sourceFPS <= 60

        // Reddit's hard limit is 20,000,000 bytes. The 19.8 MB target is only
        // for newly encoded output; never re-encode a compliant source just to
        // create extra headroom that it does not need.
        if fitsSize && fitsResolution && fitsFrameRate {
            progress(1, "Reddit GIF • unchanged pass-through • no re-encoding")
            let copy = try copyForDelivery(sourceURL)
            let sizeText = formatMB(originalBytes)
            let fpsText = String(format: "%.1f", sourceFPS)
            let summary = "UNCHANGED PASS-THROUGH • original GIF preserved byte-for-byte • \(sizeText) • \(width)x\(height) • ~\(fpsText) FPS"
            return RedditMediaResult(
                url: copy,
                bytes: originalBytes,
                originalBytes: originalBytes,
                kind: .gif,
                processing: .unchangedPassThrough,
                summary: summary
            )
        }

        var reencodeReasons: [String] = []
        if !fitsSize { reencodeReasons.append("over 20 MB") }
        if !fitsResolution { reencodeReasons.append("long edge over 2560 px") }
        if !fitsFrameRate { reencodeReasons.append("source cadence over 60 FPS") }
        let reasonSummary = reencodeReasons.joined(separator: ", ")
        DiagnosticsLogger.shared.log("Reddit GIF requires re-encoding • reason=\(reasonSummary) • source=\(formatMB(originalBytes)) • dimensions=\(width)x\(height) • estimated FPS=\(String(format: "%.1f", sourceFPS))")

        let dimensions = qualityDimensions(for: baseDimension)
        let fpsLevels = qualityFPS(sourceFPS: min(60, sourceFPS))

        var attempt = 0
        let totalAttempts = max(1, dimensions.count * fpsLevels.count)

        // Keep temporal smoothness first: try each available resolution at 60 FPS
        // before lowering frame rate. The first candidate remains 1440p / 60 FPS.
        for fps in fpsLevels {
            for dimension in dimensions {
                try Task.checkCancellation()
                attempt += 1

                let frameIndices = sampledGIFFrameIndices(
                    frameTimes: frameTimes,
                    duration: duration,
                    fps: fps
                )
                if frameIndices.isEmpty { continue }

                progress(
                    min(0.96, Double(attempt - 1) / Double(totalAttempts)),
                    "Reddit GIF • \(qualityLabel(forLongEdge: dimension)) • \(String(format: "%.0f", fps)) fps • optimizing…"
                )

                let candidate = try encodeGIF(
                    outputURL: temporaryGIFURL(),
                    frameCount: frameIndices.count,
                    frameProvider: { outputIndex in
                        let sourceIndex = frameIndices[outputIndex]
                        guard let raw = CGImageSourceCreateImageAtIndex(source, sourceIndex, nil) else {
                            throw RedditMediaOptimizerError.unsupportedImage
                        }
                        let image = max(raw.width, raw.height) > dimension
                            ? try resize(raw, maxDimension: dimension)
                            : raw
                        return (image, outputFrameDelay(index: outputIndex, fps: fps))
                    }
                )

                if candidate.bytes <= targetBytes {
                    progress(
                        1,
                        "Reddit GIF • \(formatMB(candidate.bytes)) • \(qualityLabel(forLongEdge: dimension)) • \(String(format: "%.0f", fps)) fps"
                    )
                    return RedditMediaResult(
                        url: candidate.url,
                        bytes: candidate.bytes,
                        originalBytes: originalBytes,
                        kind: .gif,
                        processing: .reencoded,
                        summary: "RE-ENCODED GIF • \(formatMB(candidate.bytes)) • \(qualityLabel(forLongEdge: dimension)) • \(String(format: "%.0f", fps)) FPS • source \(formatMB(originalBytes)) • reason: \(reasonSummary)"
                    )
                }

                try? FileManager.default.removeItem(at: candidate.url)
            }
        }

        throw RedditMediaOptimizerError.couldNotFitGIF
    }

    private static func optimizeVideoAsGIFSync(
        sourceURL: URL,
        progress: @Sendable (Double, String) -> Void
    ) throws -> RedditMediaResult {
        let asset = AVURLAsset(url: sourceURL)
        let duration = CMTimeGetSeconds(asset.duration)

        guard duration.isFinite, duration > 0 else {
            throw RedditMediaOptimizerError.invalidVideo
        }

        guard let track = asset.tracks(withMediaType: .video).first else {
            throw RedditMediaOptimizerError.invalidVideo
        }

        let natural = track.naturalSize
        let transform = track.preferredTransform
        let transformedRect = CGRect(origin: .zero, size: natural).applying(transform)
        let sourceWidth = max(1, Int(abs(transformedRect.width).rounded()))
        let sourceHeight = max(1, Int(abs(transformedRect.height).rounded()))
        let baseDimension = max(sourceWidth, sourceHeight)

        let imageGenerator = AVAssetImageGenerator(asset: asset)
        imageGenerator.appliesPreferredTrackTransform = true
        imageGenerator.requestedTimeToleranceBefore = .zero
        imageGenerator.requestedTimeToleranceAfter = .zero

        let dimensions = qualityDimensions(for: baseDimension)
        let fpsLevels = qualityFPS(sourceFPS: 60)

        var attempt = 0
        let totalAttempts = max(1, dimensions.count * fpsLevels.count)

        // Prioritize keeping motion smooth while reducing dimensions as needed.
        for fps in fpsLevels {
            for dimension in dimensions {
                try Task.checkCancellation()
                attempt += 1

                let frameCount = max(1, Int(ceil(duration * fps)))
                imageGenerator.maximumSize = CGSize(width: dimension, height: dimension)

                let fraction = min(0.96, Double(attempt - 1) / Double(totalAttempts))
                progress(
                    fraction,
                    "Reddit GIF • \(qualityLabel(forLongEdge: dimension)) • \(String(format: "%.0f", fps)) fps • converting…"
                )

                let candidateURL = temporaryGIFURL()
                do {
                    let candidate = try encodeGIF(
                        outputURL: candidateURL,
                        frameCount: frameCount,
                        frameProvider: { index in
                            let time = min(
                                max(0, duration - 0.001),
                                Double(index) / fps
                            )
                            let requested = CMTimeMakeWithSeconds(time, preferredTimescale: 600)
                            let image = try imageGenerator.copyCGImage(at: requested, actualTime: nil)
                            return (image, outputFrameDelay(index: index, fps: fps))
                        }
                    )

                    if candidate.bytes <= targetBytes {
                        let outputDimensions = gifDimensions(candidate.url)
                        let dimensionSummary = outputDimensions.map { "\($0.width)x\($0.height)" } ?? "unknown dimensions"
                        progress(
                            1,
                            "Reddit GIF • \(formatMB(candidate.bytes)) • \(qualityLabel(forLongEdge: dimension)) • \(String(format: "%.0f", fps)) fps • \(dimensionSummary)"
                        )
                        return RedditMediaResult(
                            url: candidate.url,
                            bytes: candidate.bytes,
                            originalBytes: fileSize(sourceURL),
                            kind: .gif,
                            processing: .reencoded,
                            summary: "RE-ENCODED GIF • \(formatMB(candidate.bytes)) • \(dimensionSummary) • \(qualityLabel(forLongEdge: dimension)) • \(String(format: "%.0f", fps)) FPS • source \(formatMB(fileSize(sourceURL)))"
                        )
                    }

                    try? FileManager.default.removeItem(at: candidate.url)
                } catch {
                    try? FileManager.default.removeItem(at: candidateURL)
                    throw error
                }
            }
        }

        throw RedditMediaOptimizerError.couldNotFitGIF
    }

    /// The first tier always preserves the processed video's actual long edge, capped at
    /// 2560 px for GIF delivery. This is important for low-resolution inputs: the GIF
    /// exporter must not clamp a fully upscaled 480p → 960p master back to 480p.
    /// Portrait media uses the same long-edge rule, for example 1440 x 2560.
    private static func qualityDimensions(for baseDimension: Int) -> [Int] {
        let cappedSourceDimension = min(max(1, baseDimension), 2560)
        let preferred = [cappedSourceDimension, 1920, 1600, 1280, 1080, 900, 720, 540, 480, 360]
        var result: [Int] = []
        for dimension in preferred where dimension >= 240 && dimension <= cappedSourceDimension && !result.contains(dimension) {
            result.append(dimension)
        }
        return result.isEmpty ? [max(240, min(cappedSourceDimension, 360))] : result
    }

    /// GIFs are capped at 60 FPS. If the source is slower, do not manufacture
    /// extra frames; use its native average frame rate as the first quality tier.
    private static func qualityFPS(sourceFPS: Double) -> [Double] {
        let cap = min(60, max(1, sourceFPS.isFinite ? sourceFPS : 30))
        let candidates = [cap, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0]
        var result: [Double] = []
        for value in candidates where value <= cap && !result.contains(where: { abs($0 - value) < 0.01 }) {
            result.append(value)
        }
        return result
    }

    private static func qualityLabel(forLongEdge dimension: Int) -> String {
        return "\(dimension)px long edge"
    }

    /// GIF stores frame delays in 1/100-second units. Quantize cumulative target
    /// times, rather than each interval independently, to preserve average speed.
    /// Viewers may still clamp short delays, so true 60 FPS is not guaranteed.
    private static func outputFrameDelay(index: Int, fps: Double) -> Double {
        let safeFPS = max(1, fps)
        let previousCentiseconds = (Double(index) * 100.0 / safeFPS).rounded()
        let nextCentiseconds = (Double(index + 1) * 100.0 / safeFPS).rounded()
        let intervalCentiseconds = max(1, nextCentiseconds - previousCentiseconds)
        return intervalCentiseconds / 100.0
    }

    private static func sampledGIFFrameIndices(
        frameTimes: [Double],
        duration: Double,
        fps: Double
    ) -> [Int] {
        guard !frameTimes.isEmpty else { return [] }

        let step = 1.0 / max(fps, 1)
        var result: [Int] = []
        result.reserveCapacity(max(1, Int(ceil(duration * fps))))

        var target = 0.0
        var cursor = 0

        while target < duration {
            while cursor + 1 < frameTimes.count && frameTimes[cursor + 1] <= target {
                cursor += 1
            }
            // Repeating a source frame is intentional when converting to a higher
            // constant output cadence. Dropping repeats would shorten the animation.
            result.append(cursor)
            target += step
        }

        return result
    }

    private static func gifFrameDelay(source: CGImageSource, index: Int) -> Double {
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any],
              let gif = properties[kCGImagePropertyGIFDictionary] as? [CFString: Any] else {
            return 0.1
        }

        let unclamped = gif[kCGImagePropertyGIFUnclampedDelayTime] as? Double
        let clamped = gif[kCGImagePropertyGIFDelayTime] as? Double
        let delay = unclamped ?? clamped ?? 0.1
        // GIF delays are stored in centiseconds; preserve long intentional holds
        // instead of clamping them to one second, which can falsely inflate FPS.
        return delay.isFinite && delay > 0.001 ? min(delay, 655.35) : 0.1
    }

    private static func encodeGIF(
        outputURL: URL,
        frameCount: Int,
        frameProvider: (Int) throws -> (CGImage, Double)
    ) throws -> EncodedGIF {
        try? FileManager.default.removeItem(at: outputURL)

        guard let destination = CGImageDestinationCreateWithURL(
            outputURL as CFURL,
            UTType.gif.identifier as CFString,
            frameCount,
            nil
        ) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        let loopProperties: [CFString: Any] = [
            kCGImagePropertyGIFDictionary: [
                kCGImagePropertyGIFLoopCount: 0
            ]
        ]
        CGImageDestinationSetProperties(destination, loopProperties as CFDictionary)

        for index in 0..<frameCount {
            try Task.checkCancellation()
            let (image, delay) = try frameProvider(index)
            let delayProperties: [CFString: Any] = [
                kCGImagePropertyGIFDictionary: [
                    kCGImagePropertyGIFDelayTime: max(0.01, delay),
                    kCGImagePropertyGIFUnclampedDelayTime: max(0.01, delay)
                ]
            ]
            CGImageDestinationAddImage(destination, image, delayProperties as CFDictionary)
        }

        guard CGImageDestinationFinalize(destination) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        let bytes = fileSize(outputURL)
        guard bytes > 0 else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        return EncodedGIF(url: outputURL, bytes: bytes)
    }

    private struct EncodedGIF: Sendable {
        let url: URL
        let bytes: Int64
    }

    private struct EncodedImage: Sendable {
        let url: URL
        let bytes: Int64
        let quality: Double
        let width: Int
        let height: Int
    }

    private struct RawEncodedImage: Sendable {
        let url: URL
        let bytes: Int64
    }

    private static func bestJPEG(
        image: CGImage,
        targetBytes: Int64,
        minimumQuality: Double
    ) throws -> EncodedImage? {
        var low = minimumQuality
        var high = 0.99
        var best: EncodedImage?

        for _ in 0..<8 {
            let quality = (low + high) * 0.5
            let encoded = try encodeImage(image, type: .jpeg, quality: quality)

            if encoded.bytes <= targetBytes {
                if let previous = best {
                    try? FileManager.default.removeItem(at: previous.url)
                }
                best = EncodedImage(
                    url: encoded.url,
                    bytes: encoded.bytes,
                    quality: quality,
                    width: image.width,
                    height: image.height
                )
                low = quality
            } else {
                try? FileManager.default.removeItem(at: encoded.url)
                low = quality
            }
        }

        return best
    }

    private enum ImageType {
        case png
        case jpeg
    }

    private static func qualityImageDimensions(width: Int, height: Int) -> [Int] {
        let longEdge = max(width, height)
        let capped = min(max(640, longEdge), 8192)
        let preferred = [capped, longEdge, 6144, 5120, 4096, 3840, 3072, 2560, 2160, 1920, 1600, 1280, 1080, 900, 720, 640]
        var result: [Int] = []
        for value in preferred where value >= 640 && value <= capped && !result.contains(value) {
            result.append(value)
        }
        if result.isEmpty { result.append(max(640, min(longEdge, 640))) }
        return result
    }

    private static func encodeAVIF(
        image: CGImage,
        quality: Int,
        outputURL: URL
    ) throws -> (url: URL, bytes: Int64, width: Int, height: Int) {
        let data = try AVIFEncoderGate.shared.encode(image, quality: Double(quality))
        try data.write(to: outputURL, options: .atomic)

        let bytes = fileSize(outputURL)
        guard bytes > 0 else { throw RedditMediaOptimizerError.couldNotFitImage }

        // Re-open the generated container through ImageIO when the platform decoder
        // is available. The encoder gate already verifies the AVIF ISO-BMFF signature,
        // so a decoder-unavailable result does not invalidate an otherwise valid file.
        if let source = CGImageSourceCreateWithURL(outputURL as CFURL, nil) {
            guard CGImageSourceGetCount(source) > 0,
                  CGImageSourceCreateImageAtIndex(source, 0, nil) != nil else {
                try? FileManager.default.removeItem(at: outputURL)
                throw RedditMediaOptimizerError.couldNotFitImage
            }
        }

        return (outputURL, bytes, image.width, image.height)
    }

    private static func encodeImage(
        _ image: CGImage,
        type: ImageType,
        quality: Double?
    ) throws -> RawEncodedImage {
        let extensionName = type == .png ? "png" : "jpg"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-Reddit-\(UUID().uuidString).\(extensionName)")
        try? FileManager.default.removeItem(at: url)

        let utType: UTType = type == .png ? .png : .jpeg
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            utType.identifier as CFString,
            1,
            nil
        ) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        if let quality {
            let properties: [CFString: Any] = [
                kCGImageDestinationLossyCompressionQuality: quality
            ]
            CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        } else {
            CGImageDestinationAddImage(destination, image, nil)
        }

        guard CGImageDestinationFinalize(destination) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        return RawEncodedImage(url: url, bytes: fileSize(url))
    }

    private static func resize(_ image: CGImage, maxDimension: Int) throws -> CGImage {
        guard max(image.width, image.height) > maxDimension else { return image }

        let scale = Double(maxDimension) / Double(max(image.width, image.height))
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))

        let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let output = context.makeImage() else {
            throw RedditMediaOptimizerError.unsupportedImage
        }
        return output
    }

    private static func copyForDelivery(_ sourceURL: URL) throws -> URL {
        let ext = sourceURL.pathExtension.isEmpty ? "dat" : sourceURL.pathExtension.lowercased()
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-Reddit-\(UUID().uuidString).\(ext)")
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.copyItem(at: sourceURL, to: destination)
        return destination
    }

    private static func temporaryGIFURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-Reddit-GIF-\(UUID().uuidString).gif")
    }

    private static func gifDimensions(_ url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(source) > 0,
              let frame = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return nil
        }
        return (frame.width, frame.height)
    }

    private static func fileSize(_ url: URL) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static func formatMB(_ bytes: Int64) -> String {
        String(format: "%.1f MB", Double(bytes) / 1_000_000.0)
    }
}
