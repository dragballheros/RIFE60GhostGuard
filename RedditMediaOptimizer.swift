import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

struct RedditMediaResult: Sendable {
    enum Kind: String, Sendable {
        case image
        case gif
    }

    let url: URL
    let bytes: Int64
    let kind: Kind
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

        if originalBytes > 0, originalBytes <= targetBytes {
            progress(1, "Reddit image • already under 20 MB")
            let copy = try copyForDelivery(sourceURL)
            return RedditMediaResult(
                url: copy,
                bytes: originalBytes,
                kind: .image,
                summary: "Reddit image • (formatMB(originalBytes)) • no recompression needed"
            )
        }

        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RedditMediaOptimizerError.unsupportedImage
        }

        if let png = try? encodeImage(image, type: .png, quality: nil) {
            if png.bytes <= targetBytes {
                progress(1, "Reddit image • lossless PNG under 20 MB")
                return RedditMediaResult(
                    url: png.url,
                    bytes: png.bytes,
                    kind: .image,
                    summary: "Reddit image • (formatMB(png.bytes)) • lossless PNG"
                )
            }
            try? FileManager.default.removeItem(at: png.url)
        }

        let originalMaxDimension = max(image.width, image.height)
        var dimension = originalMaxDimension

        while dimension >= 640 {
            try Task.checkCancellation()

            let workingImage = dimension >= originalMaxDimension
                ? image
                : try resize(image, maxDimension: dimension)

            if let best = try bestJPEG(
                image: workingImage,
                targetBytes: targetBytes,
                minimumQuality: 0.35
            ) {
                progress(1, "Reddit image • highest-quality JPEG under 20 MB")
                return RedditMediaResult(
                    url: best.url,
                    bytes: best.bytes,
                    kind: .image,
                    summary: "Reddit image • (formatMB(best.bytes)) • JPEG quality (Int(best.quality * 100))% • (best.width)x(best.height)"
                )
            }

            dimension = Int(Double(dimension) * 0.90)
            let denominator = max(1, originalMaxDimension - 640)
            let fraction = 1.0 - min(1.0, Double(max(0, dimension - 640)) / Double(denominator))
            progress(min(0.95, fraction), "Reddit image • reducing dimensions for 20 MB target…")
        }

        throw RedditMediaOptimizerError.couldNotFitImage
    }

    private static func optimizeGIFSync(
        sourceURL: URL,
        progress: @Sendable (Double, String) -> Void
    ) throws -> RedditMediaResult {
        let originalBytes = fileSize(sourceURL)

        if originalBytes > 0, originalBytes <= targetBytes {
            progress(1, "Reddit GIF • already under 20 MB")
            let copy = try copyForDelivery(sourceURL)
            return RedditMediaResult(
                url: copy,
                bytes: originalBytes,
                kind: .gif,
                summary: "Reddit GIF • (formatMB(originalBytes)) • no recompression needed"
            )
        }

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

        let sourceImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        let baseDimension = sourceImage.map { max($0.width, $0.height) } ?? 720
        let dimensions = qualityDimensions(for: baseDimension)
        let fpsLevels = qualityFPS(for: duration)

        var attempt = 0
        let totalAttempts = max(1, dimensions.count * fpsLevels.count)

        for dimension in dimensions {
            for fps in fpsLevels {
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
                    "Reddit GIF • (dimension)p • (String(format: "%.0f", fps)) fps • optimizing…"
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
                        return (image, 1.0 / fps)
                    }
                )

                if candidate.bytes <= targetBytes {
                    progress(
                        1,
                        "Reddit GIF • (formatMB(candidate.bytes)) • (dimension)p • (String(format: "%.0f", fps)) fps"
                    )
                    return RedditMediaResult(
                        url: candidate.url,
                        bytes: candidate.bytes,
                        kind: .gif,
                        summary: "Reddit GIF • (formatMB(candidate.bytes)) • (dimension)p • (String(format: "%.0f", fps)) fps"
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
        let fpsLevels = qualityFPS(for: duration)

        var attempt = 0
        let totalAttempts = max(1, dimensions.count * fpsLevels.count)

        for dimension in dimensions {
            for fps in fpsLevels {
                try Task.checkCancellation()
                attempt += 1

                let frameCount = max(1, Int(ceil(duration * fps)))
                imageGenerator.maximumSize = CGSize(width: dimension, height: dimension)

                let fraction = min(0.96, Double(attempt - 1) / Double(totalAttempts))
                progress(
                    fraction,
                    "Reddit GIF • (dimension)p • (String(format: "%.0f", fps)) fps • converting…"
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
                            return (image, 1.0 / fps)
                        }
                    )

                    if candidate.bytes <= targetBytes {
                        progress(
                            1,
                            "Reddit GIF • (formatMB(candidate.bytes)) • (dimension)p • (String(format: "%.0f", fps)) fps"
                        )
                        return RedditMediaResult(
                            url: candidate.url,
                            bytes: candidate.bytes,
                            kind: .gif,
                            summary: "Reddit GIF • (formatMB(candidate.bytes)) • (dimension)p • (String(format: "%.0f", fps)) fps"
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

    private static func qualityDimensions(for baseDimension: Int) -> [Int] {
        let preferred = [1440, 1200, 1080, 960, 840, 720, 600, 480, 360]
        let filtered = preferred.filter { $0 <= baseDimension }
        if filtered.isEmpty {
            return [max(240, min(baseDimension, 360))]
        }
        return filtered
    }

    private static func qualityFPS(for duration: Double) -> [Double] {
        switch duration {
        case ..<8:
            return [30, 24, 20, 15, 12, 10, 8]
        case ..<16:
            return [24, 20, 15, 12, 10, 8, 6]
        case ..<30:
            return [20, 15, 12, 10, 8, 6]
        default:
            return [15, 12, 10, 8, 6, 5]
        }
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
            if result.last != cursor {
                result.append(cursor)
            }
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
        return delay.isFinite && delay > 0.001 ? min(delay, 1.0) : 0.1
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
                    kCGImagePropertyGIFDelayTime: max(0.02, delay),
                    kCGImagePropertyGIFUnclampedDelayTime: max(0.02, delay)
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
                high = quality
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

    private static func encodeImage(
        _ image: CGImage,
        type: ImageType,
        quality: Double?
    ) throws -> RawEncodedImage {
        let extensionName = type == .png ? "png" : "jpg"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-Reddit-(UUID().uuidString).(extensionName)")
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
            .appendingPathComponent("RIFE60-Reddit-(UUID().uuidString).(ext)")
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.copyItem(at: sourceURL, to: destination)
        return destination
    }

    private static func temporaryGIFURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("RIFE60-Reddit-GIF-(UUID().uuidString).gif")
    }

    private static func fileSize(_ url: URL) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static func formatMB(_ bytes: Int64) -> String {
        String(format: "%.1f MB", Double(bytes) / 1_000_000.0)
    }
}
