import Foundation
import AVFoundation
import CoreGraphics
import CoreVideo
import ImageIO

/// Converts an animated GIF into a persistent temporary MP4 so it can enter the
/// exact same RIFE/CUGAN video pipeline as a normal video. The original GIF timing
/// is carried into the video's presentation timestamps instead of manufacturing
/// a still-image export path.
struct GIFVideoBridge: Sendable {
    static func makeVideo(
        from sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void = { _, _ in }
    ) async throws -> URL {
        try await Task.detached(priority: .userInitiated) {
            try makeVideoSync(sourceURL: sourceURL, progress: progress)
        }.value
    }

    private static func makeVideoSync(
        sourceURL: URL,
        progress: @Sendable (Double, String) -> Void
    ) throws -> URL {
        guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil) else {
            throw GIFVideoBridgeError.invalidGIF
        }

        let frameCount = CGImageSourceGetCount(source)
        guard frameCount > 0,
              let firstFrame = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw GIFVideoBridgeError.invalidGIF
        }

        let width = max(2, firstFrame.width + (firstFrame.width % 2))
        let height = max(2, firstFrame.height + (firstFrame.height % 2))
        let outputRoot = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RIFEGIFBridges", isDirectory: true)
        try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)
        let outputURL = outputRoot.appendingPathComponent("GIF-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: outputURL)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let bitrate = max(6_000_000, min(35_000_000, width * height * 8))
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoExpectedSourceFrameRateKey: 60,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        input.transform = .identity
        guard writer.canAdd(input) else {
            throw GIFVideoBridgeError.couldNotCreateVideo
        }
        writer.add(input)

        let bufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: bufferAttributes
        )

        guard writer.startWriting() else {
            throw writer.error ?? GIFVideoBridgeError.couldNotCreateVideo
        }
        writer.startSession(atSourceTime: .zero)

        var timestamp = 0.0
        do {
            guard let canvasContext = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
            ) else {
                throw GIFVideoBridgeError.couldNotCreatePixelBuffer
            }
            canvasContext.setFillColor(CGColor(gray: 0, alpha: 1))
            canvasContext.fill(CGRect(x: 0, y: 0, width: width, height: height))
            canvasContext.interpolationQuality = .high

            var previousDisposal = 1
            var previousRect = CGRect.zero
            var canvasBeforePreviousFrame: CGImage?

            for index in 0..<frameCount {
                try Task.checkCancellation()

                // Apply the previous frame's disposal instruction before drawing
                // this frame's patch onto the persistent full-canvas image.
                if index > 0 {
                    switch previousDisposal {
                    case 2: // Restore to background.
                        canvasContext.clear(previousRect)
                        canvasContext.setFillColor(CGColor(gray: 0, alpha: 1))
                        canvasContext.fill(previousRect)
                    case 3: // Restore to the canvas before the previous frame.
                        if let saved = canvasBeforePreviousFrame {
                            canvasContext.setBlendMode(.copy)
                            canvasContext.draw(saved, in: CGRect(x: 0, y: 0, width: width, height: height))
                            canvasContext.setBlendMode(.normal)
                        }
                    default:
                        break // Keep the previous composited frame.
                    }
                }

                guard let image = CGImageSourceCreateImageAtIndex(source, index, nil) else {
                    throw GIFVideoBridgeError.invalidFrame(index)
                }
                let properties = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any]
                let gif = properties?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
                let left = max(0, (gif?["ImageLeft" as CFString] as? NSNumber)?.intValue ?? 0)
                let top = max(0, (gif?["ImageTop" as CFString] as? NSNumber)?.intValue ?? 0)
                let frameRect = CGRect(
                    x: left,
                    y: height - top - image.height,
                    width: image.width,
                    height: image.height
                ).intersection(CGRect(x: 0, y: 0, width: width, height: height))

                // Save the canvas state before this frame for disposal method 3.
                canvasBeforePreviousFrame = canvasContext.makeImage()
                canvasContext.draw(image, in: frameRect)
                guard let compositedFrame = canvasContext.makeImage() else {
                    throw GIFVideoBridgeError.couldNotCreatePixelBuffer
                }

                while !input.isReadyForMoreMediaData {
                    try Task.checkCancellation()
                    Thread.sleep(forTimeInterval: 0.001)
                }

                guard let pool = adaptor.pixelBufferPool else {
                    throw GIFVideoBridgeError.couldNotCreatePixelBuffer
                }
                var optionalBuffer: CVPixelBuffer?
                let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optionalBuffer)
                guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
                    throw GIFVideoBridgeError.couldNotCreatePixelBuffer
                }

                CVPixelBufferLockBaseAddress(buffer, [])
                defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
                guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
                    throw GIFVideoBridgeError.couldNotCreatePixelBuffer
                }
                guard let outputContext = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
                ) else {
                    throw GIFVideoBridgeError.couldNotCreatePixelBuffer
                }
                outputContext.setFillColor(CGColor(gray: 0, alpha: 1))
                outputContext.fill(CGRect(x: 0, y: 0, width: width, height: height))
                outputContext.interpolationQuality = .high
                outputContext.draw(compositedFrame, in: CGRect(x: 0, y: 0, width: width, height: height))

                let presentationTime = CMTime(seconds: timestamp, preferredTimescale: 600)
                guard adaptor.append(buffer, withPresentationTime: presentationTime) else {
                    throw writer.error ?? GIFVideoBridgeError.writerRejectedFrame(index)
                }

                let unclamped = gif?[kCGImagePropertyGIFUnclampedDelayTime] as? Double
                let clamped = gif?[kCGImagePropertyGIFDelayTime] as? Double
                let rawDelay = unclamped ?? clamped ?? 0.1
                let delay = rawDelay.isFinite && rawDelay > 0.001 ? min(rawDelay, 655.35) : 0.1
                timestamp += delay
                previousRect = frameRect
                previousDisposal = (gif?["DisposalMethod" as CFString] as? NSNumber)?.intValue ?? 1

                progress(
                    Double(index + 1) / Double(frameCount),
                    "GIF → video • frame \(index + 1)/\(frameCount)"
                )
            }
            input.markAsFinished()
            let semaphore = DispatchSemaphore(value: 0)
            writer.finishWriting { semaphore.signal() }
            semaphore.wait()

            try Task.checkCancellation()
            guard writer.status == .completed, FileManager.default.fileExists(atPath: outputURL.path) else {
                throw writer.error ?? GIFVideoBridgeError.couldNotFinalizeVideo
            }
            progress(1, "GIF converted to video • \(width)x\(height)")
            return outputURL
        } catch {
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: outputURL)
            throw error
        }
    }
}

enum GIFVideoBridgeError: LocalizedError {
    case invalidGIF
    case invalidFrame(Int)
    case couldNotCreateVideo
    case couldNotCreatePixelBuffer
    case writerRejectedFrame(Int)
    case couldNotFinalizeVideo

    var errorDescription: String? {
        switch self {
        case .invalidGIF:
            return "The GIF could not be decoded into an animation."
        case .invalidFrame(let index):
            return "GIF frame (index + 1) could not be decoded."
        case .couldNotCreateVideo:
            return "The GIF could not be converted to the temporary video format."
        case .couldNotCreatePixelBuffer:
            return "The GIF-to-video converter could not allocate a video frame buffer."
        case .writerRejectedFrame(let index):
            return "The temporary video writer rejected GIF frame (index + 1)."
        case .couldNotFinalizeVideo:
            return "The temporary GIF video could not be finalized."
        }
    }
}
