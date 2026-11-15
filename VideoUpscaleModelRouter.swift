import Foundation
import AVFoundation
import Vision
import CoreGraphics

enum VideoUpscaleModelRouter {
    static func makePlan(
        sourceURL: URL,
        selection: UpscaleModelSelection,
        progress: @escaping @Sendable (Double, String) -> Void
    ) async throws -> VideoUpscalePlan {
        let asset = AVURLAsset(url: sourceURL)
        let durationTime = try await asset.load(.duration)
        let duration = max(CMTimeGetSeconds(durationTime), 0.001)

        switch selection {
        case .realCUGAN:
            return .fixed(duration: duration, engine: .realCUGAN, reason: "Manual model selection")
        case .realESRGAN:
            return .fixed(duration: duration, engine: .realESRGAN, reason: "Manual model selection")
        case .auto:
            break
        }

        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            return .fixed(duration: duration, engine: .realCUGAN, reason: "No video track; conservative fallback")
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 160, height: 160)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.15, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.15, preferredTimescale: 600)
        _ = track

        let interval = max(0.75, duration / 240.0)
        let sampleCount = min(240, max(1, Int(ceil(duration / interval))))
        var samples: [VideoUpscaleSample] = []
        samples.reserveCapacity(sampleCount)

        for index in 0..<sampleCount {
            try Task.checkCancellation()
            let seconds = min(Double(index) * interval, max(duration - 0.001, 0))
            let time = CMTime(seconds: seconds, preferredTimescale: 600)
            var actualTime = CMTime.zero
            do {
                let cgImage = try generator.copyCGImage(at: time, actualTime: &actualTime)
                let visual = classify(cgImage)
                samples.append(VideoUpscaleSample(
                    timeSeconds: seconds,
                    engine: visual.engine,
                    confidence: visual.confidence,
                    reason: visual.reason
                ))
            } catch {
                samples.append(VideoUpscaleSample(
                    timeSeconds: seconds,
                    engine: nil,
                    confidence: 0,
                    reason: "Frame analysis unavailable: \(error.localizedDescription)"
                ))
            }
            if index == 0 || index % 8 == 7 || index == sampleCount - 1 {
                progress(Double(index + 1) / Double(sampleCount), "Auto model analysis • sample \(index + 1)/\(sampleCount)")
            }
        }

        let plan = VideoUpscalePlan.stabilized(duration: duration, samples: samples)
        let summary = plan.segments.map {
            "\($0.engine.displayName) \(String(format: "%.2f", $0.startSeconds))–\(String(format: "%.2f", $0.endSeconds))s confidence \(String(format: "%.2f", $0.confidence))"
        }.joined(separator: " | ")
        DiagnosticsLogger.shared.log("AUTO UPSCALE ROUTE • \(summary)")
        CrashCapture.step("Auto upscale route • \(summary)")
        return plan
    }

    private static func classify(_ image: CGImage) -> (engine: VideoUpscaleEngine?, confidence: Double, reason: String) {
        let width = 64
        let height = 64
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let byteCount = bytesPerRow * height
        let pixels = UnsafeMutablePointer<UInt8>.allocate(capacity: byteCount)
        defer { pixels.deallocate() }
        pixels.initialize(repeating: 0, count: byteCount)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return (nil, 0, "Could not sample image pixels")
        }
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var palette = Set<UInt16>()
        var flatCount = 0
        var edgeCount = 0
        var textureCount = 0
        var gradientTotal = 0.0
        let pixelCount = width * height
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * bytesPerPixel
                let r = Int(pixels[offset])
                let g = Int(pixels[offset + 1])
                let b = Int(pixels[offset + 2])
                let key = UInt16((r >> 5) << 10 | (g >> 5) << 5 | (b >> 5))
                palette.insert(key)
                if x + 1 < width && y + 1 < height {
                    let right = offset + bytesPerPixel
                    let down = offset + bytesPerRow
                    let dx = (abs(r - Int(pixels[right])) + abs(g - Int(pixels[right + 1])) + abs(b - Int(pixels[right + 2]))) / 3
                    let dy = (abs(r - Int(pixels[down])) + abs(g - Int(pixels[down + 1])) + abs(b - Int(pixels[down + 2]))) / 3
                    let gradient = (dx + dy) / 2
                    gradientTotal += Double(gradient)
                    if gradient <= 6 { flatCount += 1 }
                    else if gradient >= 30 { edgeCount += 1 }
                    else { textureCount += 1 }
                }
            }
        }

        var semanticAnime = 0.0
        var semanticPhoto = 0.0
        let request = VNClassifyImageRequest()
        request.usesCPUOnly = false
        if (try? VNImageRequestHandler(cgImage: image).perform([request])) != nil {
            for observation in (request.results ?? []).prefix(12) {
                let label = observation.identifier.lowercased()
                let confidence = Double(observation.confidence)
                if ["cartoon", "animation", "comic", "manga", "illustration", "drawing", "anime", "graphic"].contains(where: label.contains) {
                    semanticAnime = max(semanticAnime, confidence)
                }
                if ["photograph", "photography", "photo", "realistic", "metal", "machine", "robot", "vehicle", "architecture"].contains(where: label.contains) {
                    semanticPhoto = max(semanticPhoto, confidence)
                }
            }
        }

        let denom = Double(max(pixelCount, 1))
        let flat = Double(flatCount) / denom
        let edges = Double(edgeCount) / denom
        let texture = Double(textureCount) / denom
        let paletteRichness = min(Double(palette.count) / 1400.0, 1.0)
        let meanGradient = gradientTotal / denom

        let animeScore = min(1, max(0,
            flat * 0.34 +
            edges * 0.28 +
            (1 - paletteRichness) * 0.18 +
            semanticAnime * 0.42
        ))
        let photoScore = min(1, max(0,
            texture * 0.28 +
            paletteRichness * 0.24 +
            min(meanGradient / 34.0, 1.0) * 0.18 +
            semanticPhoto * 0.42
        ))
        let difference = abs(animeScore - photoScore)
        let confidence = min(0.99, max(0, 0.5 + difference * 0.65 + max(semanticAnime, semanticPhoto) * 0.18))

        if animeScore >= 0.40 && animeScore > photoScore + 0.075 {
            return (.realCUGAN, confidence, String(format: "Anime/cel cues • anime %.2f, CGI/photo %.2f", animeScore, photoScore))
        }
        if photoScore >= 0.43 && photoScore > animeScore + 0.075 {
            return (.realESRGAN, confidence, String(format: "Movie/CGI cues • anime %.2f, CGI/photo %.2f", animeScore, photoScore))
        }
        return (nil, min(confidence, 0.54), String(format: "Ambiguous visual sample • anime %.2f, CGI/photo %.2f", animeScore, photoScore))
    }
}
