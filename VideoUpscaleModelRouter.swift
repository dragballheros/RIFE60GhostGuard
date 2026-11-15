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
        var previousAnalysisImage: CGImage?

        for index in 0..<sampleCount {
            try Task.checkCancellation()
            let seconds = min(Double(index) * interval, max(duration - 0.001, 0))
            let time = CMTime(seconds: seconds, preferredTimescale: 600)
            var actualTime = CMTime.zero
            do {
                let cgImage = try generator.copyCGImage(at: time, actualTime: &actualTime)
                let visual = classify(cgImage)
                let cutScore = previousAnalysisImage.map { sceneCutScore(previous: $0, current: cgImage) } ?? 1.0
                let sceneCutBefore = index == 0 || cutScore >= 0.42
                samples.append(VideoUpscaleSample(
                    timeSeconds: seconds,
                    engine: visual.engine,
                    confidence: visual.confidence,
                    reason: visual.reason,
                    sceneCutBefore: sceneCutBefore
                ))
                if sceneCutBefore && index > 0 {
                    DiagnosticsLogger.shared.log("Upscaler scene boundary detected • time=\(String(format: "%.2f", seconds))s • visualDifference=\(String(format: "%.3f", cutScore))")
                }
                previousAnalysisImage = cgImage
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

    /// A sampled scene boundary requires a large histogram or spatial change.
    /// Consecutive model evidence is still required before the active upscaler changes.
    private static func sceneCutScore(previous: CGImage, current: CGImage) -> Double {
        let side = 32
        func rgba(_ image: CGImage) -> [UInt8]? {
            var bytes = [UInt8](repeating: 0, count: side * side * 4)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let info = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            let created = bytes.withUnsafeMutableBytes { raw -> Bool in
                guard let base = raw.baseAddress,
                      let context = CGContext(
                        data: base,
                        width: side,
                        height: side,
                        bitsPerComponent: 8,
                        bytesPerRow: side * 4,
                        space: colorSpace,
                        bitmapInfo: info
                      ) else { return false }
                context.interpolationQuality = .low
                context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
                return true
            }
            return created ? bytes : nil
        }
        guard let a = rgba(previous), let b = rgba(current) else { return 0 }
        var histA = Array(repeating: 0, count: 64)
        var histB = Array(repeating: 0, count: 64)
        var difference = 0.0
        for pixel in 0..<(side * side) {
            let offset = pixel * 4
            let ar = Int(a[offset]), ag = Int(a[offset + 1]), ab = Int(a[offset + 2])
            let br = Int(b[offset]), bg = Int(b[offset + 1]), bb = Int(b[offset + 2])
            let binA = (ar >> 6) * 16 + (ag >> 6) * 4 + (ab >> 6)
            let binB = (br >> 6) * 16 + (bg >> 6) * 4 + (bb >> 6)
            histA[binA] += 1
            histB[binB] += 1
            difference += Double(abs(ar - br) + abs(ag - bg) + abs(ab - bb)) / (3.0 * 255.0)
        }
        let pixels = Double(side * side)
        var histogramDistance = 0.0
        for index in 0..<64 {
            histogramDistance += abs(Double(histA[index] - histB[index])) / (2.0 * pixels)
        }
        let spatialDifference = difference / pixels
        return min(1.0, max(histogramDistance, spatialDifference * 0.85))
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
        if (try? VNImageRequestHandler(cgImage: image).perform([request])) != nil {
            for observation in (request.results ?? []).prefix(12) {
                let label = observation.identifier.lowercased()
                let confidence = Double(observation.confidence)
                if ["cartoon", "animation", "comic", "manga", "illustration", "drawing", "anime", "graphic"].contains(where: label.contains) {
                    semanticAnime = max(semanticAnime, confidence)
                }
                if ["photograph", "photography", "photo", "realistic", "3d render", "computer generated", "rendering"].contains(where: label.contains) {
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
