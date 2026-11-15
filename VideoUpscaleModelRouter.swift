import Foundation
import AVFoundation
import Vision
import CoreGraphics

enum UpscaleModelSelection: String, CaseIterable, Identifiable, Sendable {
    case auto
    case realCUGAN
    case realESRGAN

    var id: String { rawValue }
    var title: String {
        switch self {
        case .auto: return "Auto"
        case .realCUGAN: return "Real-CUGAN"
        case .realESRGAN: return "Real-ESRGAN"
        }
    }
    var explanation: String {
        switch self {
        case .auto: return "Analyzes the clip and routes anime/cel-shaded scenes to Real-CUGAN and realistic CGI/movie scenes to Real-ESRGAN. Mixed clips can use both."
        case .realCUGAN: return "Uses Real-CUGAN Anime 2× Noise 3 for the entire clip."
        case .realESRGAN: return "Uses Real-ESRGAN x2plus for the entire clip."
        }
    }
}

enum VideoUpscaleEngine: String, Sendable {
    case realCUGAN
    case realESRGAN

    var displayName: String {
        switch self {
        case .realCUGAN: return "Real-CUGAN"
        case .realESRGAN: return "Real-ESRGAN"
        }
    }
}

struct VideoUpscaleSegment: Sendable, Equatable {
    let startSeconds: Double
    let endSeconds: Double
    let engine: VideoUpscaleEngine
    let confidence: Double
    let reason: String
}

struct VideoUpscaleSample: Sendable, Equatable {
    let timeSeconds: Double
    let engine: VideoUpscaleEngine?
    let confidence: Double
    let reason: String
}

struct VideoUpscalePlan: Sendable {
    let duration: Double
    let segments: [VideoUpscaleSegment]

    func engine(at seconds: Double) -> VideoUpscaleEngine {
        guard !segments.isEmpty else { return .realCUGAN }
        let time = max(0, min(seconds.isFinite ? seconds : 0, duration))
        return segments.first(where: { time >= $0.startSeconds && time < $0.endSeconds })?.engine
            ?? segments.last?.engine
            ?? .realCUGAN
    }

    static func fixed(duration: Double, engine: VideoUpscaleEngine, reason: String) -> VideoUpscalePlan {
        VideoUpscalePlan(
            duration: max(duration, 0.001),
            segments: [VideoUpscaleSegment(startSeconds: 0, endSeconds: max(duration, 0.001), engine: engine, confidence: 1, reason: reason)]
        )
    }

    /// Requires three consecutive confident samples before changing engines.
    /// The switch is backdated to the first sample in that streak, then segments
    /// shorter than the minimum are merged to prevent model flicker.
    static func stabilized(
        duration: Double,
        samples: [VideoUpscaleSample],
        minimumConsecutive: Int = 3,
        minimumSegmentSeconds: Double = 2.0
    ) -> VideoUpscalePlan {
        let safeDuration = max(duration.isFinite ? duration : 0, 0.001)
        guard !samples.isEmpty else {
            return fixed(duration: safeDuration, engine: .realCUGAN, reason: "Auto fallback: insufficient visual evidence")
        }

        let ordered = samples.sorted { $0.timeSeconds < $1.timeSeconds }
        var assigned = Array(repeating: VideoUpscaleEngine.realCUGAN, count: ordered.count)
        var confidences = Array(repeating: 0.0, count: ordered.count)
        var reasons = Array(repeating: "Conservative fallback to Real-CUGAN", count: ordered.count)
        var stable = ordered.first(where: { $0.engine != nil })?.engine ?? .realCUGAN
        var pendingEngine: VideoUpscaleEngine?
        var pendingIndices: [Int] = []

        for (index, sample) in ordered.enumerated() {
            if let candidate = sample.engine, candidate != stable, sample.confidence >= 0.55 {
                if pendingEngine == candidate {
                    pendingIndices.append(index)
                } else {
                    pendingEngine = candidate
                    pendingIndices = [index]
                }
                if pendingIndices.count >= max(2, minimumConsecutive) {
                    stable = candidate
                    for pendingIndex in pendingIndices {
                        assigned[pendingIndex] = stable
                        confidences[pendingIndex] = ordered[pendingIndex].confidence
                        reasons[pendingIndex] = ordered[pendingIndex].reason
                    }
                    pendingEngine = nil
                    pendingIndices.removeAll(keepingCapacity: true)
                } else {
                    assigned[index] = assigned[index == 0 ? 0 : index - 1]
                    confidences[index] = sample.confidence
                    reasons[index] = "Pending temporal confirmation"
                }
            } else {
                pendingEngine = nil
                pendingIndices.removeAll(keepingCapacity: true)
                assigned[index] = stable
                confidences[index] = sample.confidence
                reasons[index] = sample.engine == nil ? "Low confidence; retained previous model" : sample.reason
            }
        }

        // A short terminal candidate cannot be confirmed and remains on the last stable model.
        for index in pendingIndices {
            assigned[index] = stable
            reasons[index] = "Rejected short model-switch candidate"
        }

        var raw: [VideoUpscaleSegment] = []
        for index in ordered.indices {
            let start = max(0, min(ordered[index].timeSeconds, safeDuration))
            let end = index + 1 < ordered.count
                ? max(start, min(ordered[index + 1].timeSeconds, safeDuration))
                : safeDuration
            guard end > start else { continue }
            if let last = raw.last, last.engine == assigned[index] {
                raw[raw.count - 1] = VideoUpscaleSegment(
                    startSeconds: last.startSeconds,
                    endSeconds: end,
                    engine: last.engine,
                    confidence: min(last.confidence, confidences[index]),
                    reason: last.reason == reasons[index] ? last.reason : "Temporally stabilized scene run"
                )
            } else {
                raw.append(VideoUpscaleSegment(
                    startSeconds: start,
                    endSeconds: end,
                    engine: assigned[index],
                    confidence: confidences[index],
                    reason: reasons[index]
                ))
            }
        }

        // Merge micro-segments into the longer adjacent segment. This also prevents a
        // single misclassified analysis window from switching the model.
        var changed = true
        while changed && raw.count > 1 {
            changed = false
            for index in raw.indices where raw[index].endSeconds - raw[index].startSeconds < minimumSegmentSeconds {
                let targetIndex: Int
                if index == 0 { targetIndex = 1 }
                else if index == raw.count - 1 { targetIndex = index - 1 }
                else {
                    let before = raw[index - 1].endSeconds - raw[index - 1].startSeconds
                    let after = raw[index + 1].endSeconds - raw[index + 1].startSeconds
                    targetIndex = before >= after ? index - 1 : index + 1
                }
                let replacement = raw[targetIndex]
                raw[index] = VideoUpscaleSegment(
                    startSeconds: raw[index].startSeconds,
                    endSeconds: raw[index].endSeconds,
                    engine: replacement.engine,
                    confidence: min(raw[index].confidence, replacement.confidence),
                    reason: "Merged short segment to prevent model switching"
                )
                var merged: [VideoUpscaleSegment] = []
                for segment in raw {
                    if let last = merged.last, last.engine == segment.engine {
                        merged[merged.count - 1] = VideoUpscaleSegment(
                            startSeconds: last.startSeconds,
                            endSeconds: segment.endSeconds,
                            engine: last.engine,
                            confidence: min(last.confidence, segment.confidence),
                            reason: "Temporally stabilized scene run"
                        )
                    } else {
                        merged.append(segment)
                    }
                }
                raw = merged
                changed = true
                break
            }
        }

        if raw.isEmpty {
            return fixed(duration: safeDuration, engine: .realCUGAN, reason: "Auto fallback: no stable classifications")
        }
        let normalized = raw.enumerated().map { index, segment in
            VideoUpscaleSegment(
                startSeconds: index == 0 ? 0 : segment.startSeconds,
                endSeconds: index == raw.count - 1 ? safeDuration : segment.endSeconds,
                engine: segment.engine,
                confidence: segment.confidence,
                reason: segment.reason
            )
        }
        return VideoUpscalePlan(duration: safeDuration, segments: normalized)
    }
}

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
