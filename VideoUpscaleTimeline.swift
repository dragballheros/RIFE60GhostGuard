import Foundation

enum UpscaleModelSelection: String, CaseIterable, Identifiable, Sendable, Equatable {
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

enum VideoUpscaleEngine: String, Sendable, Equatable {
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
    let sceneCutBefore: Bool

    init(
        timeSeconds: Double,
        engine: VideoUpscaleEngine?,
        confidence: Double,
        reason: String,
        sceneCutBefore: Bool = false
    ) {
        self.timeSeconds = timeSeconds
        self.engine = engine
        self.confidence = confidence
        self.reason = reason
        self.sceneCutBefore = sceneCutBefore
    }
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
        var stable = ordered.first?.engine ?? .realCUGAN
        var pendingEngine: VideoUpscaleEngine?
        var pendingIndices: [Int] = []

        for (index, sample) in ordered.enumerated() {
            if let candidate = sample.engine, candidate != stable, sample.confidence >= 0.54 {
                if pendingEngine == candidate {
                    pendingIndices.append(index)
                } else if sample.sceneCutBefore {
                    pendingEngine = candidate
                    pendingIndices = [index]
                } else {
                    pendingEngine = nil
                    pendingIndices.removeAll(keepingCapacity: true)
                }
                if pendingEngine == candidate, pendingIndices.count >= max(2, minimumConsecutive) {
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
                    reasons[index] = sample.sceneCutBefore
                        ? "Model switch pending scene confirmation"
                        : "Retained current model until a scene cut"
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

