import CoreMedia
import Foundation

struct VACEMaskKeyframe {
    var frameIndex: Int
    var mask: VACEMaskPlane
    var appearance: [Float]
}

struct VACETrackedFrame {
    var mask: VACEMaskPlane
    var confidence: Double
    var anchorFrame: Int
    var usedFallback: Bool
}

/// One drawn mask identifies the target. Later frames are chosen by optical flow
/// plus a fresh anime segmentation, scored against the previous mask. A shot cut
/// is re-acquired from the seeded colour histogram, not from where the stroke sat.
final class VACEManualTrackMaskProvider: VACEFrameMaskProvider {
    private var keyframes: [VACEMaskKeyframe]
    private var previousImage: VACERGBAFrame?
    private var previousMask: VACEMaskPlane?
    private var previousSignature: VACEFrameSignature?
    private var previousIndex = -1
    private(set) var anchorFrame = 0

    init(keyframes: [VACEMaskKeyframe]) {
        self.keyframes = keyframes.sorted { $0.frameIndex < $1.frameIndex }
        self.anchorFrame = self.keyframes.first?.frameIndex ?? 0
    }

    func mask(for frameIndex: Int, presentationTime: CMTime, source: VACERGBAFrame) async throws -> VACEMaskPlane {
        guard let image = VACESubjectSegmenter.cgImage(from: source) else {
            throw VACESubjectSegmentationError.unreadableFrame
        }
        let tracked = try track(frameIndex: frameIndex, presentationTime: presentationTime, image: image, rgba: source)
        return tracked.mask
    }

    func track(frameIndex: Int, presentationTime: CMTime, image: CGImage, rgba: VACERGBAFrame) throws -> VACETrackedFrame {
        let signature = VACEShotCut.signature(of: rgba)
        let cut: Bool
        if let previousSignature {
            let comparison = VACEShotCut.isCut(from: previousSignature, to: signature)
            cut = comparison.cut
            if cut {
                VACEShotCut.logCut(at: presentationTime, histogramDistance: comparison.histogramDistance, frameDifference: comparison.frameDifference)
            }
        } else {
            cut = false
        }
        self.previousSignature = signature

        if let keyframe = keyframes.first(where: { $0.frameIndex == frameIndex }) {
            let seeded = try seed(image: image, guide: keyframe.mask)
            anchorFrame = frameIndex
            remember(index: frameIndex, image: rgba, mask: seeded.mask)
            log(frameIndex: frameIndex, confidence: seeded.confidence, fallback: false, note: "keyframe")
            return VACETrackedFrame(mask: seeded.mask, confidence: seeded.confidence, anchorFrame: frameIndex, usedFallback: false)
        }

        let seedAppearance = keyframes.last(where: { $0.frameIndex <= frameIndex })?.appearance
            ?? keyframes.first?.appearance
            ?? []
        let canFlow = previousIndex == frameIndex - 1 && !cut
        let warped: VACEMaskPlane?
        if canFlow, let previousImage, let previousMask {
            warped = try? VACEOpticalFlowMaskProvider.warpMask(previousMask, from: previousImage, to: rgba)
        } else {
            warped = nil
        }

        let instances = (try? VACESubjectSegmenter.characterInstances(cgImage: image, guide: nil)) ?? []
        var bestScore = 0.0
        var bestPixels: [UInt8]?
        for instance in instances {
            let appearance = VACESubjectSegmenter.appearanceHistogram(cgImage: image, mask: instance.pixels)
            let appearanceScore = VACESubjectSegmenter.histogramSimilarity(appearance, seedAppearance)
            let overlapScore: Double
            if let warped {
                overlapScore = overlap(instance.pixels, warped.pixels)
            } else {
                overlapScore = appearanceScore
            }
            let score = cut || warped == nil ? appearanceScore : (0.55 * overlapScore + 0.45 * appearanceScore)
            if score > bestScore {
                bestScore = score
                bestPixels = instance.pixels
            }
        }

        let lowConfidence = bestScore < 0.28 || bestPixels == nil
        let chosen: VACEMaskPlane
        if lowConfidence {
            let fallback = try VACESubjectSegmenter.editMask(cgImage: image, mode: .autoSubject)
            chosen = fallback.foundSubject ? fallback.mask : (previousMask ?? blank(like: rgba))
            log(frameIndex: frameIndex, confidence: bestScore, fallback: true, note: cut ? "shot-cut" : "low-confidence")
        } else {
            chosen = try VACEMaskPlane(width: image.width, height: image.height, pixels: bestPixels ?? [])
            log(frameIndex: frameIndex, confidence: bestScore, fallback: false, note: cut ? "shot-cut-appearance" : "flow-and-segment")
        }
        if let key = keyframes.last(where: { $0.frameIndex <= frameIndex }) {
            anchorFrame = key.frameIndex
        }
        remember(index: frameIndex, image: rgba, mask: chosen)
        return VACETrackedFrame(mask: chosen, confidence: bestScore, anchorFrame: anchorFrame, usedFallback: lowConfidence)
    }

    private func seed(image: CGImage, guide: VACEMaskPlane) throws -> (mask: VACEMaskPlane, confidence: Double) {
        let fitted = try fit(guide, width: image.width, height: image.height)
        let instances = try VACESubjectSegmenter.characterInstances(cgImage: image, guide: fitted.pixels)
        guard !instances.isEmpty else {
            return (fitted, 1)
        }
        var pixels = [UInt8](repeating: 0, count: image.width * image.height)
        for instance in instances {
            for index in pixels.indices where instance.pixels[index] > 127 {
                pixels[index] = 255
            }
        }
        return (try VACEMaskPlane(width: image.width, height: image.height, pixels: pixels), 1)
    }

    private func overlap(_ lhs: [UInt8], _ rhs: [UInt8]) -> Double {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var intersection = 0
        var union = 0
        for index in 0..<count {
            let left = lhs[index] > 127
            let right = rhs[index] > 127
            if left || right { union += 1 }
            if left && right { intersection += 1 }
        }
        guard union > 0 else { return 0 }
        return Double(intersection) / Double(union)
    }

    private func remember(index: Int, image: VACERGBAFrame, mask: VACEMaskPlane) {
        previousIndex = index
        previousImage = image
        previousMask = mask
    }

    private func log(frameIndex: Int, confidence: Double, fallback: Bool, note: String) {
        let line = String(format: "VACE track frame %d confidence %.2f fallback %@ %@", frameIndex, confidence, fallback ? "yes" : "no", note)
        DiagnosticsLogger.shared.log(line)
        CrashCapture.step(line)
    }

    private func blank(like frame: VACERGBAFrame) -> VACEMaskPlane {
        if let previousMask { return previousMask }
        return try! VACEMaskPlane(width: frame.width, height: frame.height, pixels: [UInt8](repeating: 0, count: frame.width * frame.height))
    }

    private func fit(_ mask: VACEMaskPlane, width: Int, height: Int) throws -> VACEMaskPlane {
        if mask.width == width && mask.height == height { return mask }
        guard let image = grayImage(mask),
              let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
              ) else {
            throw VACESubjectSegmentationError.unreadableFrame
        }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { throw VACESubjectSegmentationError.unreadableFrame }
        let pixels = Array(UnsafeBufferPointer(start: data.bindMemory(to: UInt8.self, capacity: width * height), count: width * height))
        return try VACEMaskPlane(width: width, height: height, pixels: pixels)
    }

    private func grayImage(_ mask: VACEMaskPlane) -> CGImage? {
        let data = Data(mask.pixels) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(
            width: mask.width, height: mask.height,
            bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: mask.width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
