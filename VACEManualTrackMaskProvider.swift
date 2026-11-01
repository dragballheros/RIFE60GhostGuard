import CoreMedia
import Foundation

/// Tracks one drawn or tapped keyframe through the clip. Each later frame
/// re-selects the anime region under the previous mask's centre. A shot cut
/// drops the track and asks the anime segmenter for the subject again.
final class VACEManualTrackMaskProvider: VACEFrameMaskProvider {
    private let seed: VACEMaskPlane
    private var previousMask: VACEMaskPlane?
    private var previousSignature: VACEFrameSignature?

    init(seed: VACEMaskPlane) {
        self.seed = seed
    }

    func mask(for frameIndex: Int, presentationTime: CMTime, source: VACERGBAFrame) async throws -> VACEMaskPlane {
        _ = frameIndex
        _ = presentationTime
        let fitted = try fit(seed, to: source)
        let signature = VACEShotCut.signature(of: source)
        guard let previousMask, let previousSignature else {
            self.previousMask = fitted
            self.previousSignature = signature
            CrashCapture.step("VACE manual-tracked keyframe coverage \(Int(VACESubjectSegmenter.coverage(of: fitted.pixels) * 100))%")
            return fitted
        }
        let comparison = VACEShotCut.isCut(from: previousSignature, to: signature)
        self.previousSignature = signature
        if comparison.cut {
            VACEShotCut.logCut(at: presentationTime, histogramDistance: comparison.histogramDistance, frameDifference: comparison.frameDifference)
            let decision = try VACESubjectSegmenter.editMask(frame: source, mode: .autoSubject)
            let mask = decision.foundSubject ? decision.mask : fitted
            self.previousMask = mask
            CrashCapture.step("VACE manual-tracked re-anchored at a shot cut with \(decision.method)")
            return mask
        }
        let point = centroid(of: previousMask)
        guard let image = VACESubjectSegmenter.cgImage(from: source) else {
            return previousMask
        }
        let decision = try VACESubjectSegmenter.tapSelect(cgImage: image, x: point.x, y: point.y)
        let mask = decision.foundSubject ? decision.mask : previousMask
        self.previousMask = mask
        return mask
    }

    private func fit(_ mask: VACEMaskPlane, to frame: VACERGBAFrame) throws -> VACEMaskPlane {
        if mask.width == frame.width && mask.height == frame.height { return mask }
        guard let image = cgImage(mask),
              let context = CGContext(
                data: nil, width: frame.width, height: frame.height,
                bitsPerComponent: 8, bytesPerRow: frame.width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
              ) else {
            throw VACESubjectSegmentationError.unreadableFrame
        }
        context.interpolationQuality = .none
        context.draw(image, in: CGRect(x: 0, y: 0, width: frame.width, height: frame.height))
        guard let data = context.data else { throw VACESubjectSegmentationError.unreadableFrame }
        let pixels = Array(UnsafeBufferPointer(start: data.bindMemory(to: UInt8.self, capacity: frame.width * frame.height), count: frame.width * frame.height))
        return try VACEMaskPlane(width: frame.width, height: frame.height, pixels: pixels)
    }

    private func cgImage(_ mask: VACEMaskPlane) -> CGImage? {
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

    private func centroid(of mask: VACEMaskPlane) -> (x: Int, y: Int) {
        var sumX = 0
        var sumY = 0
        var count = 0
        for y in 0..<mask.height {
            for x in 0..<mask.width where mask.pixels[y * mask.width + x] > 127 {
                sumX += x
                sumY += y
                count += 1
            }
        }
        guard count > 0 else { return (mask.width / 2, mask.height / 2) }
        return (sumX / count, sumY / count)
    }
}
