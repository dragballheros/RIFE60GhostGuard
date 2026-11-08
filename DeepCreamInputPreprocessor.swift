import Foundation
import CoreGraphics

/// Preserves the checkpoint-specific input convention of DeepCreamPy v2.
/// The mosaic generator needs the actual mosaic pixels as a clue; only the bar
/// generator receives a blanked region. Both models still receive a 1=preserve,
/// 0=inpaint mask as their third input tensor.
enum DeepCreamInputPreprocessor {
    /// Mirrors DeepCreamPy's context-crop policy: keep a square patch centered
    /// on the marked region, at least 256 px, and about 1.5× the target's longest
    /// dimension when the frame allows. Never crop away the full selected region.
    static func contextCropSide(
        targetWidth: CGFloat,
        targetHeight: CGFloat,
        frameWidth: Int,
        frameHeight: Int
    ) -> Int {
        let targetSide = max(targetWidth, targetHeight)
        let preferredSide = max(CGFloat(256), ceil(targetSide * 1.5))
        let frameLimit = CGFloat(min(frameWidth, frameHeight))
        return Int(ceil(max(targetSide, min(preferredSide, frameLimit))))
    }

    static func maskedInputPixel(
        sourceValue: Float,
        knownMaskValue: Float,
        model: WatermarkModel
    ) -> Float {
        if model == .deepCreamMosaic {
            // Match official DeepCreamPy decensor.py: mosaic inputs are not
            // zeroed at the marked pixels; their pixel pattern guides recovery.
            return sourceValue
        }
        return knownMaskValue == 0 ? -1.0 : sourceValue
    }
}
