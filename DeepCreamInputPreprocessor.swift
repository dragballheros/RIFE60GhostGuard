import Foundation

/// Preserves the checkpoint-specific input convention of DeepCreamPy v2.
/// The mosaic generator needs the actual mosaic pixels as a clue; only the bar
/// generator receives a blanked region. Both models still receive a 1=preserve,
/// 0=inpaint mask as their third input tensor.
enum DeepCreamInputPreprocessor {
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
