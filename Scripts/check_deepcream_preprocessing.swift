import Foundation

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        fatalError("DEEP CREAM PREPROCESSING FAIL: \(message)")
    }
}

let sourcePixel: Float = 0.42
let mosaicMasked = DeepCreamInputPreprocessor.maskedInputPixel(
    sourceValue: sourcePixel,
    knownMaskValue: 0,
    model: .deepCreamMosaic
)
require(mosaicMasked == sourcePixel, "mosaic pixels must remain visible to the model inside the repair mask")

let barMasked = DeepCreamInputPreprocessor.maskedInputPixel(
    sourceValue: sourcePixel,
    knownMaskValue: 0,
    model: .deepCreamBar
)
require(barMasked == -1.0, "bar-censor pixels must be blanked before inpainting")

let mosaicKnown = DeepCreamInputPreprocessor.maskedInputPixel(
    sourceValue: sourcePixel,
    knownMaskValue: 1,
    model: .deepCreamMosaic
)
let barKnown = DeepCreamInputPreprocessor.maskedInputPixel(
    sourceValue: sourcePixel,
    knownMaskValue: 1,
    model: .deepCreamBar
)
require(mosaicKnown == sourcePixel && barKnown == sourcePixel,
        "known context pixels must stay byte-normalized source values for both models")

print("DEEP_CREAM_PREPROCESSING_PASS: mosaic texture preserved; bar mask blanked; known context preserved")
