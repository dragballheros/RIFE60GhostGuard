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

let smallContext = DeepCreamInputPreprocessor.contextCropSide(
    targetWidth: 150,
    targetHeight: 90,
    frameWidth: 1920,
    frameHeight: 1080
)
require(smallContext == 256, "small targets must retain at least the checkpoint's 256 px context")

let typicalContext = DeepCreamInputPreprocessor.contextCropSide(
    targetWidth: 400,
    targetHeight: 120,
    frameWidth: 1920,
    frameHeight: 1080
)
require(typicalContext == 600, "normal targets should use the official 1.5x context crop, not a 2x crop that unnecessarily downsamples the repair")

let tallFrameLimit = DeepCreamInputPreprocessor.contextCropSide(
    targetWidth: 900,
    targetHeight: 500,
    frameWidth: 1920,
    frameHeight: 1080
)
require(tallFrameLimit == 1080, "context crop should honor the short frame dimension when the target still fits")

print("DEEP_CREAM_PREPROCESSING_PASS: mosaic texture preserved; bar mask blanked; known context preserved")
