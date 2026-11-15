import Foundation

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        fputs("ADAPTIVE_OUTLINE_FAIL: \(message)\n", stderr)
        exit(1)
    }
}

let fine = AdaptiveOutlineBlend.targetWeight(narrowDetailStrength: 1)
let broad = AdaptiveOutlineBlend.targetWeight(narrowDetailStrength: 0)
let middle = AdaptiveOutlineBlend.targetWeight(narrowDetailStrength: 0.5)

require(abs(fine - 0.82) < 0.000001, "thin details must retain the current 0.82 contribution")
require(abs(broad - 0.94) < 0.000001, "broad contours must restore the former 0.94 contribution")
require(abs(middle - 0.88) < 0.000001, "intermediate widths must blend smoothly")
require(fine < middle && middle < broad, "outline contribution must increase continuously with contour width")
require(abs(AdaptiveOutlineBlend.modelAmountScale(narrowDetailStrength: 1) * broad - fine) < 0.000001,
        "fine contour effective contribution must remain 0.82 after the global 0.94 blend")
require(abs(AdaptiveOutlineBlend.modelAmountScale(narrowDetailStrength: 0) * broad - broad) < 0.000001,
        "broad contour effective contribution must remain 0.94")
require(abs(AdaptiveOutlineBlend.targetWeight(narrowDetailStrength: -1) - broad) < 0.000001,
        "negative confidence must clamp to broad contour setting")
require(abs(AdaptiveOutlineBlend.targetWeight(narrowDetailStrength: 2) - fine) < 0.000001,
        "confidence above one must clamp to fine contour setting")
require(abs(AdaptiveOutlineBlend.targetWeight(narrowDetailStrength: .nan) - broad) < 0.000001,
        "non-finite confidence must safely use broad default")

print("ADAPTIVE_OUTLINE_PASS: broad=0.94, fine=0.82, continuous width blend, bounded confidence")
