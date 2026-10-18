// Standalone macOS regression check for the shared iOS 16-compatible grade.
// CI copies this file to main.swift and compiles it with ColorPopGrade.swift.
import Foundation
import CoreVideo

enum ProcessorError: Error {
    case conversionFailed(String)
}

func makePatch(_ rgb: (Int, Int, Int), width: Int = 32, height: Int = 32) throws -> CVPixelBuffer {
    var result: CVPixelBuffer?
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferMetalCompatibilityKey as String: true]
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &result) == kCVReturnSuccess,
          let result else { throw ProcessorError.conversionFailed("test allocation") }
    CVPixelBufferLockBaseAddress(result, [])
    defer { CVPixelBufferUnlockBaseAddress(result, []) }
    guard let address = CVPixelBufferGetBaseAddress(result) else { throw ProcessorError.conversionFailed("test address") }
    let bytes = address.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(result)
    for y in 0..<height {
        for x in 0..<width {
            let offset = y * stride + x * 4
            bytes[offset] = UInt8(rgb.2)
            bytes[offset + 1] = UInt8(rgb.1)
            bytes[offset + 2] = UInt8(rgb.0)
            bytes[offset + 3] = 255
        }
    }
    return result
}

func readPatch(_ buffer: CVPixelBuffer) throws -> (Int, Int, Int) {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw ProcessorError.conversionFailed("test output address") }
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    return (Int(bytes[2]), Int(bytes[1]), Int(bytes[0]))
}

func metrics(_ rgb: (Int, Int, Int)) -> (hue: Double, saturation: Double, luma: Double) {
    let r = Double(rgb.0) / 255, g = Double(rgb.1) / 255, b = Double(rgb.2) / 255
    let high = max(r, max(g, b)), low = min(r, min(g, b)), delta = high - low
    var hue = 0.0
    if delta > 0 {
        if high == r { hue = (g - b) / delta * 60 }
        else if high == g { hue = (2 + (b - r) / delta) * 60 }
        else { hue = (4 + (r - g) / delta) * 60 }
        if hue < 0 { hue += 360 }
    }
    return (hue, high > 0 ? delta / high : 0, 0.2126 * r + 0.7152 * g + 0.0722 * b)
}

func hueDistance(_ lhs: Double, _ rhs: Double) -> Double {
    let delta = abs(lhs - rhs).truncatingRemainder(dividingBy: 360)
    return min(delta, 360 - delta)
}

func assertSkinPreserved(_ source: (Int, Int, Int), output: (Int, Int, Int), label: String, maxSaturationIncrease: Double) {
    let before = metrics(source), after = metrics(output)
    precondition(abs(after.luma - before.luma) < 0.012, "\(label) luminance shifted: \(before.luma) -> \(after.luma), output=\(output)")
    precondition(hueDistance(after.hue, before.hue) < 4.0, "\(label) hue shifted: \(before.hue) -> \(after.hue), output=\(output)")
    precondition(after.saturation - before.saturation < maxSaturationIncrease, "\(label) became oversaturated: \(before.saturation) -> \(after.saturation), output=\(output)")
}

let source = try makePatch((240, 190, 160))
let bypass = try ColorPopGrade(strength: 0, warmthMode: .off).apply(source)
precondition(Unmanaged.passUnretained(source).toOpaque() == Unmanaged.passUnretained(bypass).toOpaque(), "zero strength must return the identical buffer")

let skinSamples = [
    ("peach skin", (240, 190, 160), 0.035),
    ("shaded skin", (120, 82, 66), 0.045),
    ("light skin", (250, 237, 230), 0.060)
]
let accentSamples = [
    ("red", (245, 65, 90)),
    ("pink", (235, 70, 180)),
    ("blue", (80, 135, 230)),
    ("teal", (40, 190, 180)),
    ("green", (60, 190, 100))
]

for strength in [0.5, 1.0] {
    let grade = ColorPopGrade(strength: strength)

    // Neutral pixels and highlight endpoints must not acquire a tint or change.
    for gray in 0...255 {
        let output = try readPatch(grade.apply(makePatch((gray, gray, gray))))
        precondition(abs(output.0 - gray) <= 2 && abs(output.1 - gray) <= 2 && abs(output.2 - gray) <= 2, "gray ramp changed: \(gray) -> \(output)")
        precondition(max(output.0, max(output.1, output.2)) - min(output.0, min(output.1, output.2)) <= 1, "gray acquired a tint: \(gray) -> \(output)")
        if gray == 0 || gray == 255 { precondition(output.0 == gray && output.1 == gray && output.2 == gray, "black/white endpoint changed: \(gray) -> \(output)") }
    }

    for (label, sample, maxSatIncrease) in skinSamples {
        let output = try readPatch(grade.apply(makePatch(sample)))
        assertSkinPreserved(sample, output: output, label: label, maxSaturationIncrease: maxSatIncrease)
    }

    // Key saturated anime colors should gain chroma without large luma or hue shifts.
    for (label, sample) in accentSamples {
        let output = try readPatch(grade.apply(makePatch(sample)))
        let before = metrics(sample), after = metrics(output)
        precondition(after.saturation > before.saturation + 0.008, "\(label) failed to gain vibrance: \(sample) -> \(output)")
        precondition(abs(after.luma - before.luma) < 0.018, "\(label) luminance moved too far: \(sample) -> \(output)")
        precondition(hueDistance(after.hue, before.hue) < 4.0, "\(label) hue shifted: \(sample) -> \(output)")
        print("strength=\(strength) \(label) \(sample) -> \(output)")
    }
    print("strength=\(strength) skin/highlight/gray protections passed")
}

// A runner timing is informative, not a performance promise for an iPhone.
let grade4K = ColorPopGrade(strength: 0.5)
let patch4K = try makePatch((80, 135, 230), width: 3840, height: 2160)
_ = try grade4K.apply(patch4K) // warm the filter graph and pool
let start = CFAbsoluteTimeGetCurrent()
for _ in 0..<5 { _ = try grade4K.apply(patch4K) }
print(String(format: "Color Pop warmed 4K render: %.2f ms/frame on this CI runner", (CFAbsoluteTimeGetCurrent() - start) * 1000 / 5))
// Automatic warmth correction must remove a detected grade-added warm cast,
// while a no-change frame must remain an exact zero-strength bypass.
let warmthReference = try makePatch((120, 120, 120), width: 128, height: 128)
let warmthGraded = try makePatch((130, 120, 110), width: 128, height: 128)
let warmthCorrection = AutoWarmthCorrection()
let correctedWarmth = try readPatch(warmthCorrection.apply(reference: warmthReference, graded: warmthGraded))
print("Warmth regression output RGB=\(correctedWarmth)")
precondition(correctedWarmth.0 - correctedWarmth.2 < 20, "automatic warmth correction failed to reduce a measured warm shift: \(correctedWarmth)")
let unchangedWarmth = try AutoWarmthCorrection().apply(reference: warmthReference, graded: warmthReference)
precondition(Unmanaged.passUnretained(warmthReference).toOpaque() == Unmanaged.passUnretained(unchangedWarmth).toOpaque(), "neutral source/output must bypass warmth correction")
print("Automatic warmth correction checks passed • warm cast reduced • neutral bypass preserved")


// Existing source warmth must also be reduced when the grade itself adds no shift.
// Use a lightly warm, low-chroma midtone so this checks the source-bias branch,
// not the grade-added branch tested above. Skin/accent protections are checked
// independently below; this sample must show a measurable, conservative change.
let sourceWarm = try makePatch((160, 154, 148), width: 128, height: 128)
let sourceWarmOutput = try readPatch(AutoWarmthCorrection().apply(reference: sourceWarm, graded: sourceWarm))
let sourceWarmBefore = try readPatch(sourceWarm)
let sourceWarmSpreadBefore = sourceWarmBefore.0 - sourceWarmBefore.2
let sourceWarmSpreadAfter = sourceWarmOutput.0 - sourceWarmOutput.2
print("Source warmth regression RGB=\(sourceWarmOutput), red-blue spread=\(sourceWarmSpreadBefore)->\(sourceWarmSpreadAfter)")
precondition(sourceWarmSpreadAfter < sourceWarmSpreadBefore, "automatic warmth correction did not reduce existing source warmth: \(sourceWarmBefore) -> \(sourceWarmOutput)")
precondition(sourceWarmSpreadBefore - sourceWarmSpreadAfter >= 1, "existing source warmth reduction was not measurable: \(sourceWarmBefore) -> \(sourceWarmOutput)")

 // The four selectable modes must target distinct warmth sources.
 let modeReference = try makePatch((120, 120, 120), width: 128, height: 128)
 let addedWarmth = try makePatch((130, 120, 110), width: 128, height: 128)
 let originalOnlyAdded = try AutoWarmthCorrection().apply(reference: modeReference, graded: addedWarmth, mode: .originalOnly)
 precondition(Unmanaged.passUnretained(originalOnlyAdded).toOpaque() == Unmanaged.passUnretained(addedWarmth).toOpaque(), "Original-only mode changed warmth that exists only in the processed result")
 let processedOnlyAdded = try readPatch(AutoWarmthCorrection().apply(reference: modeReference, graded: addedWarmth, mode: .processedOnly))
 precondition(processedOnlyAdded.0 - processedOnlyAdded.2 < 20, "Processed-only mode failed to reduce processing-added warmth: \(processedOnlyAdded)")
 let processedOnlySource = try AutoWarmthCorrection().apply(reference: sourceWarm, graded: sourceWarm, mode: .processedOnly)
 precondition(Unmanaged.passUnretained(processedOnlySource).toOpaque() == Unmanaged.passUnretained(sourceWarm).toOpaque(), "Processed-only mode altered original warmth without an added processing shift")
 let originalOnlySource = try readPatch(AutoWarmthCorrection().apply(reference: sourceWarm, graded: sourceWarm, mode: .originalOnly))
 precondition(originalOnlySource.0 - originalOnlySource.2 < sourceWarmSpreadBefore, "Original-only mode failed to reduce original source warmth")
 let gradeOriginalOnly = try readPatch(ColorPopGrade(strength: 0, warmthMode: .originalOnly).apply(sourceWarm))
 precondition(gradeOriginalOnly.0 - gradeOriginalOnly.2 < sourceWarmSpreadBefore, "Zero-strength Color Pop grade did not apply Original-only warmth correction")
 let gradeOff = try ColorPopGrade(strength: 0, warmthMode: .off, skinToneMode: .off).apply(sourceWarm)
 precondition(Unmanaged.passUnretained(gradeOff).toOpaque() == Unmanaged.passUnretained(sourceWarm).toOpaque(), "Zero-strength Color Pop grade must preserve exact input in Off mode")
 let fullSource = try readPatch(AutoWarmthCorrection().apply(reference: sourceWarm, graded: sourceWarm, mode: .full))
 precondition(fullSource.0 - fullSource.2 < sourceWarmSpreadBefore, "Full mode no longer reduces original source warmth")
 let offSource = try AutoWarmthCorrection().apply(reference: sourceWarm, graded: sourceWarm, mode: .off)
 precondition(Unmanaged.passUnretained(offSource).toOpaque() == Unmanaged.passUnretained(sourceWarm).toOpaque(), "Off mode must be an exact identity bypass")
 print("Four-mode warmth checks passed • Off / Original / Processed / Full")

let brightSkin = try makePatch((250, 210, 185), width: 128, height: 128)
let brightSkinBefore = metrics(try readPatch(brightSkin)).luma
let skinOff = try ColorPopGrade(strength: 0, warmthMode: .off, skinToneMode: .off).apply(brightSkin)
precondition(Unmanaged.passUnretained(skinOff).toOpaque() == Unmanaged.passUnretained(brightSkin).toOpaque(), "Color Pop skin correction Off must be an exact identity bypass")
let skinOriginalOnly = try readPatch(ColorPopGrade(strength: 0, warmthMode: .off, skinToneMode: .originalOnly).apply(brightSkin))
precondition(metrics(skinOriginalOnly).luma < brightSkinBefore - 0.002, "Original-only skin correction did not lower already-bright source skin")
let skinProcessedOnly = try readPatch(ColorPopGrade(strength: 0, warmthMode: .off, skinToneMode: .processedOnly).apply(brightSkin))
precondition(abs(metrics(skinProcessedOnly).luma - brightSkinBefore) < 0.002, "Processed-only skin correction changed source skin without a Color Pop brightness increase")
let skinFull = try readPatch(ColorPopGrade(strength: 0, warmthMode: .off, skinToneMode: .full).apply(brightSkin))
precondition(metrics(skinFull).luma < brightSkinBefore - 0.002, "Full skin correction did not correct bright source skin")
print("Color Pop skin-tone modes passed • Off / Original / Processed / Full")




print("Color Pop checks passed")
