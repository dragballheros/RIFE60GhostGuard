// Standalone macOS smoke check for the shared iOS 16-compatible grade.
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
    let high = max(r, g, b), low = min(r, g, b), delta = high - low
    var hue = 0.0
    if delta > 0 {
        if high == r { hue = (g - b) / delta * 60 }
        else if high == g { hue = (2 + (b - r) / delta) * 60 }
        else { hue = (4 + (r - g) / delta) * 60 }
        if hue < 0 { hue += 360 }
    }
    return (hue, high > 0 ? delta / high : 0, 0.2126 * r + 0.7152 * g + 0.0722 * b)
}

let source = try makePatch((240, 190, 160))
let bypass = try ColorPopGrade(strength: 0).apply(source)
precondition(Unmanaged.passUnretained(source).toOpaque() == Unmanaged.passUnretained(bypass).toOpaque(), "zero strength must return the identical buffer")

for strength in [0.5, 1.0] {
    let grade = ColorPopGrade(strength: strength)
    for gray in 0...255 {
        let output = try readPatch(grade.apply(makePatch((gray, gray, gray))))
        precondition(abs(output.0 - gray) <= 1 && abs(output.1 - gray) <= 1 && abs(output.2 - gray) <= 1, "gray ramp changed: \(gray) -> \(output)")
        precondition(output.0 == output.1 && output.1 == output.2, "gray acquired a tint")
        if gray == 0 || gray == 255 { precondition(output.0 == gray, "black/white changed") }
    }
    let skin = try readPatch(grade.apply(source))
    let before = metrics((240, 190, 160)), after = metrics(skin)
    precondition(after.luma > before.luma && after.hue < before.hue && after.saturation < before.saturation, "skin must brighten, desaturate and move away from orange: \(skin)")
    for color in [(80, 135, 230), (60, 190, 100)] {
        let result = try readPatch(grade.apply(makePatch(color)))
        precondition(metrics(result).saturation > metrics(color).saturation, "colored patch did not gain vibrance: \(color) -> \(result)")
        print("strength=\(strength) color \(color) -> \(result)")
    }
    print("strength=\(strength) skin (240,190,160) -> \(skin); neutral ramp/black/white pass")
}

// A runner timing is informative, not a performance promise for an iPhone.
let grade4K = ColorPopGrade(strength: 0.5)
let patch4K = try makePatch((80, 135, 230), width: 3840, height: 2160)
_ = try grade4K.apply(patch4K) // warm the filter graph and pool
let start = CFAbsoluteTimeGetCurrent()
for _ in 0..<5 { _ = try grade4K.apply(patch4K) }
print(String(format: "Color Pop warmed 4K render: %.2f ms/frame on this CI runner", (CFAbsoluteTimeGetCurrent() - start) * 1000 / 5))
print("Color Pop checks passed")
