import Foundation
import CoreVideo
import CoreGraphics

enum ProcessorError: Error { case conversionFailed(String) }
final class DiagnosticsLogger {
    static let shared = DiagnosticsLogger()
    func log(_ text: String) { print(text) }
}
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw ProcessorError.conversionFailed(message) }
}
let region = WatermarkRegion(x: 0.35, y: 0.3, width: 0.2, height: 0.12)
let configuration = WatermarkConfiguration(enabled: true, regions: [region], paddingPixels: 3)
try require(WatermarkConfiguration.decodeRegions(configuration.serializedRegions) == [region], "mask recovery round trip")
try require(!WatermarkConfiguration(enabled: true).isValid, "empty enabled masks rejected")
let transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 320, ty: 0)
let mapped = configuration.inEncodedOrientation(size: CGSize(width: 640, height: 320), transform: transform)
try require(mapped.isValid && abs(mapped.regions[0].x - region.y) < 0.000001 && abs(mapped.regions[0].y - (1 - region.x - region.width)) < 0.000001, "rotated preview mapping")
let width = 640, height = 320
var optional: CVPixelBuffer?
let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
try require(CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &optional) == kCVReturnSuccess, "allocate fixture")
guard let source = optional else { fatalError("missing fixture") }
CVPixelBufferLockBaseAddress(source, [])
guard let sourceBase = CVPixelBufferGetBaseAddress(source) else { fatalError("missing pixels") }
let stride = CVPixelBufferGetBytesPerRow(source)
let pixels = sourceBase.assumingMemoryBound(to: UInt8.self)
for y in 0..<height {
    for x in 0..<width {
        let offset = y * stride + x * 4
        let line = x % 100 < 3 || y % 80 < 3
        pixels[offset] = line ? 25 : 155
        pixels[offset + 1] = line ? 25 : 190
        pixels[offset + 2] = line ? 25 : 235
        pixels[offset + 3] = 255
        if x >= 224 && x < 352 && y >= 96 && y < 135 {
            pixels[offset] = 255; pixels[offset + 1] = 255; pixels[offset + 2] = 255
        }
    }
}
CVPixelBufferUnlockBaseAddress(source, [])
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
let remover = try AnimeWatermarkRemover(configuration: configuration, modelURL: modelURL)
let start = Date()
let result = try remover.apply(source)
CVPixelBufferLockBaseAddress(source, .readOnly)
CVPixelBufferLockBaseAddress(result, .readOnly)
guard let resultBase = CVPixelBufferGetBaseAddress(result) else { fatalError("missing result") }
let output = resultBase.assumingMemoryBound(to: UInt8.self)
let resultStride = CVPixelBufferGetBytesPerRow(result)
var changedInside = 0
for y in 0..<height {
    for x in 0..<width {
        let inside = x >= 221 && x < 355 && y >= 93 && y < 138
        for channel in 0..<4 {
            let original = pixels[y * stride + x * 4 + channel]
            let actual = output[y * resultStride + x * 4 + channel]
            if !inside || channel == 3 { try require(original == actual, "unmasked pixels / alpha must be byte identical at \(x),\(y)") }
            else if original != actual { changedInside += 1 }
        }
    }
}
CVPixelBufferUnlockBaseAddress(result, .readOnly)
CVPixelBufferUnlockBaseAddress(source, .readOnly)
try require(changedInside > 100, "model must restore the selected watermark")
print("ANIME_INPAINT_PASS: recovery/orientation/real inference/outside-mask identity; changedChannels=\(changedInside); seconds=\(Date().timeIntervalSince(start))")
