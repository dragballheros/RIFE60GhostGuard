import Foundation
import CoreVideo

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fatalError("FAST_MOTION_REGION_CHECK_FAILED: \(message)") }
}

func makeFrame(kind: String, corruptPatch: Bool = false,
               width: Int = 640, height: Int = 360) throws -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess,
          let buffer = optional else { throw NSError(domain: "FastMotionRegionFixture", code: 1) }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw NSError(domain: "FastMotionRegionFixture", code: 2) }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let pixels = base.assumingMemoryBound(to: UInt8.self)

    for y in 0..<height {
        let row = pixels.advanced(by: y * rowBytes)
        for x in 0..<width {
            let checker = ((x / 8 + y / 8) % 2 == 0)
            let value: UInt8
            switch kind {
            case "previous":
                value = checker ? 0 : 100
            case "current":
                value = checker ? 180 : 80
            default:
                value = 90
            }
            let patch = corruptPatch && kind == "generated" && x < 80 && y < 60
            let v: UInt8 = patch ? 255 : value
            let i = x * 4
            row[i] = v
            row[i + 1] = v
            row[i + 2] = v
            row[i + 3] = 255
        }
    }
    return buffer
}

let previous = try makeFrame(kind: "previous")
let current = try makeFrame(kind: "current")
let cleanMidpoint = try makeFrame(kind: "generated")
let partialArtifact = try makeFrame(kind: "generated", corruptPatch: true)
let guarder = FastMotionGhostGuard(sensitivity: 1.35)

let cleanResult = guarder.inspect(previous: previous, generated: cleanMidpoint, current: current)
require(cleanResult.fastMotion, "fixture must exercise the high-motion path")
require(!cleanResult.reject, "ordinary temporal midpoint should not be hard-rejected")

let damagedResult = guarder.inspect(previous: previous, generated: partialArtifact, current: current)
require(damagedResult.fastMotion, "partial artifact should still be classified within high motion")
require(damagedResult.reject, "localized severe patch must be rejected despite frame-wide dilution")
require(damagedResult.artifactScore > cleanResult.artifactScore, "regional artifact should raise the artifact score")

print("FAST_MOTION_REGION_PASS: a concentrated damaged patch is detected without hard-rejecting the clean midpoint")
