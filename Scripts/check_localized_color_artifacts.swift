import Foundation
import CoreVideo

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fatalError("COLOR_ARTIFACT_CHECK_FAILED: \(message)") }
}

func makeFrame(width: Int = 960, height: Int = 540, patch: CGRect? = nil) throws -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess,
          let buffer = optional else {
        throw NSError(domain: "ColorArtifactFixture", code: 1)
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw NSError(domain: "ColorArtifactFixture", code: 2) }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
        let row = pixels.advanced(by: y * rowBytes)
        for x in 0..<width {
            let i = x * 4
            var blue: UInt8 = 120
            var green: UInt8 = 120
            var red: UInt8 = 120
            if let patch, patch.contains(CGPoint(x: x, y: y)) {
                // Saturated blue/cyan corruption with almost unchanged luma.
                blue = 190
                green = 110
                red = 120
            }
            row[i] = blue
            row[i + 1] = green
            row[i + 2] = red
            row[i + 3] = 255
        }
    }
    return buffer
}

func pixel(_ buffer: CVPixelBuffer, x: Int, y: Int) -> (UInt8, UInt8, UInt8) {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(buffer))
    let i = x * 4
    return (row[i], row[i + 1], row[i + 2])
}

let guarder = LocalizedColorArtifactGuard()
let previous = try makeFrame()
let next = try makeFrame()
let damaged = try makeFrame(patch: CGRect(x: 240, y: 240, width: 80, height: 60))
let inspection = guarder.inspect(previous: previous, candidate: damaged, next: next)
require(inspection.isAnomaly, "a localized blue color patch should be detected")
let repaired = guarder.repair(previous: previous, candidate: damaged, next: next, inspection: inspection)
require(repaired != nil, "localized patch should be repairable")
let centerBefore = pixel(damaged, x: 275, y: 270)
let centerAfter = pixel(repaired!, x: 275, y: 270)
require(Int(centerAfter.0) < Int(centerBefore.0) - 20, "repair should suppress the blue chroma patch")
let untouchedBefore = pixel(damaged, x: 100, y: 100)
let untouchedAfter = pixel(repaired!, x: 100, y: 100)
require(untouchedBefore.0 == untouchedAfter.0 && untouchedBefore.1 == untouchedAfter.1 && untouchedBefore.2 == untouchedAfter.2,
        "pixels outside affected regions must remain unchanged")

let cleanInspection = guarder.inspect(previous: previous, candidate: previous, next: next)
require(!cleanInspection.isAnomaly, "clean static frames must not trigger repair")

let movingPrevious = try makeFrame(patch: CGRect(x: 240, y: 240, width: 80, height: 60))
let movingCandidate = try makeFrame(patch: CGRect(x: 280, y: 240, width: 80, height: 60))
let movingNext = try makeFrame(patch: CGRect(x: 320, y: 240, width: 80, height: 60))
let movingInspection = guarder.inspect(previous: movingPrevious, candidate: movingCandidate, next: movingNext)
require(!movingInspection.isAnomaly, "a moving patch with temporal endpoint changes must not be classed as a one-frame color flash")

print("LOCALIZED_COLOR_ARTIFACT_PASS: local chroma detection, luma-preserving repair, untouched-region preservation, static-frame stability, moving-object guard")
