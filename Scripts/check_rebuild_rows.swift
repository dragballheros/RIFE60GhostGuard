import CoreVideo
import Foundation

func fail(_ message: String) -> Never {
    fputs("REBUILD_CHECK_FAILED: \(message)\n", stderr)
    exit(1)
}

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fail(message) }
}

func makeBuffer(width: Int, height: Int, seed: Int) -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
    ]
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &optional) == kCVReturnSuccess,
          let buffer = optional else { fail("could not allocate \(width)x\(height)") }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { fail("missing base address") }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
        let row = pixels.advanced(by: y * rowBytes)
        for x in 0..<width {
            let p = row.advanced(by: x * 4)
            let n = (x * 13 + y * 7 + seed * 29) & 255
            p[0] = UInt8(n)
            p[1] = UInt8((n * 3) & 255)
            p[2] = UInt8((x == y || x == width / 2) ? 12 : 210)
            p[3] = 255
            if y > height / 3 && y < height / 3 + 8 && x > 8 && x < 28 {
                p[0] = 140
                p[1] = 170
                p[2] = 210
            }
            if y == height / 2 && x % 5 != 0 {
                p[0] = 8
                p[1] = 8
                p[2] = 8
            }
        }
    }
    return buffer
}

func samePixels(_ left: CVPixelBuffer, _ right: CVPixelBuffer, label: String) {
    let width = CVPixelBufferGetWidth(left)
    let height = CVPixelBufferGetHeight(left)
    guard width == CVPixelBufferGetWidth(right), height == CVPixelBufferGetHeight(right) else {
        fail("\(label) size mismatch")
    }
    CVPixelBufferLockBaseAddress(left, .readOnly)
    CVPixelBufferLockBaseAddress(right, .readOnly)
    defer {
        CVPixelBufferUnlockBaseAddress(right, .readOnly)
        CVPixelBufferUnlockBaseAddress(left, .readOnly)
    }
    guard let leftBase = CVPixelBufferGetBaseAddress(left),
          let rightBase = CVPixelBufferGetBaseAddress(right) else { fail("\(label) base address") }
    let leftRow = CVPixelBufferGetBytesPerRow(left)
    let rightRow = CVPixelBufferGetBytesPerRow(right)
    let leftPixels = leftBase.assumingMemoryBound(to: UInt8.self)
    let rightPixels = rightBase.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
        let a = leftPixels.advanced(by: y * leftRow)
        let b = rightPixels.advanced(by: y * rightRow)
        for x in 0..<(width * 4) where a[x] != b[x] {
            fail("\(label) pixel mismatch at \(x / 4),\(y) serial=\(a[x]) parallel=\(b[x])")
        }
    }
}

func compare(width: Int, height: Int, colorStrength: Double, outlineStrength: Double) {
    let first = makeBuffer(width: width, height: height, seed: 1)
    let second = makeBuffer(width: width, height: height, seed: 2)
    RebuildRows.forceSerial = true
    let serialColor = ColorRebuild()
    let serialFirst = serialColor.apply(first, strength: colorStrength)
    let serialSecond = serialColor.apply(second, strength: colorStrength)
    let serialOutline = OutlineRebuild().apply(first, strength: outlineStrength)
    RebuildRows.forceSerial = false
    let parallelColor = ColorRebuild()
    let parallelFirst = parallelColor.apply(first, strength: colorStrength)
    let parallelSecond = parallelColor.apply(second, strength: colorStrength)
    let parallelOutline = OutlineRebuild().apply(first, strength: outlineStrength)
    samePixels(serialFirst, parallelFirst, label: "color frame1 \(width)x\(height) strength \(colorStrength)")
    samePixels(serialSecond, parallelSecond, label: "color frame2 \(width)x\(height) strength \(colorStrength)")
    samePixels(serialOutline, parallelOutline, label: "outline \(width)x\(height) strength \(outlineStrength)")
}

// A three-pixel L-shaped neighborhood is not a contour bridge. It used
// to satisfy the broad >=3-neighbor closing rule and could darken the empty
// center into a one-frame dot.
func makeOutlineFixture(points: [(Int, Int, UInt8)], width: Int = 64, height: Int = 48) -> CVPixelBuffer {
    let buffer = makeBuffer(width: width, height: height, seed: 4)
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height {
        let row = base.advanced(by: y * rowBytes)
        for x in 0..<width {
            let p = row.advanced(by: x * 4)
            p[0] = 210; p[1] = 210; p[2] = 210; p[3] = 255
        }
    }
    for (x, y, value) in points {
        let p = base.advanced(by: y * rowBytes + x * 4)
        p[0] = value; p[1] = value; p[2] = value; p[3] = 255
    }
    return buffer
}

func readGray(_ buffer: CVPixelBuffer, x: Int, y: Int) -> UInt8 {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    return base[y * CVPixelBufferGetBytesPerRow(buffer) + x * 4]
}

let lShape = makeOutlineFixture(points: [
    (29, 30, 20), (29, 29, 20), (30, 29, 20),
    (28, 30, 210), (32, 30, 150), (30, 28, 210), (30, 32, 150)
])
let lShapeOutput = OutlineRebuild().apply(lShape, strength: 0.5)
require(readGray(lShapeOutput, x: 30, y: 30) == 210,
        "a one-sided L-shaped neighborhood must not manufacture a dark center dot")

let bridgedLine = makeOutlineFixture(points: [
    (47, 30, 20), (48, 30, 20), (49, 30, 20),
    (51, 30, 20), (52, 30, 20), (53, 30, 20),
    (46, 30, 210), (54, 30, 150)
])
let bridgedOutput = OutlineRebuild().apply(bridgedLine, strength: 0.5)
require(readGray(bridgedOutput, x: 50, y: 30) < 200,
        "a genuine two-sided horizontal contour gap should still be repaired")

// Two nearby parallel outlines are separate contours, not a gap to bridge.
// This specifically models tight shirt/skin, finger/nail, and similar
// close-outline geometry.
let parallelLines = makeOutlineFixture(points: [
    (29, 28, 20), (29, 29, 20), (29, 30, 20), (29, 31, 20), (29, 32, 20),
    (31, 28, 20), (31, 29, 20), (31, 30, 20), (31, 31, 20), (31, 32, 20)
])
let parallelOutput = OutlineRebuild().apply(parallelLines, strength: 0.5)
require(readGray(parallelOutput, x: 30, y: 30) == 210,
        "parallel close outlines must not be bridged into a dark seam")

compare(width: 80, height: 64, colorStrength: 0.45, outlineStrength: 0.5)
compare(width: 96, height: 64, colorStrength: 0.85, outlineStrength: 0.8)
compare(width: 40, height: 24, colorStrength: 0.45, outlineStrength: 0.5)
print("REBUILD_ROW_CHECK_OK")
