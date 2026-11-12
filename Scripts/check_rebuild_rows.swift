import CoreVideo
import Foundation

func fail(_ message: String) -> Never {
    fputs("REBUILD_CHECK_FAILED: \(message)\n", stderr)
    exit(1)
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

compare(width: 80, height: 64, colorStrength: 0.45, outlineStrength: 0.5)
compare(width: 96, height: 64, colorStrength: 0.85, outlineStrength: 0.8)
compare(width: 40, height: 24, colorStrength: 0.45, outlineStrength: 0.5)
print("REBUILD_ROW_CHECK_OK")
