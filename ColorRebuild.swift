import CoreVideo
import Foundation

/// Row loops in the 4K rebuilds are independent per pixel. Tests set `forceSerial`
/// and compare that path with the parallel one; production keeps the parallel path.
enum RebuildRows {
    static var forceSerial = false

    static func forEach(_ count: Int, _ body: (Int) -> Void) {
        if forceSerial || count < 48 {
            for index in 0..<count { body(index) }
            return
        }
        DispatchQueue.concurrentPerform(iterations: count, execute: body)
    }
}

/// Reconstructs chroma from the clean luma instead of blurring the whole frame.
/// Flat anime cels are consolidated, speckles and bands are pulled toward the
/// edge-aware local color, and line art keeps its chroma because strong luma
/// edges reduce the mix. Skin tones are mixed less so they do not wash out.
/// Video frames blend stable pixels toward the previous rebuilt chroma.
final class ColorRebuild {
    private var previousLuma: [UInt8] = []
    private var previousCb: [UInt8] = []
    private var previousCr: [UInt8] = []
    private var previousWidth = 0
    private var previousHeight = 0
    private var hasHistory = false

    func apply(_ source: CVPixelBuffer, strength: Double) -> CVPixelBuffer {
        let amount = min(max(strength, 0), 1)
        guard amount > 0.001 else { return source }
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        guard width > 4, height > 4 else { return source }
        guard let destination = Self.makeBuffer(width: width, height: height) else { return source }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination) else { return source }
        let src = sourceBase.assumingMemoryBound(to: UInt8.self)
        let dst = destinationBase.assumingMemoryBound(to: UInt8.self)
        let srcRow = CVPixelBufferGetBytesPerRow(source)
        let dstRow = CVPixelBufferGetBytesPerRow(destination)
        let count = width * height
        if previousWidth != width || previousHeight != height || previousLuma.count != count {
            previousLuma = [UInt8](repeating: 0, count: count)
            previousCb = [UInt8](repeating: 128, count: count)
            previousCr = [UInt8](repeating: 128, count: count)
            previousWidth = width
            previousHeight = height
            hasHistory = false
        }

        let radius = amount > 0.7 ? 4 : 3
        var luma = [UInt8](repeating: 0, count: count)
        var cb = [Int16](repeating: 0, count: count)
        var cr = [Int16](repeating: 0, count: count)
        luma.withUnsafeMutableBufferPointer { lumaBuf in
        cb.withUnsafeMutableBufferPointer { cbBuf in
        cr.withUnsafeMutableBufferPointer { crBuf in
        previousLuma.withUnsafeMutableBufferPointer { previousLumaBuf in
        previousCb.withUnsafeMutableBufferPointer { previousCbBuf in
        previousCr.withUnsafeMutableBufferPointer { previousCrBuf in
            RebuildRows.forEach(height) { y in
                let row = src.advanced(by: y * srcRow)
                let base = y * width
                for x in 0..<width {
                    let p = row.advanced(by: x * 4)
                    let b = Int(p[0])
                    let g = Int(p[1])
                    let r = Int(p[2])
                    let lumaWeighted: Int = 77 * r + 150 * g + 29 * b
                    let y8 = UInt8(min(255, lumaWeighted >> 8))
                    lumaBuf[base + x] = y8
                    let cbWeighted: Int = -43 * r - 85 * g + 128 * b
                    let crWeighted: Int = 128 * r - 107 * g - 21 * b
                    cbBuf[base + x] = Int16((cbWeighted >> 8) + 128)
                    crBuf[base + x] = Int16((crWeighted >> 8) + 128)
                }
            }

            var guidedCb = [Int16](repeating: 0, count: count)
            var guidedCr = [Int16](repeating: 0, count: count)
            let lumaRead = UnsafeBufferPointer(lumaBuf)
            let cbRead = UnsafeBufferPointer(cbBuf)
            let crRead = UnsafeBufferPointer(crBuf)
            guidedCb.withUnsafeMutableBufferPointer { guidedCbBuf in
                Self.edgeAware(channel: cbRead, guide: lumaRead, width: width, height: height, radius: radius, output: guidedCbBuf)
            }
            guidedCr.withUnsafeMutableBufferPointer { guidedCrBuf in
                Self.edgeAware(channel: crRead, guide: lumaRead, width: width, height: height, radius: radius, output: guidedCrBuf)
            }
            let hasPrevious = hasHistory
            guidedCb.withUnsafeBufferPointer { guidedCbPtr in
            guidedCr.withUnsafeBufferPointer { guidedCrPtr in
                RebuildRows.forEach(height) { y in
                    let row = src.advanced(by: y * srcRow)
                    let out = dst.advanced(by: y * dstRow)
                    let base = y * width
                    for x in 0..<width {
                        let index = base + x
                        let y8 = Int(lumaBuf[index])
                        let cb0 = Int(cbBuf[index])
                        let cr0 = Int(crBuf[index])
                        let p = row.advanced(by: x * 4)
                        let o = out.advanced(by: x * 4)

                        // Preserve narrow source contours and colored micro-details before
                        // chroma reconstruction. This catches thin background ink and
                        // blush strokes whose luma contrast alone can be weak.
                        let fineDetail = Self.fineDetailProtection(
                            luma: lumaRead, cb: cbRead, cr: crRead, x: x, y: y, width: width, height: height
                        )
                        if fineDetail >= 0.82 {
                            o[0] = p[0]
                            o[1] = p[1]
                            o[2] = p[2]
                            o[3] = p[3]
                            previousLumaBuf[index] = UInt8(y8)
                            previousCbBuf[index] = UInt8(cb0)
                            previousCrBuf[index] = UInt8(cr0)
                            continue
                        }

                        // Blush hatch is too faint for the threshold above, and the
                        // temporal chroma hold below would park last frame's pink
                        // on a stroke that moved by a pixel. Keep the source bytes.
                        if BlushStrokeGuard.protects(pixels: UnsafePointer(src), x: x, y: y, width: width, height: height, rowBytes: srcRow, bgra: true) {
                            o[0] = p[0]
                            o[1] = p[1]
                            o[2] = p[2]
                            o[3] = p[3]
                            previousLumaBuf[index] = UInt8(y8)
                            previousCbBuf[index] = UInt8(cb0)
                            previousCrBuf[index] = UInt8(cr0)
                            continue
                        }

                        let edge = Self.lumaEdge(lumaRead, x: x, y: y, width: width, height: height)
                        let flatMix: Double = edge < 12 ? 0.82 : (edge < 28 ? 0.38 : 0.12)
                        var mix: Double = amount * flatMix
                        mix *= (1.0 - fineDetail)
                        if Self.isSkin(y: y8, cb: cb0, cr: cr0) { mix *= 0.42 }
                        let keep: Double = 1.0 - mix
                        let cbSource = Double(cb0)
                        let crSource = Double(cr0)
                        let cbGuide = Double(guidedCbPtr[index])
                        let crGuide = Double(guidedCrPtr[index])
                        let mixedCb: Double = cbSource * keep + cbGuide * mix
                        let mixedCr: Double = crSource * keep + crGuide * mix
                        var outCb = Int(mixedCb.rounded())
                        var outCr = Int(mixedCr.rounded())
                        if hasPrevious {
                            let motion = abs(y8 - Int(previousLumaBuf[index]))
                            if motion < 8 {
                                // Temporal chroma reuse is also suppressed on line detail.
                                let hold: Double = 0.34 * (1.0 - fineDetail)
                                let kept: Double = 1.0 - hold
                                let heldCb: Double = Double(outCb) * kept + Double(previousCbBuf[index]) * hold
                                let heldCr: Double = Double(outCr) * kept + Double(previousCrBuf[index]) * hold
                                outCb = Int(heldCb.rounded())
                                outCr = Int(heldCr.rounded())
                            }
                        }
                        outCb = min(255, max(0, outCb))
                        outCr = min(255, max(0, outCr))
                        let rgb = Self.rgb(y: y8, cb: outCb, cr: outCr)
                        o[0] = rgb.b
                        o[1] = rgb.g
                        o[2] = rgb.r
                        o[3] = p[3]
                        previousLumaBuf[index] = UInt8(y8)
                        previousCbBuf[index] = UInt8(outCb)
                        previousCrBuf[index] = UInt8(outCr)
                    }
                }
            }
            }
        }
        }
        }
        }
        }
        }
        hasHistory = true
        return destination
    }


    /// Returns high confidence only when the pixel is a narrow feature with
    /// matching surroundings on both sides. Luma catches dark ink; Cb/Cr catches
    /// colored facial marks such as blush that luma-only gating misses.
    private static func fineDetailProtection(
        luma: UnsafeBufferPointer<UInt8>, cb: UnsafeBufferPointer<Int16>, cr: UnsafeBufferPointer<Int16>,
        x: Int, y: Int, width: Int, height: Int
    ) -> Double {
        guard width > 4, height > 4 else { return 0 }
        let lx = max(0, x - 2)
        let rx = min(width - 1, x + 2)
        let ty = max(0, y - 2)
        let by = min(height - 1, y + 2)
        let center = y * width + x
        let left = y * width + lx
        let right = y * width + rx
        let top = ty * width + x
        let bottom = by * width + x

        @inline(__always) func response(_ c: Int, _ a: Int, _ b: Int, flankLimit: Double) -> Double {
            let contrast = min(abs(c - a), abs(c - b))
            let coherence = max(0.0, 1.0 - Double(abs(a - b)) / flankLimit)
            return Double(contrast) * coherence
        }

        let lumaResponse = max(
            response(Int(luma[center]), Int(luma[left]), Int(luma[right]), flankLimit: 44),
            response(Int(luma[center]), Int(luma[top]), Int(luma[bottom]), flankLimit: 44)
        )
        let chromaResponse = max(
            max(
                response(Int(cb[center]), Int(cb[left]), Int(cb[right]), flankLimit: 52),
                response(Int(cr[center]), Int(cr[left]), Int(cr[right]), flankLimit: 52)
            ),
            max(
                response(Int(cb[center]), Int(cb[top]), Int(cb[bottom]), flankLimit: 52),
                response(Int(cr[center]), Int(cr[top]), Int(cr[bottom]), flankLimit: 52)
            )
        )
        let lumaProtection = smoothstep(10, 34, lumaResponse)
        let chromaProtection = smoothstep(12, 38, chromaResponse)
        return max(lumaProtection, chromaProtection)
    }

    private static func smoothstep(_ low: Double, _ high: Double, _ value: Double) -> Double {
        let t = min(max((value - low) / max(high - low, 0.000001), 0), 1)
        return t * t * (3 - 2 * t)
    }

    private static func isSkin(y: Int, cb: Int, cr: Int) -> Bool {
        y > 40 && y < 235 && cb > 85 && cb < 135 && cr > 140 && cr < 180
    }

    private static func lumaEdge(_ luma: UnsafeBufferPointer<UInt8>, x: Int, y: Int, width: Int, height: Int) -> Int {
        let cx = min(max(x, 1), width - 2)
        let cy = min(max(y, 1), height - 2)
        let c = Int(luma[cy * width + cx])
        let dx = abs(Int(luma[cy * width + cx + 1]) - Int(luma[cy * width + cx - 1]))
        let dy = abs(Int(luma[(cy + 1) * width + cx]) - Int(luma[(cy - 1) * width + cx]))
        return max(dx, dy, abs(c - c))
    }

    private static func edgeAware(
        channel: UnsafeBufferPointer<Int16>,
        guide: UnsafeBufferPointer<UInt8>,
        width: Int,
        height: Int,
        radius: Int,
        output: UnsafeMutableBufferPointer<Int16>
    ) {
        var horizontal = [Int16](repeating: 0, count: width * height)
        horizontal.withUnsafeMutableBufferPointer { horizontalBuf in
            RebuildRows.forEach(height) { y in
                for x in 0..<width {
                    var sum = 0
                    var weight = 0
                    let center = Int(guide[y * width + x])
                    let x0 = max(0, x - radius)
                    let x1 = min(width - 1, x + radius)
                    for xx in x0...x1 {
                        let w = 64 / (1 + abs(center - Int(guide[y * width + xx])) / 3)
                        sum += Int(channel[y * width + xx]) * w
                        weight += w
                    }
                    horizontalBuf[y * width + x] = Int16(sum / max(weight, 1))
                }
            }
            RebuildRows.forEach(height) { y in
                let y0 = max(0, y - radius)
                let y1 = min(height - 1, y + radius)
                for x in 0..<width {
                    var sum = 0
                    var weight = 0
                    let center = Int(guide[y * width + x])
                    for yy in y0...y1 {
                        let w = 64 / (1 + abs(center - Int(guide[yy * width + x])) / 3)
                        sum += Int(horizontalBuf[yy * width + x]) * w
                        weight += w
                    }
                    output[y * width + x] = Int16(sum / max(weight, 1))
                }
            }
        }
    }

    private static func rgb(y: Int, cb: Int, cr: Int) -> (r: UInt8, g: UInt8, b: UInt8) {
        let c = cb - 128
        let d = cr - 128
        let redShift: Int = (359 * d) >> 8
        let greenShift: Int = (88 * c + 183 * d) >> 8
        let blueShift: Int = (454 * c) >> 8
        let r = min(255, max(0, y + redShift))
        let g = min(255, max(0, y - greenShift))
        let b = min(255, max(0, y + blueShift))
        return (UInt8(r), UInt8(g), UInt8(b))
    }

    private static func makeBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &buffer) == kCVReturnSuccess else { return nil }
        return buffer
    }
}
