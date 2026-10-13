import Foundation
import CoreVideo

struct GhostGuardResult {
    let reject: Bool
    let reason: String
}

struct GhostGuard {
    var sensitivity: Double = 1.0
    var enableSceneCuts: Bool = true

    func inspect(previous: CVPixelBuffer, generated: CVPixelBuffer, current: CVPixelBuffer) -> GhostGuardResult {
        guard let a = SampledBuffer(buffer: previous),
              let m = SampledBuffer(buffer: generated),
              let b = SampledBuffer(buffer: current) else {
            return .init(reject: false, reason: "analysis unavailable")
        }

        let ab = a.meanAbsoluteDifference(to: b)

        // Never synthesize across a likely shot cut. Clamp sensitivity so a bad
        // UI/config value cannot make thresholds infinite or negative.
        let safeSensitivity = sensitivity.isFinite ? min(max(sensitivity, 0.35), 2.5) : 1.0
        if enableSceneCuts && ab > 0.20 / safeSensitivity {
            return .init(reject: true, reason: "scene cut / transition discontinuity")
        }

        let am = a.meanAbsoluteDifference(to: m)
        let mb = m.meanAbsoluteDifference(to: b)
        if ab > 0.025 && min(am, mb) > ab * (1.18 / safeSensitivity) {
            return .init(reject: true, reason: "temporal outlier")
        }

        // Detect catastrophic exposure/clipping introduced by interpolation.
        // Use fractions rather than mean luma so a small bright highlight does
        // not trigger; require a large excess over BOTH source endpoints.
        let sourceWhite = max(a.clippedFraction(above: 0.975), b.clippedFraction(above: 0.975))
        let sourceBlack = max(a.clippedFraction(below: 0.025), b.clippedFraction(below: 0.025))
        let generatedWhite = m.clippedFraction(above: 0.975)
        let generatedBlack = m.clippedFraction(below: 0.025)
        if generatedWhite > 0.38 && generatedWhite - sourceWhite > 0.20 {
            return .init(reject: true, reason: "interpolated-frame white clipping/exposure spike")
        }
        if generatedBlack > 0.38 && generatedBlack - sourceBlack > 0.20 {
            return .init(reject: true, reason: "interpolated-frame black clipping spike")
        }

        // Check local regions as well as global statistics. A corrupted patch
        // can be hidden by a mostly normal frame; compare per-tile luma against
        // the midpoint of source endpoints and reject only large local outliers.
        let spatial = m.maxTileDeviation(from: a, and: b, grid: 4)
        if ab > 0.018 && spatial > 0.34 {
            return .init(reject: true, reason: "localized spatial corruption")
        }

        let ea = a.edgeEnergy()
        let em = m.edgeEnergy()
        let eb = b.edgeEnergy()
        let edgeLimit = max(ea, eb) * (1.62 / safeSensitivity) + 0.008
        if ab > 0.035 && em > edgeLimit {
            return .init(reject: true, reason: "double-edge/ghost energy")
        }

        let blend = a.blendDifference(mid: m, other: b)
        if ab > 0.04 && blend > 0.23 / safeSensitivity {
            return .init(reject: true, reason: "midframe inconsistency")
        }

        return .init(reject: false, reason: "ok")
    }
}

private struct SampledBuffer {
    let w: Int
    let h: Int
    let luma: [Float]

    init?(buffer: CVPixelBuffer) {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let srcW = CVPixelBufferGetWidth(buffer)
        let srcH = CVPixelBufferGetHeight(buffer)
        let row = CVPixelBufferGetBytesPerRow(buffer)
        let step = max(1, max(srcW, srcH) / 160)
        let outW = max(1, srcW / step)
        let outH = max(1, srcH / step)
        let p = base.assumingMemoryBound(to: UInt8.self)
        var values = [Float]()
        values.reserveCapacity(outW * outH)

        for y in stride(from: 0, to: srcH, by: step) {
            for x in stride(from: 0, to: srcW, by: step) {
                let i = y * row + x * 4
                let r: Float, g: Float, b: Float
                if format == kCVPixelFormatType_32BGRA {
                    b = Float(p[i]) / 255; g = Float(p[i+1]) / 255; r = Float(p[i+2]) / 255
                } else {
                    r = Float(p[i]) / 255; g = Float(p[i+1]) / 255; b = Float(p[i+2]) / 255
                }
                values.append(0.2126*r + 0.7152*g + 0.0722*b)
            }
        }
        self.w = Int(ceil(Double(srcW) / Double(step)))
        self.h = Int(ceil(Double(srcH) / Double(step)))
        self.luma = values
    }

    func meanAbsoluteDifference(to other: SampledBuffer) -> Double {
        let n = min(luma.count, other.luma.count)
        guard n > 0 else { return 0 }
        var s: Float = 0
        for i in 0..<n { s += abs(luma[i] - other.luma[i]) }
        return Double(s / Float(n))
    }

    func clippedFraction(above threshold: Float) -> Double {
        guard !luma.isEmpty else { return 0 }
        var count = 0
        for value in luma where value >= threshold { count += 1 }
        return Double(count) / Double(luma.count)
    }

    func clippedFraction(below threshold: Float) -> Double {
        guard !luma.isEmpty else { return 0 }
        var count = 0
        for value in luma where value <= threshold { count += 1 }
        return Double(count) / Double(luma.count)
    }

    func maxTileDeviation(from a: SampledBuffer, and b: SampledBuffer, grid: Int) -> Double {
        let width = min(w, min(a.w, b.w))
        let height = min(h, min(a.h, b.h))
        guard width >= grid, height >= grid else { return 0 }
        var worst = 0.0
        for gy in 0..<grid {
            for gx in 0..<grid {
                let x0 = gx * width / grid
                let x1 = (gx + 1) * width / grid
                let y0 = gy * height / grid
                let y1 = (gy + 1) * height / grid
                var deviation: Float = 0
                var count = 0
                for y in y0..<y1 {
                    for x in x0..<x1 {
                        let i = y * w + x
                        let ai = y * a.w + x
                        let bi = y * b.w + x
                        guard i < luma.count, ai < a.luma.count, bi < b.luma.count else { continue }
                        let expected = (a.luma[ai] + b.luma[bi]) * 0.5
                        deviation += abs(luma[i] - expected)
                        count += 1
                    }
                }
                if count > 0 { worst = max(worst, Double(deviation / Float(count))) }
            }
        }
        return worst
    }

    func edgeEnergy() -> Double {
        guard w > 2, h > 2, luma.count >= w*h else { return 0 }
        var sum: Float = 0; var n: Float = 0
        for y in 1..<(h-1) {
            for x in 1..<(w-1) {
                let i = y*w+x
                let gx = luma[i+1] - luma[i-1]
                let gy = luma[i+w] - luma[i-w]
                sum += abs(gx) + abs(gy); n += 1
            }
        }
        return Double(sum / max(n, 1))
    }

    func blendDifference(mid: SampledBuffer, other: SampledBuffer) -> Double {
        let n = min(luma.count, min(mid.luma.count, other.luma.count))
        guard n > 0 else { return 0 }
        var s: Float = 0
        for i in 0..<n {
            let linear = (luma[i] + other.luma[i]) * 0.5
            s += abs(mid.luma[i] - linear)
        }
        return Double(s / Float(n))
    }
}
