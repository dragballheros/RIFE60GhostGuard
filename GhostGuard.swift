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
        if enableSceneCuts && ab > 0.30 / sensitivity {
            return .init(reject: true, reason: "scene cut")
        }

        let am = a.meanAbsoluteDifference(to: m)
        let mb = m.meanAbsoluteDifference(to: b)
        if ab > 0.025 && min(am, mb) > ab * (1.18 / sensitivity) {
            return .init(reject: true, reason: "temporal outlier")
        }

        let ea = a.edgeEnergy()
        let em = m.edgeEnergy()
        let eb = b.edgeEnergy()
        let edgeLimit = max(ea, eb) * (1.62 / sensitivity) + 0.008
        if ab > 0.035 && em > edgeLimit {
            return .init(reject: true, reason: "double-edge/ghost energy")
        }

        let blend = a.blendDifference(mid: m, other: b)
        if ab > 0.04 && blend > 0.23 / sensitivity {
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
