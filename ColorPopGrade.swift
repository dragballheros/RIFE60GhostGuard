import Foundation
import CoreImage
import CoreVideo

/// SDR selective grade. One instance belongs to one serial pipeline stage/job:
/// its immutable strength builds one LUT, never one LUT per frame. To change
/// strength, create a new instance. No custom shader or additional encode pass.
final class ColorPopGrade {
    private static let cubeDimension = 33
    private let strength: Double
    private let colorSpace: CGColorSpace
    private let context: CIContext
    private let filter: CIFilter?
    private var pool: CVPixelBufferPool?
    private var poolWidth = 0
    private var poolHeight = 0

    init(strength: Double) {
        let amount = strength.isFinite ? min(max(strength, 0), 1) : 0
        self.strength = amount
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        self.colorSpace = space
        self.context = CIContext(options: [
            .cacheIntermediates: false,
            .workingColorSpace: space,
            .outputColorSpace: space
        ])
        if amount > 0, let cube = CIFilter(name: "CIColorCubeWithColorSpace") {
            cube.setValue(Self.cubeDimension, forKey: "inputCubeDimension")
            cube.setValue(Self.makeCube(strength: amount), forKey: "inputCubeData")
            cube.setValue(space, forKey: "inputColorSpace")
            self.filter = cube
        } else {
            self.filter = nil
        }
    }

    func apply(_ buffer: CVPixelBuffer) throws -> CVPixelBuffer {
        // Exact bypass: no conversion, allocation, filter or render at zero.
        guard strength > 0 else { return buffer }
        guard let filter else { throw ProcessorError.conversionFailed("Color Pop LUT filter unavailable") }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        try preparePool(width: width, height: height)
        guard let pool else { throw ProcessorError.conversionFailed("Color Pop buffer pool unavailable") }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &destination) == kCVReturnSuccess,
              let destination else { throw ProcessorError.conversionFailed("could not allocate Color Pop BGRA frame") }

        // These pipeline frames are SDR BGRA. Explicit sRGB interpretation
        // avoids an implicit device/display profile introducing a gray tint.
        filter.setValue(CIImage(cvPixelBuffer: buffer, options: [.colorSpace: colorSpace]), forKey: kCIInputImageKey)
        // Release the previous input graph promptly; retain the cube for the job.
        defer { filter.setValue(nil, forKey: kCIInputImageKey) }
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        guard let image = filter.outputImage else { throw ProcessorError.conversionFailed("Color Pop LUT produced no image") }
        context.render(image, to: destination, bounds: bounds, colorSpace: colorSpace)
        CVBufferSetAttachment(destination, kCVImageBufferCGColorSpaceKey, colorSpace, .shouldPropagate)
        return destination
    }

    private func preparePool(width: Int, height: Int) throws {
        if pool != nil, poolWidth == width, poolHeight == height { return }
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var newPool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &newPool) == kCVReturnSuccess,
              let newPool else { throw ProcessorError.conversionFailed("could not create Color Pop buffer pool") }
        pool = newPool
        poolWidth = width
        poolHeight = height
    }

    private static func makeCube(strength: Double) -> Data {
        var rgba = [Float]()
        rgba.reserveCapacity(cubeDimension * cubeDimension * cubeDimension * 4)
        let divisor = Double(cubeDimension - 1)
        // CI cube ordering: red changes fastest, then green, then blue.
        for blue in 0..<cubeDimension {
            for green in 0..<cubeDimension {
                for red in 0..<cubeDimension {
                    let rgb = gradeNode(Double(red) / divisor, Double(green) / divisor, Double(blue) / divisor, strength: strength)
                    rgba.append(Float(rgb.r))
                    rgba.append(Float(rgb.g))
                    rgba.append(Float(rgb.b))
                    rgba.append(1) // Opaque, so RGB is already premultiplied.
                }
            }
        }
        return rgba.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private static func gradeNode(_ r: Double, _ g: Double, _ b: Double, strength s: Double) -> (r: Double, g: Double, b: Double) {
        let original = hsv(r, g, b)
        let y = luma(r, g, b)
        let chroma = max(r, g, b) - min(r, g, b)

        // Keep neutral/very-low-chroma pixels stable. The previous version
        // also used this gate for skin, which accidentally disabled most
        // saturated anime skin and allowed the global warm grade to make it
        // increasingly yellow.
        let neutralWeight = 1 - smoothstep(0.0, 0.055, chroma)
        let protection = (1 - neutralWeight) * smoothstep(0.015, 0.10, y)
            * (1 - smoothstep(0.92, 1.0, y))
        guard s > 0, y > 0 else { return (r, g, b) }

        // Endpoint-preserving luma S-curve. Mid-gray derivative is 1 + .08*s.
        let newY = clamp(y + 0.08 * s * (y - 0.5) * 4 * y * (1 - y))
        let ratio = y > 0 ? newY / y : 1
        var rr = clamp(r * ratio)
        var gg = clamp(g * ratio)
        var bb = clamp(b * ratio)
        var color = hsv(rr, gg, bb)

        // Anime skin is commonly a fairly saturated orange/peach. Detect it
        // directly and correct its white balance independently of the global
        // grade. The narrower hue window avoids most green foliage and strong
        // yellow highlights.
        let skinHue = smoothstep(10, 16, original.h)
            * (1 - smoothstep(34, 40, original.h))
        let skinSaturation = smoothstep(0.18, 0.34, original.s)
            * (1 - smoothstep(0.78, 0.90, original.s))
        let skinLuma = smoothstep(0.22, 0.42, y)
            * (1 - smoothstep(0.94, 1.0, y))
        let skinRedGreen = smoothstep(1.04, 1.10, r / max(g, 0.001))
            * (1 - smoothstep(1.60, 1.90, r / max(g, 0.001)))
        let skinGreenBlue = smoothstep(1.15, 1.30, g / max(b, 0.001))
            * (1 - smoothstep(2.50, 3.10, g / max(b, 0.001)))
        let skinWeight = skinHue * skinSaturation * skinLuma
            * skinRedGreen * skinGreenBlue

        // Keep the global pop from over-warming likely skin. Saturation boost
        // remains available for scenery and clothing.
        let vibranceGain = 1 + 0.35 * s * pow(1 - original.s, 1.5) * (1 - skinWeight)
        color.s = clamp(color.s * vibranceGain)
        (rr, gg, bb) = rgb(color.h, color.s, color.v)

        // Global white-balance nudge: slightly cool the whole grade without
        // changing luma. Skin gets a stronger targeted correction below.
        let beforeCool = luma(rr, gg, bb)
        rr *= 1 - 0.003 * s
        bb *= 1 + 0.006 * s
        let afterCool = luma(rr, gg, bb)
        if afterCool > 0 {
            let normalize = beforeCool / afterCool
            rr = clamp(rr * normalize)
            gg = clamp(gg * normalize)
            bb = clamp(bb * normalize)
        }

        color = hsv(rr, gg, bb)

        // Pull yellow-orange skin toward a neutral peach/cream, reduce yellow
        // chroma, and add a small blue contribution. Preserve luminance so
        // this changes temperature/hue rather than simply increasing exposure.
        if skinWeight > 0 {
            let correction = clamp(s * skinWeight)
            let targetHue = 21.0
            let hueDelta = shortestHueDelta(from: color.h, to: targetHue)
            color.h = normalizeHue(color.h + hueDelta * 0.45 * correction)
            color.s = clamp(color.s * (1 - 0.09 * correction))
            // Whiten by correcting temperature/hue, not by lifting exposure.
            // The small value change keeps skin from looking washed out.
            color.v = clamp(color.v * (1 + 0.006 * correction))
            (rr, gg, bb) = rgb(color.h, color.s, color.v)

            let skinLumaBefore = luma(rr, gg, bb)
            rr *= 1 - 0.012 * correction
            gg *= 1 - 0.003 * correction
            bb *= 1 + 0.025 * correction
            let skinLumaAfter = luma(rr, gg, bb)
            if skinLumaAfter > 0 {
                let normalize = skinLumaBefore / skinLumaAfter
                rr = clamp(rr * normalize)
                gg = clamp(gg * normalize)
                bb = clamp(bb * normalize)
            }
        }

        (rr, gg, bb) = rgb(color.h, color.s, color.v)

        // Fade the global grade, but let the skin-specific temperature/hue
        // correction remain active where skin is confidently detected.
        let globalAmount = protection * s
        let skinAmount = clamp(skinWeight * s)
        let amount = max(globalAmount, skinAmount)
        return (clamp(r + (rr - r) * amount),
                clamp(g + (gg - g) * amount),
                clamp(b + (bb - b) * amount))
    }

    private static func shortestHueDelta(from: Double, to: Double) -> Double {
        var delta = (to - from).truncatingRemainder(dividingBy: 360)
        if delta > 180 { delta -= 360 }
        if delta < -180 { delta += 360 }
        return delta
    }

    private static func normalizeHue(_ hue: Double) -> Double {
        (hue.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360)
    }

    private static func luma(_ r: Double, _ g: Double, _ b: Double) -> Double {
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    private static func clamp(_ x: Double) -> Double { min(max(x, 0), 1) }

    private static func smoothstep(_ low: Double, _ high: Double, _ x: Double) -> Double {
        let t = clamp((x - low) / (high - low))
        return t * t * (3 - 2 * t)
    }

    private static func hsv(_ r: Double, _ g: Double, _ b: Double) -> (h: Double, s: Double, v: Double) {
        let high = max(r, g, b)
        let low = min(r, g, b)
        let delta = high - low
        guard high > 0, delta > 0 else { return (0, 0, high) }
        let sector: Double
        if high == r { sector = (g - b) / delta }
        else if high == g { sector = 2 + (b - r) / delta }
        else { sector = 4 + (r - g) / delta }
        let degrees = sector * 60
        return (degrees < 0 ? degrees + 360 : degrees, delta / high, high)
    }

    private static func rgb(_ hue: Double, _ saturation: Double, _ value: Double) -> (Double, Double, Double) {
        let h = (hue.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360) / 60
        let c = value * saturation
        let x = c * (1 - abs(h.truncatingRemainder(dividingBy: 2) - 1))
        let m = value - c
        let channels: (Double, Double, Double)
        switch h {
        case 0..<1: channels = (c, x, 0)
        case 1..<2: channels = (x, c, 0)
        case 2..<3: channels = (0, c, x)
        case 3..<4: channels = (0, x, c)
        case 4..<5: channels = (x, 0, c)
        default: channels = (c, 0, x)
        }
        return (channels.0 + m, channels.1 + m, channels.2 + m)
    }
}
