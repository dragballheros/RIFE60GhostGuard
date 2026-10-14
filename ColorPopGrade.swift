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
        if amount > 0, let cube = CIFilter(name: "CIColorCube") {
            cube.setValue(Self.cubeDimension, forKey: "inputCubeDimension")
            cube.setValue(Self.makeCube(strength: amount), forKey: "inputCubeData")
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

        // Keep the same explicit SDR color-space interpretation and output
        // attachment as the existing BT.709 video-writer handoff.
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

    /// Luminance-anchored chroma enhancement. Instead of pushing RGB channels
    /// independently (which can brighten skin, clip saturated primaries, or
    /// create a global color cast), expand the color vector around its luma.
    /// Skin-like hues receive almost no chroma gain and no contrast change.
    private static func gradeNode(_ r: Double, _ g: Double, _ b: Double, strength s: Double) -> (r: Double, g: Double, b: Double) {
        guard s > 0 else { return (r, g, b) }
        let original = hsv(r, g, b)
        let y = luma(r, g, b)
        guard y > 0 else { return (r, g, b) }

        let high = max(r, max(g, b))
        let low = min(r, min(g, b))
        let chroma = high - low
        // Avoid amplifying codec noise and tint in neutral/near-neutral pixels.
        let chromaWeight = smoothstep(0.020, 0.095, chroma)
        guard chromaWeight > 0 else { return (r, g, b) }

        // Soft skin likelihood: peach/orange hue plus common anime-skin RGB
        // relationships. It only suppresses the grade; it never forces skin
        // toward white, orange, or a target hue.
        let skinHue = smoothstep(2, 10, original.h) * (1 - smoothstep(43, 56, original.h))
        let skinSaturation = smoothstep(0.045, 0.19, original.s) * (1 - smoothstep(0.82, 0.94, original.s))
        let skinLuma = smoothstep(0.08, 0.20, y) * (1 - smoothstep(0.95, 0.99, y))
        let redGreen = r / max(g, 0.001)
        let greenBlue = g / max(b, 0.001)
        let skinRedGreen = smoothstep(0.99, 1.045, redGreen) * (1 - smoothstep(1.62, 1.92, redGreen))
        let skinGreenBlue = smoothstep(1.00, 1.09, greenBlue) * (1 - smoothstep(2.65, 3.10, greenBlue))
        let skinWeight = skinHue * skinSaturation * skinLuma * skinRedGreen * skinGreenBlue

        // Fade the grade out near black and near peak white. The tiny S-curve
        // improves midtone separation but is disabled for skin-like pixels.
        let shadowWeight = smoothstep(0.015, 0.075, y)
        let highlightWeight = 1 - smoothstep(0.84, 0.97, y)
        let usableWeight = chromaWeight * shadowWeight * highlightWeight
        let contrastWeight = usableWeight * (1 - skinWeight)
        let contrastAmount = 0.035 * s * contrastWeight
        let targetY = clamp(y + contrastAmount * (y - 0.5) * 4 * y * (1 - y))

        // Hue-family weighting gives anime reds/pinks, blues and teal accents
        // a slightly stronger lift than generic low-saturation vibrance.
        let h = original.h
        let redWeight = max(1 - smoothstep(8, 28, h), smoothstep(334, 354, h))
        let pinkWeight = smoothstep(285, 307, h) * (1 - smoothstep(340, 357, h))
        let tealWeight = smoothstep(150, 173, h) * (1 - smoothstep(202, 224, h))
        let blueWeight = smoothstep(198, 220, h) * (1 - smoothstep(258, 280, h))
        let greenWeight = smoothstep(65, 88, h) * (1 - smoothstep(145, 170, h))
        let orangeWeight = smoothstep(14, 26, h) * (1 - smoothstep(42, 55, h))
        let familyAccent = 0.10 * redWeight
            + 0.14 * pinkWeight
            + 0.12 * tealWeight
            + 0.12 * blueWeight
            + 0.07 * greenWeight
            + 0.05 * orangeWeight

        // Lift muted colors while retaining a useful extra boost for already
        // vivid anime colors. Skin keeps only 8% of the proposed saturation gain.
        let mutedColorLift = 0.28 * pow(max(1 - original.s, 0), 0.75)
        let proposedGain = 1 + s * (mutedColorLift + familyAccent) * usableWeight
            * (1 - 0.92 * skinWeight)

        // First apply the very small luma curve with a uniform channel scale,
        // then expand chroma around targetY. Luma remains targetY by construction.
        let toneRatio = targetY / y
        let tr = r * toneRatio
        let tg = g * toneRatio
        let tb = b * toneRatio
        let dr = tr - targetY
        let dg = tg - targetY
        let db = tb - targetY

        // Gamut-map by reducing chroma only as much as needed. This prevents
        // clipping a saturated pink/red/blue channel at 1.0 and shifting hue.
        var gamutLimit = 1.0
        for delta in [dr, dg, db] {
            if delta > 0 {
                gamutLimit = min(gamutLimit, (1 - targetY) / delta)
            } else if delta < 0 {
                gamutLimit = min(gamutLimit, targetY / -delta)
            }
        }
        let safeGain = min(proposedGain, max(0, gamutLimit))
        return (
            clamp(targetY + dr * safeGain),
            clamp(targetY + dg * safeGain),
            clamp(targetY + db * safeGain)
        )
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
        let high = max(r, max(g, b))
        let low = min(r, min(g, b))
        let delta = high - low
        guard high > 0, delta > 0 else { return (0, 0, high) }
        let sector: Double
        if high == r { sector = (g - b) / delta }
        else if high == g { sector = 2 + (b - r) / delta }
        else { sector = 4 + (r - g) / delta }
        let degrees = sector * 60
        return (degrees < 0 ? degrees + 360 : degrees, delta / high, high)
    }
}
