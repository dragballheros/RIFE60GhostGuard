import Foundation
import CoreImage
import CoreVideo
import CoreGraphics

/// Conservative, source-relative white-balance correction for the Color Pop output.
/// It only counteracts a measurable warm shift introduced by the grade itself.
/// Intentional warmth already present in the input is therefore retained.
final class AutoWarmthCorrection {
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let context: CIContext
    private let kernel: CIColorKernel?
    private let lock = NSLock()
    private var smoothedAmount: CGFloat = 0
    private var pool: CVPixelBufferPool?
    private var poolWidth = 0
    private var poolHeight = 0

    init() {
        context = CIContext(options: [
            .cacheIntermediates: false,
            .workingColorSpace: colorSpace,
            .outputColorSpace: colorSpace
        ])
        kernel = CIColorKernel(source: """
        kernel vec4 autoWarmth(__sample pixel, float amount) {
            vec3 c = pixel.rgb;
            float hi = max(c.r, max(c.g, c.b));
            float lo = min(c.r, min(c.g, c.b));
            float chroma = hi - lo;
            float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
            if (y < 0.035 || y > 0.965 || chroma < 0.002) {
                return pixel;
            }
            float neutralWeight = 1.0 - smoothstep(0.075, 0.22, chroma);
            float maxc = hi;
            float minc = lo;
            float delta = maxc - minc;
            float hue = 0.0;
            if (delta > 0.0001) {
                if (maxc == c.r) hue = 60.0 * mod((c.g - c.b) / delta, 6.0);
                else if (maxc == c.g) hue = 60.0 * ((c.b - c.r) / delta + 2.0);
                else hue = 60.0 * ((c.r - c.g) / delta + 4.0);
            }
            if (hue < 0.0) hue += 360.0;
            float skinHue = smoothstep(2.0, 12.0, hue) * (1.0 - smoothstep(42.0, 58.0, hue));
            float skinSaturation = smoothstep(0.07, 0.18, chroma / max(maxc, 0.001))
                * (1.0 - smoothstep(0.72, 0.9, chroma / max(maxc, 0.001)));
            float skinWeight = skinHue * skinSaturation;
            float weight = max(neutralWeight, 0.16 * skinWeight);
            float correction = amount * weight;
            // Small red reduction plus blue increase, balanced to keep luminance
            // nearly constant. This is a temperature-only adjustment, not a grade.
            c.r -= correction * 0.52;
            c.b += correction * 0.48;
            return vec4(clamp(c, 0.0, 1.0), pixel.a);
        }
        """)
    }

    /// Detects only a warm-balance increase between the pre-grade frame and
    /// Color Pop output. Neutral midtones carry the estimate; saturated accents,
    /// skin-like hues, shadows, and clipped highlights are excluded from it.
    func apply(reference: CVPixelBuffer, graded: CVPixelBuffer) throws -> CVPixelBuffer {
        guard CVPixelBufferGetWidth(reference) == CVPixelBufferGetWidth(graded),
              CVPixelBufferGetHeight(reference) == CVPixelBufferGetHeight(graded) else {
            return graded
        }
        let referenceBias = try neutralWarmBias(reference)
        let gradedBias = try neutralWarmBias(graded)
        let addedWarmth = max(0, gradedBias - referenceBias)
        // Ignore measurement noise and cap correction to a subtle Snapseed-like
        // warmth move. No positive shift in warmth means exact zero-strength bypass.
        let target = min(max((addedWarmth - 0.004) * 0.55, 0), 0.018)

        lock.lock()
        smoothedAmount = smoothedAmount * 0.68 + target * 0.32
        let amount = smoothedAmount
        lock.unlock()

        guard amount > 0.001, let kernel else { return graded }
        let width = CVPixelBufferGetWidth(graded)
        let height = CVPixelBufferGetHeight(graded)
        try preparePool(width: width, height: height)
        guard let pool else { return graded }
        var output: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &output) == kCVReturnSuccess,
              let output else {
            throw ProcessorError.conversionFailed("could not allocate automatic warmth-correction buffer")
        }
        let input = CIImage(cvPixelBuffer: graded, options: [.colorSpace: colorSpace])
        guard let corrected = kernel.apply(extent: input.extent, arguments: [input, amount]) else {
            throw ProcessorError.conversionFailed("automatic warmth correction produced no image")
        }
        context.render(corrected, to: output, bounds: input.extent, colorSpace: colorSpace)
        CVBufferSetAttachment(output, kCVImageBufferCGColorSpaceKey, colorSpace, .shouldPropagate)
        return output
    }

    private func neutralWarmBias(_ buffer: CVPixelBuffer) throws -> CGFloat {
        let image = CIImage(cvPixelBuffer: buffer, options: [.colorSpace: colorSpace])
        let small = image.transformed(by: CGAffineTransform(
            scaleX: 64.0 / CGFloat(CVPixelBufferGetWidth(buffer)),
            y: 64.0 / CGFloat(CVPixelBufferGetHeight(buffer))
        )).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
        guard let sample = context.createCGImage(small, from: small.extent) else {
            throw ProcessorError.conversionFailed("could not analyze frame warmth")
        }
        // Normalize the sampled image to explicit RGBA byte order, independent
        // of the Core Graphics provider's native pixel layout.
        var pixels = [UInt8](repeating: 0, count: 64 * 64 * 4)
        guard let bitmap = CGContext(
            data: &pixels,
            width: 64,
            height: 64,
            bitsPerComponent: 8,
            bytesPerRow: 64 * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw ProcessorError.conversionFailed("could not allocate warmth-analysis bitmap")
        }
        bitmap.draw(sample, in: CGRect(x: 0, y: 0, width: 64, height: 64))
        var sum: CGFloat = 0
        var count: CGFloat = 0
        for y in 0..<64 {
            for x in 0..<64 {
                let offset = (y * 64 + x) * 4
                let r = CGFloat(pixels[offset]) / 255
                let g = CGFloat(pixels[offset + 1]) / 255
                let b = CGFloat(pixels[offset + 2]) / 255
                let hi = max(r, max(g, b))
                let lo = min(r, min(g, b))
                let chroma = hi - lo
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                // Prefer genuinely neutral pixels; exclude likely skin/beige and
                // saturated scene colors so sunsets and warm character art survive.
                guard luma > 0.12, luma < 0.88, chroma < 0.105,
                      abs(r - g) < 0.075, abs(g - b) < 0.075 else { continue }
                sum += r - b
                count += 1
            }
        }
        // Fewer than 12 neutral samples is not enough evidence for a safe change.
        guard count >= 12 else { return 0 }
        return sum / count
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
              let newPool else {
            throw ProcessorError.conversionFailed("could not create automatic warmth-correction buffer pool")
        }
        pool = newPool
        poolWidth = width
        poolHeight = height
    }
}
