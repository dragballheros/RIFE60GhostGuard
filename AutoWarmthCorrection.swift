import Foundation
import CoreImage
import CoreVideo
import CoreGraphics

/// Reference-relative warmth correction for Color Pop.
/// Measures the *change* introduced by grading, not the absolute warmth of the scene.
/// Neutral and low-chroma source pixels form the reference set, so warm sunsets,
/// wood, and skin already present in the source are not treated as errors.
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
            if (y < 0.025 || y > 0.985 || chroma < 0.002) return pixel;

            float delta = hi - lo;
            float hue = 0.0;
            if (delta > 0.0001) {
                if (hi == c.r) hue = 60.0 * mod((c.g - c.b) / delta, 6.0);
                else if (hi == c.g) hue = 60.0 * ((c.b - c.r) / delta + 2.0);
                else hue = 60.0 * ((c.r - c.g) / delta + 4.0);
            }
            if (hue < 0.0) hue += 360.0;
            float sat = chroma / max(hi, 0.001);
            float skinHue = smoothstep(3.0, 12.0, hue) * (1.0 - smoothstep(42.0, 55.0, hue));
            float skinLike = skinHue * smoothstep(0.10, 0.24, sat) * (1.0 - smoothstep(0.78, 0.92, sat));
            float neutralWeight = 1.0 - smoothstep(0.055, 0.23, chroma);
            float correctionWeight = max(neutralWeight, 0.12 * skinLike);
            float correction = amount * correctionWeight;

            // Cool the red-blue axis gently while keeping luminance approximately stable.
            c.r -= correction * 0.52;
            c.b += correction * 0.48;
            return vec4(clamp(c, 0.0, 1.0), pixel.a);
        }
        """)
    }

    /// Applies only the positive warmth shift introduced by grading.
    /// If evidence is weak or the grade did not add warmth, output is unchanged.
    func apply(reference: CVPixelBuffer, graded: CVPixelBuffer) throws -> CVPixelBuffer {
        guard CVPixelBufferGetWidth(reference) == CVPixelBufferGetWidth(graded),
              CVPixelBufferGetHeight(reference) == CVPixelBufferGetHeight(graded) else { return graded }

        let estimate = try addedWarmth(reference: reference, graded: graded)
        // Ignore quantization/noise, then correct most of a clearly measured shift.
        // The cap prevents the auto pass from becoming a global creative white-balance tool.
        let target = min(max((estimate.shift - 0.0018) * 0.92, 0), 0.040)
        let confidence = CGFloat(min(max(Double(estimate.samples - 8) / 48.0, 0.0), 1.0))
        let confidenceScaledTarget = target * confidence

        lock.lock()
        // Smooth temporal estimates for video; still-image instances receive the full
        // estimate on their first call. Smoothing never creates a correction if target is zero.
        smoothedAmount = confidenceScaledTarget <= 0
            ? smoothedAmount * 0.35
            : smoothedAmount * 0.35 + confidenceScaledTarget * 0.65
        let amount = smoothedAmount
        lock.unlock()

        guard amount > 0.0012, let kernel else { return graded }
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
        DiagnosticsLogger.shared.log("Auto warmth • measuredAddedRedBlueBias=\(String(format: "%.5f", Double(estimate.shift))) • samples=\(estimate.samples) • confidence=\(String(format: "%.2f", Double(confidence))) • correction=\(String(format: "%.4f", Double(amount)))")
        return output
    }

    private func addedWarmth(reference: CVPixelBuffer, graded: CVPixelBuffer) throws -> (shift: CGFloat, samples: Int) {
        let ref = try sample(reference)
        let out = try sample(graded)
        var deltas: [CGFloat] = []
        deltas.reserveCapacity(1024)
        for i in stride(from: 0, to: min(ref.count, out.count), by: 4) {
            let r = CGFloat(ref[i]) / 255
            let g = CGFloat(ref[i + 1]) / 255
            let b = CGFloat(ref[i + 2]) / 255
            let hi = max(r, max(g, b))
            let lo = min(r, min(g, b))
            let chroma = hi - lo
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            let sat = chroma / max(hi, 0.001)
            // Include neutral and moderately low-chroma reference pixels. Exclude
            // deep shadows, clipped whites, and saturated warm scene content.
            guard luma > 0.10, luma < 0.91, chroma < 0.19,
                  abs(r - g) < 0.13, abs(g - b) < 0.13,
                  sat < 0.52 else { continue }
            let gr = CGFloat(out[i]) / 255
            let gb = CGFloat(out[i + 2]) / 255
            let delta = (gr - gb) - (r - b)
            // A red-blue increase is the warm direction. Ignore large local changes
            // which are more likely to be colored objects or quantization outliers.
            if delta.isFinite, abs(delta) < 0.18 { deltas.append(delta) }
        }
        guard deltas.count >= 12 else { return (0, deltas.count) }
        deltas.sort()
        let lower = deltas.count / 10
        let upper = max(lower + 1, deltas.count - lower)
        let trimmed = Array(deltas[lower..<upper])
        // Median is robust to a small number of beige/wood pixels in the reference set.
        let middle = trimmed.count / 2
        let median: CGFloat = trimmed.count.isMultiple(of: 2)
            ? (trimmed[middle - 1] + trimmed[middle]) * 0.5
            : trimmed[middle]
        return (max(0, median), deltas.count)
    }

    private func sample(_ buffer: CVPixelBuffer) throws -> [UInt8] {
        let image = CIImage(cvPixelBuffer: buffer, options: [.colorSpace: colorSpace])
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let scale = min(64.0 / CGFloat(width), 64.0 / CGFloat(height))
        let small = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cgImage = context.createCGImage(small, from: small.extent) else {
            throw ProcessorError.conversionFailed("could not analyze image warmth")
        }
        let w = cgImage.width
        let h = cgImage.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let ok = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let bitmap = CGContext(
                data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            ) else { return false }
            bitmap.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw ProcessorError.conversionFailed("could not allocate warmth-analysis bitmap") }
        return pixels
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
