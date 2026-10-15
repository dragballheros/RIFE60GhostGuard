import Foundation
import CoreImage
import CoreVideo
import CoreGraphics

/// Adaptive warmth correction for Color Pop.
/// Removes both warmth introduced by grading and a limited amount of excess warmth
/// already present in the source. The source baseline is estimated only from neutral
/// and low-chroma midtones, with confidence gating to avoid cooling colorful scenes.
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

    /// Removes measured added warmth plus a conservative portion of source warmth.
    /// The source component is intentionally weaker than the added-shift component.
    func apply(reference: CVPixelBuffer, graded: CVPixelBuffer) throws -> CVPixelBuffer {
        guard CVPixelBufferGetWidth(reference) == CVPixelBufferGetWidth(graded),
              CVPixelBufferGetHeight(reference) == CVPixelBufferGetHeight(graded) else { return graded }

        let estimate = try warmthEstimate(reference: reference, graded: graded)
        // Correct nearly all clearly introduced warmth. Also gently reduce positive
        // source warmth above a small neutral tolerance, even when Color Pop added none.
        // Source warmth gets a lower gain to reduce unwanted cast without flattening
        // intentional warm lighting. A hard cap prevents aggressive white balancing.
        let addedComponent = max(estimate.addedShift - 0.0015, 0) * 0.92
        let sourceComponent = max(estimate.sourceBias - 0.012, 0) * 0.42
        let target = min(max(addedComponent + sourceComponent, 0), 0.045)
        let confidence = CGFloat(min(max(Double(estimate.samples - 8) / 40.0, 0.0), 1.0))
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

    private func warmthEstimate(reference: CVPixelBuffer, graded: CVPixelBuffer) throws -> (sourceBias: CGFloat, addedShift: CGFloat, samples: Int) {
        let ref = try sample(reference)
        let out = try sample(graded)
        var sourceBiases: [CGFloat] = []
        var addedDeltas: [CGFloat] = []
        sourceBiases.reserveCapacity(1024)
        addedDeltas.reserveCapacity(1024)
        for i in stride(from: 0, to: min(ref.count, out.count), by: 4) {
            let r = CGFloat(ref[i]) / 255
            let g = CGFloat(ref[i + 1]) / 255
            let b = CGFloat(ref[i + 2]) / 255
            let hi = max(r, max(g, b))
            let lo = min(r, min(g, b))
            let chroma = hi - lo
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            let sat = chroma / max(hi, 0.001)

            // Candidate reference pixels should be low-chroma midtones. Keep the
            // gate broad enough to find neutral surfaces but exclude saturated objects.
            guard luma > 0.10, luma < 0.91, chroma < 0.19,
                  abs(r - g) < 0.13, abs(g - b) < 0.13,
                  sat < 0.52 else { continue }

            let gr = CGFloat(out[i]) / 255
            let gb = CGFloat(out[i + 2]) / 255
            let delta = (gr - gb) - (r - b)
            let sourceBias = r - b
            if sourceBias.isFinite, abs(sourceBias) < 0.20 { sourceBiases.append(sourceBias) }
            if delta.isFinite, abs(delta) < 0.18 { addedDeltas.append(delta) }
        }
        guard sourceBiases.count >= 12 else { return (0, 0, sourceBiases.count) }
        sourceBiases.sort()
        addedDeltas.sort()
        let sourceMedian: CGFloat = {
            let m = sourceBiases.count / 2
            return sourceBiases.count.isMultiple(of: 2)
                ? (sourceBiases[m - 1] + sourceBiases[m]) * 0.5
                : sourceBiases[m]
        }()
        let addedMedian: CGFloat = {
            guard !addedDeltas.isEmpty else { return 0 }
            let m = addedDeltas.count / 2
            return addedDeltas.count.isMultiple(of: 2)
                ? (addedDeltas[m - 1] + addedDeltas[m]) * 0.5
                : addedDeltas[m]
        }()
        return (max(0, sourceMedian), max(0, addedMedian), min(sourceBiases.count, addedDeltas.count))
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
