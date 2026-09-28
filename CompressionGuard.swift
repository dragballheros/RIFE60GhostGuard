import Foundation
import CoreImage
import CoreVideo

final class CompressionGuard {
    enum Error: Swift.Error {
        case allocationFailed
    }

    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

    // Shadow-only cleanup for colored speckle/blotchy noise in dark anime areas.
    // It uses a tiny spatial blur as a chroma reference, then preserves the original
    // luminance and strongly protects edges/black line art. Bright areas are untouched.
    private static let shadowChromaKernel: CIKernel? = {
        let source = """
        kernel vec4 shadowChromaCleanup(sampler original, sampler smooth) {
            vec2 p = samplerCoord(original);
            vec4 o = sample(original, p);
            vec4 s = sample(smooth, samplerTransform(smooth, destCoord()));

            const vec3 lumaW = vec3(0.2126, 0.7152, 0.0722);
            float yo = dot(o.rgb, lumaW);
            float ys = dot(s.rgb, lumaW);

            // Smooth chroma while restoring the exact source luminance. This attacks
            // red/green/blue shadow crawling without lifting blacks or erasing line luma.
            vec3 chromaClean = clamp(s.rgb + vec3(yo - ys), 0.0, 1.0);

            // Full strength only in deep shadows, then fade out before midtones.
            float shadow = 1.0 - smoothstep(0.10, 0.30, yo);

            // Strong protection for original anime outlines and cel boundaries.
            float localDelta = length(o.rgb - s.rgb);
            float edgeProtect = 1.0 - smoothstep(0.035, 0.105, localDelta);

            float amount = 0.78 * shadow * edgeProtect;
            vec3 rgb = mix(o.rgb, chromaClean, amount);
            return vec4(rgb, o.a);
        }
        """
        return CIKernel(source: source)
    }()

    private func allocateLike(_ input: CVPixelBuffer) throws -> (CVPixelBuffer, CGRect) {
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        let extent = CGRect(x: 0, y: 0, width: width, height: height)
        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelBufferPixelFormatTypeKey == kCVPixelBufferPixelFormatTypeKey ? kCVPixelFormatType_32BGRA : kCVPixelFormatType_32BGRA, attrs as CFDictionary, &output)
        guard status == kCVReturnSuccess, let output else { throw Error.allocationFailed }
        return (output, extent)
    }

    /// Deep-shadow chroma cleanup only. This is intentionally run BEFORE Sharpie so
    /// the final neural line work can never be softened by this spatial chroma filter.
    func cleanShadowChroma(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        try autoreleasepool {
            let (output, extent) = try allocateLike(input)
            let original = CIImage(cvPixelBuffer: input)
            var image = original

            if let kernel = Self.shadowChromaKernel,
               let blur = CIFilter(name: "CIGaussianBlur") {
                blur.setValue(original, forKey: kCIInputImageKey)
                blur.setValue(1.45, forKey: kCIInputRadiusKey)
                if let smooth = blur.outputImage?.cropped(to: extent),
                   let cleaned = kernel.apply(
                    extent: extent,
                    roiCallback: { index, rect in index == 1 ? rect.insetBy(dx: -3, dy: -3) : rect },
                    arguments: [original, smooth]
                   ) {
                    image = cleaned.cropped(to: extent)
                }
            }

            context.render(image, to: output, bounds: extent, colorSpace: colorSpace)
            context.clearCaches()
            return output
        }
    }

    /// Light final compression polish only. No shadow/chroma blur is allowed here,
    /// because this runs AFTER Sharpie and must leave the finished line style intact.
    func cleanFinalCompression(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        try autoreleasepool {
            let (output, extent) = try allocateLike(input)
            var image = CIImage(cvPixelBuffer: input)

            if let noise = CIFilter(name: "CINoiseReduction") {
                noise.setValue(image, forKey: kCIInputImageKey)
                noise.setValue(0.012, forKey: "inputNoiseLevel")
                noise.setValue(0.50, forKey: "inputSharpness")
                if let result = noise.outputImage { image = result }
            }

            if let sharpen = CIFilter(name: "CISharpenLuminance") {
                sharpen.setValue(image, forKey: kCIInputImageKey)
                sharpen.setValue(0.16, forKey: kCIInputSharpnessKey)
                sharpen.setValue(1.0, forKey: kCIInputRadiusKey)
                if let result = sharpen.outputImage { image = result }
            }

            context.render(image.cropped(to: extent), to: output, bounds: extent, colorSpace: colorSpace)
            context.clearCaches()
            return output
        }
    }

    /// Legacy/full Compression Guard path used by earlier pipeline stages.
    /// Keep behavior compatible: shadow chroma cleanup followed by normal polish.
    func clean(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let shadowCleaned = try cleanShadowChroma(input)
        return try cleanFinalCompression(shadowCleaned)
    }
}
