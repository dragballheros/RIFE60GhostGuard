import Foundation
import CoreImage
import CoreVideo

final class CompressionGuard {
    enum Error: Swift.Error {
        case allocationFailed
    }

    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

    // Exact Build 115 chroma cleanup. Keep this behavior unchanged.
    private static let shadowChromaKernel: CIKernel? = {
        let source = """
        kernel vec4 shadowChromaCleanup(sampler original, sampler smooth) {
            vec2 p = samplerCoord(original);
            vec4 o = sample(original, p);
            vec4 s = sample(smooth, samplerTransform(smooth, destCoord()));

            const vec3 lumaW = vec3(0.2126, 0.7152, 0.0722);
            float yo = dot(o.rgb, lumaW);
            float ys = dot(s.rgb, lumaW);
            vec3 chromaClean = clamp(s.rgb + vec3(yo - ys), 0.0, 1.0);
            float shadow = 1.0 - smoothstep(0.10, 0.30, yo);
            float localDelta = length(o.rgb - s.rgb);
            float edgeProtect = 1.0 - smoothstep(0.035, 0.105, localDelta);
            float amount = 0.78 * shadow * edgeProtect;
            vec3 rgb = mix(o.rgb, chromaClean, amount);
            return vec4(rgb, o.a);
        }
        """
        return CIKernel(source: source)
    }()

    // Strong anime clarity tuned toward the supplied CapCut Clarity 100 reference.
    // The contrast boost remains luminance-only: chroma is never sharpened, which avoids
    // recreating the colored dark noise that motivated replacing CapCut in the first place.
    // There is also no OLED threshold or forced-black classifier.
    private static let animeClarityKernel: CIKernel? = {
        let source = """
        kernel vec4 animeClarity(sampler original, sampler localBase) {
            vec2 p = samplerCoord(original);
            vec4 o = sample(original, p);
            vec4 b = sample(localBase, samplerTransform(localBase, destCoord()));
            const vec3 lumaW = vec3(0.2126, 0.7152, 0.0722);

            float y = dot(o.rgb, lumaW);
            float yb = dot(b.rgb, lumaW);
            float detail = y - yb;

            // Stronger mid-scale tonal separation, while tapering on very strong edges
            // because Sharpie is responsible for the final line definition.
            float edgeProtect = 1.0 - smoothstep(0.075, 0.190, abs(detail));
            float tonalWindow = smoothstep(0.025, 0.13, y) * (1.0 - smoothstep(0.90, 0.995, y));
            float clarity = clamp(detail * 0.58, -0.030, 0.030) * edgeProtect * tonalWindow;

            // Richer blacks via a continuous proportional curve only. This cannot turn a
            // normal dark anime region into hard black: the maximum deepening is 5.5% of
            // its own luminance and fades away through the midtones.
            float shadowWeight = 1.0 - smoothstep(0.055, 0.38, y);
            float shadowDeepen = y * 0.055 * shadowWeight;
            float targetY = clamp(y + clarity - shadowDeepen, 0.0, 1.0);

            // Small vibrance boost for the more defined CapCut-like color separation.
            // Deep shadows remain excluded so Build 115's chroma cleanup stays effective.
            float colorWindow = smoothstep(0.14, 0.34, y) * (1.0 - smoothstep(0.84, 0.98, y));
            float sat = 1.0 + 0.085 * colorWindow;
            vec3 chroma = o.rgb - vec3(y);
            vec3 rgb = clamp(vec3(targetY) + chroma * sat, 0.0, 1.0);
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
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &output)
        guard status == kCVReturnSuccess, let output else { throw Error.allocationFailed }
        return (output, extent)
    }

    /// Exact Build 115 shadow chroma operation, isolated so pass 1 can keep its old
    /// behavior while the post-CUGAN pre-Sharpie path can add Anime Clarity afterward.
    private func cleanShadowChromaOnly(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
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

    private func applyAnimeClarity(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        try autoreleasepool {
            let (output, extent) = try allocateLike(input)
            let original = CIImage(cvPixelBuffer: input)
            var image = original
            if let kernel = Self.animeClarityKernel,
               let blur = CIFilter(name: "CIGaussianBlur") {
                blur.setValue(original, forKey: kCIInputImageKey)
                // Slightly broader base than #122 so the effect reads as clarity/local
                // contrast rather than conventional edge sharpening.
                blur.setValue(7.0, forKey: kCIInputRadiusKey)
                if let base = blur.outputImage?.cropped(to: extent),
                   let result = kernel.apply(
                    extent: extent,
                    roiCallback: { index, rect in index == 1 ? rect.insetBy(dx: -14, dy: -14) : rect },
                    arguments: [original, base]
                   ) {
                    image = result.cropped(to: extent)
                }
            }
            context.render(image, to: output, bounds: extent, colorSpace: colorSpace)
            context.clearCaches()
            return output
        }
    }

    /// Post-CUGAN pre-Sharpie path: preserve Build 115 chroma cleanup exactly, then
    /// add Anime Clarity. Sharpie sees the finished color/contrast master.
    func cleanShadowChroma(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let chromaCleaned = try cleanShadowChromaOnly(input)
        return try applyAnimeClarity(chromaCleaned)
    }

    /// Light final compression polish only. No spatial clarity/chroma work after Sharpie.
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

    /// Earlier Compression Guard pass remains Build 115-compatible and does NOT apply
    /// Anime Clarity here. This prevents the new look from being applied twice.
    func clean(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let shadowCleaned = try cleanShadowChromaOnly(input)
        return try cleanFinalCompression(shadowCleaned)
    }
}
