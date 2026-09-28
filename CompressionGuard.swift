import Foundation
import CoreImage
import CoreVideo

final class CompressionGuard {
    enum Error: Swift.Error { case allocationFailed }

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

    // CapCut-Clarity-style anime treatment. Build 123 matched much of the tonal clarity,
    // but its fixed +8.5% saturation was too weak and its luminance window excluded too
    // much colored hair. This version adds adaptive vibrance/color separation while still
    // keeping deep near-black regions protected from chroma amplification.
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

            // Strong local/midtone luminance separation from Build 123.
            float edgeProtect = 1.0 - smoothstep(0.075, 0.190, abs(detail));
            float tonalWindow = smoothstep(0.025, 0.13, y) * (1.0 - smoothstep(0.90, 0.995, y));
            float clarity = clamp(detail * 0.58, -0.030, 0.030) * edgeProtect * tonalWindow;

            // Richer blacks remain proportional only; no hard-black/OLED classifier.
            float shadowWeight = 1.0 - smoothstep(0.055, 0.38, y);
            float shadowDeepen = y * 0.055 * shadowWeight;
            float targetY = clamp(y + clarity - shadowDeepen, 0.0, 1.0);

            // Adaptive vibrance is the missing part of the CapCut 100 look. Measure
            // existing chroma, then boost colorful anime midtones more strongly while
            // tapering already extreme colors to avoid clipping/neon artifacts.
            vec3 sourceChroma = o.rgb - vec3(y);
            float chromaMagnitude = length(sourceChroma);
            float colorPresence = smoothstep(0.025, 0.115, chromaMagnitude);
            float extremeProtect = 1.0 - 0.35 * smoothstep(0.28, 0.52, chromaMagnitude);

            // Start the color window lower than #123 so pink/red/blue hair shading is
            // included. Still fade to zero in near-black shadows so Build 115's cleaned
            // dark chroma noise is never re-amplified.
            float shadowColorProtect = smoothstep(0.075, 0.19, y);
            float highlightColorProtect = 1.0 - smoothstep(0.88, 0.985, y);
            float colorWindow = shadowColorProtect * highlightColorProtect;

            // Up to ~20% chroma gain on genuinely colorful midtones, versus 8.5% in #123.
            // Neutral whites/grays barely move because colorPresence approaches zero.
            float satGain = 0.055 + 0.145 * colorPresence;
            float sat = 1.0 + satGain * colorWindow * extremeProtect;

            vec3 rgb = clamp(vec3(targetY) + sourceChroma * sat, 0.0, 1.0);
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

    private func cleanShadowChromaOnly(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        try autoreleasepool {
            let (output, extent) = try allocateLike(input)
            let original = CIImage(cvPixelBuffer: input)
            var image = original
            if let kernel = Self.shadowChromaKernel, let blur = CIFilter(name: "CIGaussianBlur") {
                blur.setValue(original, forKey: kCIInputImageKey)
                blur.setValue(1.45, forKey: kCIInputRadiusKey)
                if let smooth = blur.outputImage?.cropped(to: extent),
                   let cleaned = kernel.apply(extent: extent,
                    roiCallback: { index, rect in index == 1 ? rect.insetBy(dx: -3, dy: -3) : rect },
                    arguments: [original, smooth]) {
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
            if let kernel = Self.animeClarityKernel, let blur = CIFilter(name: "CIGaussianBlur") {
                blur.setValue(original, forKey: kCIInputImageKey)
                blur.setValue(7.0, forKey: kCIInputRadiusKey)
                if let base = blur.outputImage?.cropped(to: extent),
                   let result = kernel.apply(extent: extent,
                    roiCallback: { index, rect in index == 1 ? rect.insetBy(dx: -14, dy: -14) : rect },
                    arguments: [original, base]) {
                    image = result.cropped(to: extent)
                }
            }
            context.render(image, to: output, bounds: extent, colorSpace: colorSpace)
            context.clearCaches()
            return output
        }
    }

    // Post-CUGAN, pre-Sharpie: Build 115 chroma cleanup first, clarity/color second.
    func cleanShadowChroma(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let chromaCleaned = try cleanShadowChromaOnly(input)
        return try applyAnimeClarity(chromaCleaned)
    }

    // Light final polish only; no spatial color processing after Sharpie.
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

    // Earlier Compression Guard remains Build 115-compatible; clarity is not doubled.
    func clean(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let shadowCleaned = try cleanShadowChromaOnly(input)
        return try cleanFinalCompression(shadowCleaned)
    }
}
