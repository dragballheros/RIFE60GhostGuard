import Foundation
import CoreImage
import CoreVideo

final class CompressionGuard {
    enum Error: Swift.Error {
        case allocationFailed
    }

    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

    // Pre-Sharpie dark-region restoration. It removes chroma/luma compression noise,
    // then gives genuinely flat near-black regions an OLED-black floor. The black-floor
    // mask deliberately requires low luma + neutral chroma + local/broad flatness, so
    // normal cast shadows and shaded object detail are preserved instead of crushed.
    private static let shadowNoiseKernel: CIKernel? = {
        let source = """
        kernel vec4 shadowNoiseCleanup(sampler original, sampler smoothWide, sampler smoothLocal) {
            vec2 p = samplerCoord(original);
            vec4 o = sample(original, p);
            vec4 w = sample(smoothWide, samplerTransform(smoothWide, destCoord()));
            vec4 l = sample(smoothLocal, samplerTransform(smoothLocal, destCoord()));

            const vec3 lumaW = vec3(0.2126, 0.7152, 0.0722);
            float yo = dot(o.rgb, lumaW);
            float yw = dot(w.rgb, lumaW);
            float yl = dot(l.rgb, lumaW);

            // General dark cleanup fades away before normal midtones.
            float shadow = 1.0 - smoothstep(0.09, 0.31, yo);

            // A tiny reference identifies true line/cel boundaries and fine object detail.
            float fineDelta = length(o.rgb - l.rgb);
            float edgeProtect = 1.0 - smoothstep(0.045, 0.125, fineDelta);

            // First remove broad dark chroma/luma blotching. Luma follows the wide
            // reference moderately so intended lighting gradients are not flattened.
            float targetY = mix(yo, yw, 0.58);
            vec3 target = clamp(w.rgb + vec3(targetY - yw), 0.0, 1.0);
            float cleanupAmount = 0.88 * shadow * edgeProtect;
            vec3 cleaned = mix(o.rgb, target, cleanupAmount);

            // OLED BLACK FLOOR -------------------------------------------------------
            // Only consider genuinely near-black pixels. This is intentionally much
            // narrower than the general shadow-cleanup range.
            float nearBlack = 1.0 - smoothstep(0.055, 0.135, yo);

            // Background blacks are usually neutral. Colored dark material/shadows get
            // progressively less black-floor influence.
            float maxC = max(o.r, max(o.g, o.b));
            float minC = min(o.r, min(o.g, o.b));
            float chromaSpread = maxC - minC;
            float neutral = 1.0 - smoothstep(0.018, 0.075, chromaSpread);

            // Require the area to be spatially flat at BOTH fine and broad scales.
            // Real shadows cast over an object normally retain a gradient, texture,
            // highlight, cel boundary, or local variation and therefore fail this mask.
            float localVariation = abs(yo - yl);
            float broadVariation = abs(yo - yw);
            float localFlat = 1.0 - smoothstep(0.004, 0.020, localVariation);
            float broadFlat = 1.0 - smoothstep(0.010, 0.045, broadVariation);
            float detailSafe = localFlat * broadFlat * edgeProtect;

            float oledMask = nearBlack * neutral * detailSafe;

            // Soft toe first, then true zero only for the darkest/flattest portion.
            // This avoids a hard threshold/banding ring around legitimate shadows.
            float cleanedY = dot(cleaned, lumaW);
            float toe = smoothstep(0.0, 0.115, cleanedY);
            float oledY = cleanedY * toe * toe;
            float hardBlack = 1.0 - smoothstep(0.030, 0.060, cleanedY);
            oledY = mix(oledY, 0.0, hardBlack);

            vec3 oledRGB;
            if (cleanedY > 0.0001) {
                oledRGB = clamp(cleaned * (oledY / cleanedY), 0.0, 1.0);
            } else {
                oledRGB = vec3(0.0);
            }

            // Strong in confidently flat background black, zero on detected detail.
            vec3 rgb = mix(cleaned, oledRGB, 0.94 * oledMask);
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

    /// Deep-shadow cleanup + selective OLED-black floor. This always runs BEFORE
    /// Sharpie so no spatial denoising can soften the final generated line style.
    func cleanShadowChroma(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        try autoreleasepool {
            let (output, extent) = try allocateLike(input)
            let original = CIImage(cvPixelBuffer: input)
            var image = original

            if let kernel = Self.shadowNoiseKernel,
               let wideBlur = CIFilter(name: "CIGaussianBlur"),
               let localBlur = CIFilter(name: "CIGaussianBlur") {
                wideBlur.setValue(original, forKey: kCIInputImageKey)
                wideBlur.setValue(8.0, forKey: kCIInputRadiusKey)
                localBlur.setValue(original, forKey: kCIInputImageKey)
                localBlur.setValue(1.15, forKey: kCIInputRadiusKey)

                if let wide = wideBlur.outputImage?.cropped(to: extent),
                   let local = localBlur.outputImage?.cropped(to: extent),
                   let cleaned = kernel.apply(
                    extent: extent,
                    roiCallback: { index, rect in
                        if index == 1 { return rect.insetBy(dx: -16, dy: -16) }
                        if index == 2 { return rect.insetBy(dx: -3, dy: -3) }
                        return rect
                    },
                    arguments: [original, wide, local]
                   ) {
                    image = cleaned.cropped(to: extent)
                }
            }

            context.render(image, to: output, bounds: extent, colorSpace: colorSpace)
            context.clearCaches()
            return output
        }
    }

    /// Light final compression polish only. No wide shadow blur or black-floor operation
    /// is allowed here because this runs AFTER Sharpie and must leave its lines intact.
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
    func clean(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let shadowCleaned = try cleanShadowChroma(input)
        return try cleanFinalCompression(shadowCleaned)
    }
}
