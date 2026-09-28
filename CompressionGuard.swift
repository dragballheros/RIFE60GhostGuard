import Foundation
import CoreImage
import CoreVideo

final class CompressionGuard {
    enum Error: Swift.Error {
        case allocationFailed
    }

    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

    // Safe pre-Sharpie cleanup for dark anime material.
    //
    // Important: this deliberately does NOT try to decide which dark pixels are
    // "background" and force them to OLED black. Build 117 proved that a flat black
    // anime object (hair/clothing/etc.) can satisfy the same pixel-level tests as a
    // black background. Instead we remove chroma noise strongly and smooth low-frequency
    // dark-luma blotching with a HARD per-pixel luminance-change limit. This makes it
    // mathematically impossible for this pass to turn normal dark artwork into black.
    private static let shadowCleanupKernel: CIKernel? = {
        let source = """
        kernel vec4 shadowCleanup(sampler original, sampler smoothLocal, sampler smoothWide) {
            vec2 p = samplerCoord(original);
            vec4 o = sample(original, p);
            vec4 l = sample(smoothLocal, samplerTransform(smoothLocal, destCoord()));
            vec4 w = sample(smoothWide, samplerTransform(smoothWide, destCoord()));

            const vec3 lumaW = vec3(0.2126, 0.7152, 0.0722);
            float yo = dot(o.rgb, lumaW);
            float yl = dot(l.rgb, lumaW);
            float yw = dot(w.rgb, lumaW);

            // Only dark material is eligible. Fade out well before normal cel colors.
            float shadow = 1.0 - smoothstep(0.10, 0.30, yo);

            // Protect line art, cel boundaries, hair highlights and textured detail.
            // A pixel that differs substantially from its tiny local reference is treated
            // as intentional structure rather than compression noise.
            float fineDelta = length(o.rgb - l.rgb);
            float detailProtect = 1.0 - smoothstep(0.030, 0.090, fineDelta);

            // CHROMA: use the small local reference but restore source luminance first.
            // This removes colored crawling/speckle without changing brightness.
            vec3 chromaClean = clamp(l.rgb + vec3(yo - yl), 0.0, 1.0);
            float chromaAmount = 0.80 * shadow * detailProtect;
            vec3 cleaned = mix(o.rgb, chromaClean, chromaAmount);

            // LUMA: attack the broad dark patches/banding visible in near-black anime
            // backgrounds, but NEVER allow the pass to make a large brightness change.
            // 0.012 in normalized RGB is ~3/255. Even if a black-haired region is
            // misclassified as flat background, one pass can therefore move it by at
            // most about three code values -- unlike Build 117, it cannot collapse it.
            float wideDelta = yw - yo;
            float boundedDelta = clamp(wideDelta, -0.012, 0.012);

            // Only trust the wide reference when the tiny neighborhood is already flat.
            // This further protects shadows cast across visible object structure.
            float localVariation = abs(yo - yl);
            float flatness = 1.0 - smoothstep(0.004, 0.025, localVariation);
            float lumaAmount = 0.72 * shadow * detailProtect * flatness;
            float targetY = yo + boundedDelta * lumaAmount;

            // Apply only the bounded luminance correction to the chroma-cleaned color.
            float yc = dot(cleaned, lumaW);
            vec3 rgb = clamp(cleaned + vec3(targetY - yc), 0.0, 1.0);
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

    /// Dark chroma + bounded luma cleanup. Always BEFORE Sharpie so the final generated
    /// line style cannot be softened by either spatial reference.
    func cleanShadowChroma(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        try autoreleasepool {
            let (output, extent) = try allocateLike(input)
            let original = CIImage(cvPixelBuffer: input)
            var image = original

            if let kernel = Self.shadowCleanupKernel,
               let localBlur = CIFilter(name: "CIGaussianBlur"),
               let wideBlur = CIFilter(name: "CIGaussianBlur") {
                localBlur.setValue(original, forKey: kCIInputImageKey)
                localBlur.setValue(1.45, forKey: kCIInputRadiusKey)
                wideBlur.setValue(original, forKey: kCIInputImageKey)
                wideBlur.setValue(8.0, forKey: kCIInputRadiusKey)

                if let local = localBlur.outputImage?.cropped(to: extent),
                   let wide = wideBlur.outputImage?.cropped(to: extent),
                   let cleaned = kernel.apply(
                    extent: extent,
                    roiCallback: { index, rect in
                        if index == 1 { return rect.insetBy(dx: -3, dy: -3) }
                        if index == 2 { return rect.insetBy(dx: -16, dy: -16) }
                        return rect
                    },
                    arguments: [original, local, wide]
                   ) {
                    image = cleaned.cropped(to: extent)
                }
            }

            context.render(image, to: output, bounds: extent, colorSpace: colorSpace)
            context.clearCaches()
            return output
        }
    }

    /// Light final compression polish only. No wide shadow processing is allowed here,
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
    func clean(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let shadowCleaned = try cleanShadowChroma(input)
        return try cleanFinalCompression(shadowCleaned)
    }
}
