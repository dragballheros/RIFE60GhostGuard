import Foundation
import CoreImage
import CoreVideo

final class CompressionGuard {
    enum Error: Swift.Error {
        case allocationFailed
    }

    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

    // Shadow cleanup for dark anime backgrounds. Build 115 deliberately restored the
    // exact source luminance, which meant dark luma macroblocking/banding could survive
    // even when the colored noise was removed. This version cleans BOTH components:
    // chroma strongly and luma moderately, but only in flat shadows and before Sharpie.
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

            // Restrict the correction to dark material. It is strongest in deep blacks
            // and smoothly disappears before normal midtones / skin / bright cel colors.
            float shadow = 1.0 - smoothstep(0.09, 0.31, yo);

            // Detect true line/cel edges with a tiny local reference. Broad compression
            // blotches and banding are intentionally NOT treated as edges, so they can
            // be flattened by the wider reference.
            float fineDelta = length(o.rgb - l.rgb);
            float edgeProtect = 1.0 - smoothstep(0.045, 0.125, fineDelta);

            // Wide reference removes the large dark patches visible in flat backgrounds.
            // Chroma follows it strongly. Luma follows it only 58%, preserving intended
            // lighting gradients while suppressing block/band brightness variation.
            float targetY = mix(yo, yw, 0.58);
            vec3 target = clamp(w.rgb + vec3(targetY - yw), 0.0, 1.0);

            float amount = 0.88 * shadow * edgeProtect;
            vec3 rgb = mix(o.rgb, target, amount);
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

    /// Deep-shadow noise cleanup. Kept under the existing method name so the final
    /// pipeline order remains unchanged: CUGAN -> shadow cleanup -> Sharpie -> polish.
    /// Unlike Build 115, this also suppresses low-frequency luminance blotching.
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

    /// Light final compression polish only. No wide shadow blur is allowed here,
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
