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

            // Smooth chroma while restoring the source luminance. This removes
            // red/green/blue shadow crawling without washing out legitimate blacks.
            vec3 chromaClean = clamp(s.rgb + vec3(yo - ys), 0.0, 1.0);

            // Only operate in shadows. Full strength below ~10% luma and smoothly
            // disappear before midtones so normal anime colors remain unchanged.
            float shadow = 1.0 - smoothstep(0.10, 0.30, yo);

            // Do not smear outlines, hard cel-shading boundaries, text, or detail.
            float localDelta = length(o.rgb - s.rgb);
            float edgeProtect = 1.0 - smoothstep(0.035, 0.105, localDelta);

            float amount = 0.78 * shadow * edgeProtect;
            vec3 rgb = mix(o.rgb, chromaClean, amount);
            return vec4(rgb, o.a);
        }
        """
        return CIKernel(source: source)
    }()

    func clean(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        try autoreleasepool {
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
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                width,
                height,
                kCVPixelFormatType_32BGRA,
                attrs as CFDictionary,
                &output
            )
            guard status == kCVReturnSuccess, let output else {
                throw Error.allocationFailed
            }

            var image = CIImage(cvPixelBuffer: input)

            // First remove the colored crawling/blotching that is most visible in
            // near-black gradients. This is deliberately before general NR/sharpen.
            if let kernel = Self.shadowChromaKernel,
               let blur = CIFilter(name: "CIGaussianBlur") {
                blur.setValue(image, forKey: kCIInputImageKey)
                blur.setValue(1.45, forKey: kCIInputRadiusKey)
                if let smooth = blur.outputImage?.cropped(to: extent),
                   let cleaned = kernel.apply(
                    extent: extent,
                    roiCallback: { index, rect in
                        index == 1 ? rect.insetBy(dx: -3, dy: -3) : rect
                    },
                    arguments: [image, smooth]
                   ) {
                    image = cleaned.cropped(to: extent)
                }
            }

            if let noise = CIFilter(name: "CINoiseReduction") {
                noise.setValue(image, forKey: kCIInputImageKey)
                noise.setValue(0.012, forKey: "inputNoiseLevel")
                noise.setValue(0.50, forKey: "inputSharpness")
                if let result = noise.outputImage {
                    image = result
                }
            }

            if let sharpen = CIFilter(name: "CISharpenLuminance") {
                sharpen.setValue(image, forKey: kCIInputImageKey)
                sharpen.setValue(0.16, forKey: kCIInputSharpnessKey)
                sharpen.setValue(1.0, forKey: kCIInputRadiusKey)
                if let result = sharpen.outputImage {
                    image = result
                }
            }

            context.render(
                image.cropped(to: extent),
                to: output,
                bounds: extent,
                colorSpace: colorSpace
            )

            // Do not let Core Image retain temporary textures across a long video.
            context.clearCaches()
            return output
        }
    }
}
