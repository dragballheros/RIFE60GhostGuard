import Foundation
import CoreImage
import CoreVideo

final class CompressionGuard {
    enum Error: Swift.Error {
        case allocationFailed
        case renderFailed
    }

    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    func clean(_ input: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)

        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
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

        // Conservative cleanup intended for compressed anime sources:
        // remove low-amplitude ringing/mosquito noise without softening line art heavily.
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
            image,
            to: output,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            colorSpace: colorSpace
        )
        return output
    }
}
