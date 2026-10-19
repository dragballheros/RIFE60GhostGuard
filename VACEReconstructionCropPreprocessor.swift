import Foundation

enum VACEReconstructionCropError: LocalizedError {
    case dimensionMismatch
    case invalidTargetSize
    case targetTooLarge(Int)

    var errorDescription: String? {
        switch self {
        case .dimensionMismatch: return "VACE crop image and mask dimensions must match."
        case .invalidTargetSize: return "VACE inference crop size must be positive and aligned to 16 pixels."
        case .targetTooLarge(let pixels): return "VACE inference crop would contain \(pixels) pixels, over the memory budget."
        }
    }
}

struct VACEPreparedInferenceCrop {
    let image: VACERGBAFrame
    let mask: VACEMaskPlane
    let sourceWidth: Int
    let sourceHeight: Int
}

/// Resizes a bounded source crop for the VACE model while retaining full-resolution
/// source dimensions for the final composite. RGB uses bilinear half-pixel sampling;
/// masks use nearest-exact sampling so thin masks do not become semi-transparent
/// through interpolation. Generated output crops can be resized back to source size.
enum VACEReconstructionCropPreprocessor {
    static func prepare(
        image: VACERGBAFrame,
        mask: VACEMaskPlane,
        targetWidth: Int,
        targetHeight: Int,
        maximumInferencePixels: Int = 832 * 480
    ) throws -> VACEPreparedInferenceCrop {
        guard image.width == mask.width, image.height == mask.height else {
            throw VACEReconstructionCropError.dimensionMismatch
        }
        guard targetWidth > 0, targetHeight > 0,
              targetWidth.isMultiple(of: 16), targetHeight.isMultiple(of: 16),
              maximumInferencePixels > 0 else {
            throw VACEReconstructionCropError.invalidTargetSize
        }
        let targetPixels = targetWidth.multipliedReportingOverflow(by: targetHeight)
        guard !targetPixels.overflow, targetPixels.partialValue <= maximumInferencePixels else {
            throw VACEReconstructionCropError.targetTooLarge(targetPixels.overflow ? Int.max : targetPixels.partialValue)
        }
        let resizedImage = try resizeRGBA(image, width: targetWidth, height: targetHeight,
                                          maximumPixels: maximumInferencePixels)
        let resizedMask = try resizeMaskNearestExact(mask, width: targetWidth, height: targetHeight,
                                                      maximumPixels: maximumInferencePixels)
        return VACEPreparedInferenceCrop(image: resizedImage, mask: resizedMask,
                                         sourceWidth: image.width, sourceHeight: image.height)
    }

    static func restoreGeneratedCrop(
        _ generated: VACERGBAFrame,
        toSourceWidth width: Int,
        height: Int,
        maximumPixels: Int = 832 * 480
    ) throws -> VACERGBAFrame {
        try resizeRGBA(generated, width: width, height: height, maximumPixels: maximumPixels)
    }

    static func resizeRGBA(
        _ frame: VACERGBAFrame,
        width: Int,
        height: Int,
        maximumPixels: Int = 832 * 480
    ) throws -> VACERGBAFrame {
        guard width > 0, height > 0 else { throw VACEReconstructionCropError.invalidTargetSize }
        let product = width.multipliedReportingOverflow(by: height)
        guard !product.overflow, product.partialValue <= maximumPixels else {
            throw VACEReconstructionCropError.targetTooLarge(product.overflow ? Int.max : product.partialValue)
        }
        if frame.width == width && frame.height == height { return frame }
        var output = [UInt8](repeating: 0, count: product.partialValue * 4)
        let xScale = Float(frame.width) / Float(width)
        let yScale = Float(frame.height) / Float(height)
        for y in 0..<height {
            let sourceY = min(Float(frame.height - 1), max(0, (Float(y) + 0.5) * yScale - 0.5))
            let y0 = Int(sourceY.rounded(.down)), y1 = min(frame.height - 1, y0 + 1)
            let fy = sourceY - Float(y0)
            for x in 0..<width {
                let sourceX = min(Float(frame.width - 1), max(0, (Float(x) + 0.5) * xScale - 0.5))
                let x0 = Int(sourceX.rounded(.down)), x1 = min(frame.width - 1, x0 + 1)
                let fx = sourceX - Float(x0)
                let p00 = (y0 * frame.width + x0) * 4
                let p10 = (y0 * frame.width + x1) * 4
                let p01 = (y1 * frame.width + x0) * 4
                let p11 = (y1 * frame.width + x1) * 4
                let out = (y * width + x) * 4
                for channel in 0..<4 {
                    let top = Float(frame.bytes[p00 + channel]) +
                        (Float(frame.bytes[p10 + channel]) - Float(frame.bytes[p00 + channel])) * fx
                    let bottom = Float(frame.bytes[p01 + channel]) +
                        (Float(frame.bytes[p11 + channel]) - Float(frame.bytes[p01 + channel])) * fx
                    output[out + channel] = UInt8(min(255, max(0, Int((top + (bottom - top) * fy).rounded()))))
                }
            }
        }
        return try VACERGBAFrame(width: width, height: height, bytes: output)
    }

    static func resizeMaskNearestExact(
        _ mask: VACEMaskPlane,
        width: Int,
        height: Int,
        maximumPixels: Int = 832 * 480
    ) throws -> VACEMaskPlane {
        guard width > 0, height > 0 else { throw VACEReconstructionCropError.invalidTargetSize }
        let product = width.multipliedReportingOverflow(by: height)
        guard !product.overflow, product.partialValue <= maximumPixels else {
            throw VACEReconstructionCropError.targetTooLarge(product.overflow ? Int.max : product.partialValue)
        }
        if mask.width == width && mask.height == height { return mask }
        var output = [UInt8](repeating: 0, count: product.partialValue)
        for y in 0..<height {
            // Exact nearest interpolation uses floor((destination + 0.5) * input/output).
            let sourceY = min(mask.height - 1, ((2 * y + 1) * mask.height) / (2 * height))
            for x in 0..<width {
                let sourceX = min(mask.width - 1, ((2 * x + 1) * mask.width) / (2 * width))
                output[y * width + x] = mask[sourceX, sourceY]
            }
        }
        return try VACEMaskPlane(width: width, height: height, pixels: output)
    }
}
