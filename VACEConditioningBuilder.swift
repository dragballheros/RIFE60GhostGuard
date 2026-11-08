import Foundation

enum VACEConditioningBuildError: LocalizedError {
    case dimensionMismatch
    case invalidStride
    case incompatibleSpatialDimensions(width: Int, height: Int, strideX: Int, strideY: Int)
    case tensorTooLarge(elements: Int)
    case inputTooLarge(pixels: Int)

    var errorDescription: String? {
        switch self {
        case .dimensionMismatch:
            return "VACE source frame and edit mask dimensions must match."
        case .invalidStride:
            return "VACE temporal and spatial strides must all be positive."
        case .incompatibleSpatialDimensions(let width, let height, let strideX, let strideY):
            return "VACE mask dimensions \(width)x\(height) must be divisible by twice the configured spatial strides \(strideX)x\(strideY)."
        case .tensorTooLarge(let elements):
            return "VACE control-mask tensor would contain \(elements) values, over the configured safety limit."
        case .inputTooLarge(let pixels):
            return "VACE frame conditioning crop has \(pixels) pixels, over the configured working-set limit."
        }
    }
}

/// Per-frame RGB conditioning tensors in channel-first order (CHW), normalized
/// to [-1, 1]. These are the two masked image inputs encoded separately by the
/// VACE VAE; they are not diffusion noise and are not the final generated frame.
struct VACEFrameSplitConditioning {
    let width: Int
    let height: Int
    let inactiveRGBCHW: [Float]
    let reactiveRGBCHW: [Float]

    var channelPlaneSize: Int { width * height }
}

/// Implements the source-image and edit-mask conventions used by the upstream
/// Wan2.1 VACE encoder. White/nonzero mask values above 0.5 mean "replace".
/// For VAE conditioning, the two branches contain normalized source RGB only
/// where they are active; pixels outside each branch are set to numeric zero.
enum VACEConditioningBuilder {
    static func splitFrame(
        _ frame: VACERGBAFrame,
        editMask: VACEMaskPlane,
        maximumPixels: Int = 832 * 480
    ) throws -> VACEFrameSplitConditioning {
        guard frame.width == editMask.width, frame.height == editMask.height else {
            throw VACEConditioningBuildError.dimensionMismatch
        }
        guard maximumPixels > 0 else { throw VACEConditioningBuildError.inputTooLarge(pixels: frame.width * frame.height) }
        let pixelCount = frame.width.multipliedReportingOverflow(by: frame.height)
        guard !pixelCount.overflow, pixelCount.partialValue <= maximumPixels else {
            throw VACEConditioningBuildError.inputTooLarge(pixels: pixelCount.overflow ? Int.max : pixelCount.partialValue)
        }
        let planeSize = pixelCount.partialValue
        let floatCount = planeSize.multipliedReportingOverflow(by: 3)
        guard !floatCount.overflow, floatCount.partialValue <= Int.max / MemoryLayout<Float>.stride else {
            throw VACEConditioningBuildError.inputTooLarge(pixels: planeSize)
        }

        var inactive = [Float](repeating: 0, count: floatCount.partialValue)
        var reactive = [Float](repeating: 0, count: floatCount.partialValue)
        for pixel in 0..<planeSize {
            let maskIsEditable = editMask.pixels[pixel] > 127
            let sourceOffset = pixel * 4
            for channel in 0..<3 {
                let normalized = Float(frame.bytes[sourceOffset + channel]) / 127.5 - 1.0
                let destinationOffset = channel * planeSize + pixel
                if maskIsEditable {
                    reactive[destinationOffset] = normalized
                } else {
                    inactive[destinationOffset] = normalized
                }
            }
        }
        return VACEFrameSplitConditioning(
            width: frame.width, height: frame.height,
            inactiveRGBCHW: inactive, reactiveRGBCHW: reactive
        )
    }

    /// Reorders a window of per-frame edit masks exactly into VACE's control
    /// mask channel layout: each spatial stride cell is rearranged into
    /// strideY*strideX channels, then the temporal dimension is resized with
    /// nearest-exact sampling to ceil(frameCount/temporalStride).
    ///
    /// Upstream Wan2.1 VACE does not simply average-pool a white mask to one
    /// channel. It preserves sub-pixel locations as channels for the VACE
    /// control path. Spatial dimensions therefore must match VACE's required
    /// 2*stride divisibility; callers should crop/pad frames before this stage.
    static func packControlMask(
        _ masks: [VACEMaskPlane],
        temporalStride: Int = 4,
        spatialStrideY: Int = 8,
        spatialStrideX: Int = 8,
        referenceLatentFrames: Int = 0,
        maximumTensorElements: Int = 8_388_608
    ) throws -> VACEControlMaskTensor {
        guard temporalStride > 0, spatialStrideY > 0, spatialStrideX > 0,
              referenceLatentFrames >= 0, maximumTensorElements > 0 else {
            throw VACEConditioningBuildError.invalidStride
        }
        guard let first = masks.first else {
            return VACEControlMaskTensor(channels: spatialStrideY * spatialStrideX,
                                         frames: 0, height: 0, width: 0, values: [])
        }
        guard masks.allSatisfy({ $0.width == first.width && $0.height == first.height }) else {
            throw VACEConditioningBuildError.dimensionMismatch
        }
        let requiredMultipleY = spatialStrideY.multipliedReportingOverflow(by: 2)
        let requiredMultipleX = spatialStrideX.multipliedReportingOverflow(by: 2)
        guard !requiredMultipleY.overflow, !requiredMultipleX.overflow,
              first.height % requiredMultipleY.partialValue == 0,
              first.width % requiredMultipleX.partialValue == 0 else {
            throw VACEConditioningBuildError.incompatibleSpatialDimensions(
                width: first.width, height: first.height,
                strideX: spatialStrideX, strideY: spatialStrideY
            )
        }

        let latentHeight = 2 * (first.height / requiredMultipleY.partialValue)
        let latentWidth = 2 * (first.width / requiredMultipleX.partialValue)
        let latentFrameCount = (masks.count + temporalStride - 1) / temporalStride
        let outputFrames = referenceLatentFrames + latentFrameCount
        let channels = spatialStrideY.multipliedReportingOverflow(by: spatialStrideX)
        guard !channels.overflow else { throw VACEConditioningBuildError.invalidStride }
        let channelFrames = channels.partialValue.multipliedReportingOverflow(by: outputFrames)
        guard !channelFrames.overflow else { throw VACEConditioningBuildError.tensorTooLarge(elements: Int.max) }
        let spatial = latentHeight.multipliedReportingOverflow(by: latentWidth)
        guard !spatial.overflow else { throw VACEConditioningBuildError.tensorTooLarge(elements: Int.max) }
        let total = channelFrames.partialValue.multipliedReportingOverflow(by: spatial.partialValue)
        guard !total.overflow, total.partialValue <= maximumTensorElements else {
            throw VACEConditioningBuildError.tensorTooLarge(elements: total.overflow ? Int.max : total.partialValue)
        }
        var output = [Float](repeating: 0, count: total.partialValue)
        let inputWidth = first.width
        let outputSpatial = spatial.partialValue
        let perChannelFrameSize = outputSpatial

        // Prefix zeros for reference-image latent frames, matching upstream VACE.
        for channel in 0..<channels.partialValue {
            for outFrame in 0..<latentFrameCount {
                // PyTorch interpolate(mode="nearest-exact") selects floor((i+0.5)*in/out).
                let numerator = (2 * outFrame + 1).multipliedReportingOverflow(by: masks.count)
                guard !numerator.overflow else { throw VACEConditioningBuildError.tensorTooLarge(elements: Int.max) }
                let sourceFrame = min(masks.count - 1, numerator.partialValue / (2 * latentFrameCount))
                let subY = channel / spatialStrideX
                let subX = channel % spatialStrideX
                for latentY in 0..<latentHeight {
                    for latentX in 0..<latentWidth {
                        let sourceY = latentY * spatialStrideY + subY
                        let sourceX = latentX * spatialStrideX + subX
                        let sourcePixel = sourceFrame * first.pixelCount + sourceY * inputWidth + sourceX
                        let isEditable = masks[sourceFrame].pixels[sourcePixel - sourceFrame * first.pixelCount] > 127
                        let destinationIndex =
                            channel * outputFrames * perChannelFrameSize +
                            (referenceLatentFrames + outFrame) * perChannelFrameSize +
                            latentY * latentWidth + latentX
                        output[destinationIndex] = isEditable ? 1.0 : 0.0
                    }
                }
            }
        }
        return VACEControlMaskTensor(
            channels: channels.partialValue,
            frames: outputFrames,
            height: latentHeight,
            width: latentWidth,
            values: output
        )
    }
}

struct VACEControlMaskTensor {
    /// Channel order is (subY * spatialStrideX + subX), matching VACE rearrangement.
    let channels: Int
    let frames: Int
    let height: Int
    let width: Int
    /// Channel-first, then time, then row-major spatial values.
    let values: [Float]

    subscript(channel: Int, frame: Int, y: Int, x: Int) -> Float {
        values[((channel * frames + frame) * height + y) * width + x]
    }
}
