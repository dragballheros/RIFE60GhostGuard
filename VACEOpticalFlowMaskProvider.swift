import CoreVideo
import Foundation
import Vision

enum VACEOpticalFlowProviderError: LocalizedError {
    case maskDoesNotMatchFrame
    case framesMustBeSequential(expected: Int, actual: Int)
    case flowUnavailable
    case visionFailed(String)

    var errorDescription: String? {
        switch self {
        case .maskDoesNotMatchFrame: return "The keyframe edit mask must match the first source frame."
        case .framesMustBeSequential(let expected, let actual):
            return "Optical-flow masks must be requested in order; expected frame \(expected), got \(actual)."
        case .flowUnavailable: return "Vision did not return an optical-flow pixel buffer."
        case .visionFailed(let message): return "On-device optical-flow analysis failed: \(message)"
        }
    }
}

private struct VACEFlowAnalysisFrame {
    let image: VACERGBAFrame
    let pixelBuffer: CVPixelBuffer
}

/// Propagates a hand-authored keyframe mask forward through a video using
/// Apple's on-device Vision optical flow. Seed the first frame's mask; calls
/// must then arrive in sequential frame order, as the streaming video pipeline
/// provides. A scene cut stops propagation and emits empty masks afterwards,
/// preventing the mask from jumping to an unrelated shot. Flow is computed on
/// a downscaled proxy frame to bound the per-frame analysis working set.
final class VACEOpticalFlowMaskProvider: VACEFrameMaskProvider {
    private let seedMask: VACEMaskPlane
    private let maximumFlowDimension: Int
    private let maximumDisplacementPixels: Float
    private let sceneCutMeanLumaThreshold: Float

    private var lastFrameIndex: Int?
    private var lastProxy: VACEFlowAnalysisFrame?
    private var currentFullResolutionMask: VACEMaskPlane?
    private var didReachSceneCut = false

    /// True after the provider detects a hard cut. It then returns an empty mask
    /// for the rest of the clip; callers can surface this as a new-shot warning.
    var sceneCutDetected: Bool { didReachSceneCut }

    init(
        firstFrameMask: VACEMaskPlane,
        maximumFlowDimension: Int = 640,
        maximumDisplacementPixels: Float = 64,
        sceneCutMeanLumaThreshold: Float = 48
    ) {
        self.seedMask = firstFrameMask
        self.maximumFlowDimension = max(64, maximumFlowDimension)
        self.maximumDisplacementPixels = max(1, maximumDisplacementPixels)
        self.sceneCutMeanLumaThreshold = min(255, max(1, sceneCutMeanLumaThreshold))
    }

    func mask(for frameIndex: Int, presentationTime: CMTime, source: VACERGBAFrame) async throws -> VACEMaskPlane {
        _ = presentationTime
        if lastFrameIndex == nil {
            guard frameIndex == 0,
                  source.width == seedMask.width,
                  source.height == seedMask.height else {
                throw VACEOpticalFlowProviderError.maskDoesNotMatchFrame
            }
            let proxy = try Self.makeProxy(source, maximumDimension: maximumFlowDimension)
            lastProxy = VACEFlowAnalysisFrame(image: proxy, pixelBuffer: try Self.makePixelBuffer(proxy))
            currentFullResolutionMask = seedMask
            lastFrameIndex = frameIndex
            return seedMask
        }

        guard let previousIndex = lastFrameIndex,
              frameIndex == previousIndex + 1,
              let previousProxy = lastProxy,
              let previousMask = currentFullResolutionMask else {
            throw VACEOpticalFlowProviderError.framesMustBeSequential(
                expected: (lastFrameIndex ?? -1) + 1, actual: frameIndex
            )
        }
        guard source.width == seedMask.width, source.height == seedMask.height else {
            throw VACEOpticalFlowProviderError.maskDoesNotMatchFrame
        }

        let currentProxyImage = try Self.makeProxy(source, maximumDimension: maximumFlowDimension)
        let currentProxy = VACEFlowAnalysisFrame(
            image: currentProxyImage, pixelBuffer: try Self.makePixelBuffer(currentProxyImage)
        )

        if Self.meanAbsoluteLumaDifference(previousProxy.image, currentProxy.image) >= sceneCutMeanLumaThreshold {
            didReachSceneCut = true
            let emptyMask = try VACEMaskPlane(
                width: source.width, height: source.height,
                pixels: [UInt8](repeating: 0, count: source.width * source.height)
            )
            currentFullResolutionMask = emptyMask
            lastProxy = currentProxy
            lastFrameIndex = frameIndex
            return emptyMask
        }

        if didReachSceneCut {
            let emptyMask = try VACEMaskPlane(
                width: source.width, height: source.height,
                pixels: [UInt8](repeating: 0, count: source.width * source.height)
            )
            currentFullResolutionMask = emptyMask
            lastProxy = currentProxy
            lastFrameIndex = frameIndex
            return emptyMask
        }

        // VNGenerateOpticalFlowRequest's field maps pixels in the targeted image
        // to corresponding locations in the handler image. Target=current and
        // handler=previous yields the backward field required for mask sampling.
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: currentProxy.pixelBuffer)
        request.computationAccuracy = .medium
        request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
        request.keepNetworkOutput = false
        let handler = VNImageRequestHandler(cvPixelBuffer: previousProxy.pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw VACEOpticalFlowProviderError.visionFailed(error.localizedDescription)
        }
        guard let flowBuffer = request.results?.first?.pixelBuffer else {
            throw VACEOpticalFlowProviderError.flowUnavailable
        }

        let lowResolutionMask = try Self.downsampleMaskByMaximum(
            previousMask, width: currentProxy.image.width, height: currentProxy.image.height
        )
        let warpedProxyMask = try Self.propagateProxyMask(
            lowResolutionMask, flowBuffer: flowBuffer,
            maximumDisplacementPixels: maximumDisplacementPixels
        )
        let fullResolutionMask = try Self.upsampleMaskBilinear(
            warpedProxyMask, width: source.width, height: source.height
        )
        currentFullResolutionMask = fullResolutionMask
        lastProxy = currentProxy
        lastFrameIndex = frameIndex
        return fullResolutionMask
    }

    private static func makeProxy(_ frame: VACERGBAFrame, maximumDimension: Int) throws -> VACERGBAFrame {
        let scale = min(1.0, Double(maximumDimension) / Double(max(frame.width, frame.height)))
        let width = max(1, Int((Double(frame.width) * scale).rounded(.down)))
        let height = max(1, Int((Double(frame.height) * scale).rounded(.down)))
        if width == frame.width && height == frame.height { return frame }

        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            let sourceY = min(frame.height - 1, Int(Double(y) / scale))
            for x in 0..<width {
                let sourceX = min(frame.width - 1, Int(Double(x) / scale))
                let sourceOffset = (sourceY * frame.width + sourceX) * 4
                let destinationOffset = (y * width + x) * 4
                bytes[destinationOffset] = frame.bytes[sourceOffset]
                bytes[destinationOffset + 1] = frame.bytes[sourceOffset + 1]
                bytes[destinationOffset + 2] = frame.bytes[sourceOffset + 2]
                bytes[destinationOffset + 3] = frame.bytes[sourceOffset + 3]
            }
        }
        return try VACERGBAFrame(width: width, height: height, bytes: bytes)
    }

    private static func makePixelBuffer(_ frame: VACERGBAFrame) throws -> CVPixelBuffer {
        var optionalBuffer: CVPixelBuffer?
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: frame.width,
            kCVPixelBufferHeightKey as String: frame.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, frame.width, frame.height,
                                         kCVPixelFormatType_32BGRA, attributes as CFDictionary, &optionalBuffer)
        guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
            throw VACEOpticalFlowProviderError.visionFailed("Could not allocate an analysis pixel buffer.")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw VACEOpticalFlowProviderError.visionFailed("Analysis pixel buffer has no base address.")
        }
        let destination = base.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<frame.height {
            let destinationRow = destination.advanced(by: y * rowBytes)
            for x in 0..<frame.width {
                let sourceOffset = (y * frame.width + x) * 4
                let destinationOffset = x * 4
                destinationRow[destinationOffset] = frame.bytes[sourceOffset + 2]
                destinationRow[destinationOffset + 1] = frame.bytes[sourceOffset + 1]
                destinationRow[destinationOffset + 2] = frame.bytes[sourceOffset]
                destinationRow[destinationOffset + 3] = frame.bytes[sourceOffset + 3]
            }
        }
        return buffer
    }

    private static func downsampleMaskByMaximum(_ mask: VACEMaskPlane, width: Int, height: Int) throws -> VACEMaskPlane {
        guard width > 0, height > 0 else { throw VACEOpticalFlowWarpError.invalidDimensions }
        var output = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            let y0 = min(mask.height - 1, y * mask.height / height)
            let y1 = max(y0 + 1, min(mask.height, (y + 1) * mask.height / height))
            for x in 0..<width {
                let x0 = min(mask.width - 1, x * mask.width / width)
                let x1 = max(x0 + 1, min(mask.width, (x + 1) * mask.width / width))
                var maximum: UInt8 = 0
                for sy in y0..<y1 {
                    for sx in x0..<x1 { maximum = max(maximum, mask[sx, sy]) }
                }
                output[y * width + x] = maximum
            }
        }
        return try VACEMaskPlane(width: width, height: height, pixels: output)
    }

    private static func propagateProxyMask(
        _ mask: VACEMaskPlane,
        flowBuffer: CVPixelBuffer,
        maximumDisplacementPixels: Float
    ) throws -> VACEMaskPlane {
        CVPixelBufferLockBaseAddress(flowBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowBuffer, .readOnly) }
        let width = CVPixelBufferGetWidth(flowBuffer), height = CVPixelBufferGetHeight(flowBuffer)
        guard width == mask.width, height == mask.height,
              CVPixelBufferGetPixelFormatType(flowBuffer) == kCVPixelFormatType_TwoComponent32Float,
              let base = CVPixelBufferGetBaseAddress(flowBuffer) else {
            throw VACEOpticalFlowProviderError.flowUnavailable
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(flowBuffer)
        let floatStride = rowBytes / MemoryLayout<Float>.stride
        let baseFloats = base.assumingMemoryBound(to: Float.self)
        var xy = [Float](repeating: 0, count: width * height * 2)
        for y in 0..<height {
            let sourceRow = baseFloats.advanced(by: y * floatStride)
            for x in 0..<width {
                let sourceOffset = x * 2
                let destinationOffset = (y * width + x) * 2
                xy[destinationOffset] = sourceRow[sourceOffset]
                xy[destinationOffset + 1] = sourceRow[sourceOffset + 1]
            }
        }
        let flow = try VACEBackwardFlowField(width: width, height: height, xy: xy)
        return try VACEOpticalFlowMaskWarp.propagate(
            previousMask: mask, backwardFlow: flow, maximumDisplacementPixels: maximumDisplacementPixels
        )
    }

    private static func upsampleMaskBilinear(_ mask: VACEMaskPlane, width: Int, height: Int) throws -> VACEMaskPlane {
        if mask.width == width && mask.height == height { return mask }
        var output = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            let sourceY = min(Float(mask.height - 1), max(0, (Float(y) + 0.5) * Float(mask.height) / Float(height) - 0.5))
            let y0 = Int(sourceY.rounded(.down)), y1 = min(mask.height - 1, y0 + 1)
            let fy = sourceY - Float(y0)
            for x in 0..<width {
                let sourceX = min(Float(mask.width - 1), max(0, (Float(x) + 0.5) * Float(mask.width) / Float(width) - 0.5))
                let x0 = Int(sourceX.rounded(.down)), x1 = min(mask.width - 1, x0 + 1)
                let fx = sourceX - Float(x0)
                let p00 = Float(mask[x0, y0]), p10 = Float(mask[x1, y0])
                let p01 = Float(mask[x0, y1]), p11 = Float(mask[x1, y1])
                let top = p00 + (p10 - p00) * fx
                let bottom = p01 + (p11 - p01) * fx
                output[y * width + x] = UInt8(min(255, max(0, Int((top + (bottom - top) * fy).rounded()))))
            }
        }
        return try VACEMaskPlane(width: width, height: height, pixels: output)
    }

    private static func meanAbsoluteLumaDifference(_ lhs: VACERGBAFrame, _ rhs: VACERGBAFrame) -> Float {
        guard lhs.width == rhs.width, lhs.height == rhs.height else { return 255 }
        let pixelCount = lhs.width * lhs.height
        // Sampling every fourth pixel bounds this check; a hard cut normally
        // changes a large fraction of the frame, so full resolution is unnecessary.
        let sampleStep = 4
        var absoluteDifference: Double = 0
        var samples = 0
        for pixel in Swift.stride(from: 0, to: pixelCount, by: sampleStep) {
            let offset = pixel * 4
            let leftLuma = 0.2126 * Double(lhs.bytes[offset]) + 0.7152 * Double(lhs.bytes[offset + 1]) + 0.0722 * Double(lhs.bytes[offset + 2])
            let rightLuma = 0.2126 * Double(rhs.bytes[offset]) + 0.7152 * Double(rhs.bytes[offset + 1]) + 0.0722 * Double(rhs.bytes[offset + 2])
            absoluteDifference += abs(leftLuma - rightLuma)
            samples += 1
        }
        return samples == 0 ? 0 : Float(absoluteDifference / Double(samples))
    }
}
