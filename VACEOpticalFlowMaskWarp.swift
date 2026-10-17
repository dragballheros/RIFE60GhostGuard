import Foundation

enum VACEOpticalFlowWarpError: LocalizedError {
    case invalidDimensions
    case valueCountMismatch(expected: Int, actual: Int)
    case maskFlowDimensionMismatch
    case unsafeFlowField

    var errorDescription: String? {
        switch self {
        case .invalidDimensions: return "Optical-flow dimensions must be positive."
        case .valueCountMismatch(let expected, let actual): return "Flow field has \(actual) values; expected \(expected)."
        case .maskFlowDimensionMismatch: return "Mask and optical-flow grid dimensions must match."
        case .unsafeFlowField: return "Optical-flow field exceeds the configured safety limit."
        }
    }
}

/// Dense backward flow: each target-grid pixel stores the source-grid displacement
/// to sample when producing the target mask at that pixel, in (dx,dy) pixel units.
struct VACEBackwardFlowField {
    let width: Int
    let height: Int
    let xy: [Float]

    init(width: Int, height: Int, xy: [Float]) throws {
        guard width > 0, height > 0 else { throw VACEOpticalFlowWarpError.invalidDimensions }
        let pixelCount = width.multipliedReportingOverflow(by: height)
        guard !pixelCount.overflow, pixelCount.partialValue <= 8_388_608 else {
            throw VACEOpticalFlowWarpError.unsafeFlowField
        }
        let valueCount = pixelCount.partialValue.multipliedReportingOverflow(by: 2)
        guard !valueCount.overflow else { throw VACEOpticalFlowWarpError.unsafeFlowField }
        guard xy.count == valueCount.partialValue else {
            throw VACEOpticalFlowWarpError.valueCountMismatch(expected: valueCount.partialValue, actual: xy.count)
        }
        self.width = width
        self.height = height
        self.xy = xy
    }

    func displacement(x: Int, y: Int) -> (dx: Float, dy: Float) {
        let offset = (y * width + x) * 2
        return (xy[offset], xy[offset + 1])
    }
}

/// Pure mask propagation operation; can be unit-tested without Vision. It uses
/// backward flow so each target pixel samples a location in the previous mask,
/// avoiding the holes typical of forward splatting. Unsafe/extreme flow vectors
/// clear the target mask at that pixel instead of stretching edits across cuts.
enum VACEOpticalFlowMaskWarp {
    static func propagate(
        previousMask: VACEMaskPlane,
        backwardFlow: VACEBackwardFlowField,
        maximumDisplacementPixels: Float = 96
    ) throws -> VACEMaskPlane {
        guard previousMask.width == backwardFlow.width,
              previousMask.height == backwardFlow.height else {
            throw VACEOpticalFlowWarpError.maskFlowDimensionMismatch
        }
        guard maximumDisplacementPixels.isFinite, maximumDisplacementPixels > 0 else {
            throw VACEOpticalFlowWarpError.unsafeFlowField
        }
        let width = previousMask.width, height = previousMask.height
        var output = [UInt8](repeating: 0, count: previousMask.pixelCount)

        for y in 0..<height {
            for x in 0..<width {
                let vector = backwardFlow.displacement(x: x, y: y)
                guard vector.dx.isFinite, vector.dy.isFinite,
                      abs(vector.dx) <= maximumDisplacementPixels,
                      abs(vector.dy) <= maximumDisplacementPixels else { continue }
                let sourceX = Float(x) + vector.dx
                let sourceY = Float(y) + vector.dy
                guard sourceX >= 0, sourceY >= 0,
                      sourceX <= Float(width - 1), sourceY <= Float(height - 1) else { continue }

                let x0 = Int(sourceX.rounded(.down)), y0 = Int(sourceY.rounded(.down))
                let x1 = min(width - 1, x0 + 1), y1 = min(height - 1, y0 + 1)
                let fx = sourceX - Float(x0), fy = sourceY - Float(y0)
                let p00 = Float(previousMask[x0, y0])
                let p10 = Float(previousMask[x1, y0])
                let p01 = Float(previousMask[x0, y1])
                let p11 = Float(previousMask[x1, y1])
                let upper = p00 + (p10 - p00) * fx
                let lower = p01 + (p11 - p01) * fx
                let interpolated = upper + (lower - upper) * fy
                output[y * width + x] = UInt8(min(255, max(0, Int(interpolated.rounded()))))
            }
        }
        return try VACEMaskPlane(width: width, height: height, pixels: output)
    }
}
