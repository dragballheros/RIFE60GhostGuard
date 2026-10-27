import Foundation

/// BGRA tile crop and stitch that follow Core Image's bottom-left coordinates
/// and clamped-to-extent edge repeat. Callers lock the pixel buffers.
enum RealCUGANTileCopy {
    static func crop(
        source: UnsafePointer<UInt8>,
        sourceStride: Int,
        sourceWidth: Int,
        sourceHeight: Int,
        destination: UnsafeMutablePointer<UInt8>,
        destinationStride: Int,
        ciOriginX: Int,
        ciOriginY: Int,
        tileWidth: Int,
        tileHeight: Int
    ) {
        for row in 0..<tileHeight {
            let ciY = ciOriginY + (tileHeight - 1 - row)
            let clampedY = min(max(ciY, 0), sourceHeight - 1)
            let sourceRow = sourceHeight - 1 - clampedY
            let dst = destination.advanced(by: row * destinationStride)
            for col in 0..<tileWidth {
                let ciX = ciOriginX + col
                let clampedX = min(max(ciX, 0), sourceWidth - 1)
                let src = source.advanced(by: sourceRow * sourceStride + clampedX * 4)
                dst.advanced(by: col * 4).update(from: src, count: 4)
            }
        }
    }

    /// Copies the valid core out of a 2× model output into the stitched frame.
    /// `coreX` and `coreY` are source-frame CI coordinates. The core in the
    /// model output starts `overlap * 2` pixels up from that buffer's CI origin.
    static func stitch(
        modelOutput: UnsafePointer<UInt8>,
        outputStride: Int,
        outputHeight: Int,
        destination: UnsafeMutablePointer<UInt8>,
        destinationStride: Int,
        destinationHeight: Int,
        coreX: Int,
        coreY: Int,
        coreWidth: Int,
        coreHeight: Int,
        overlap: Int
    ) {
        let copyWidth = coreWidth * 2
        let copyHeight = coreHeight * 2
        let outputRow = outputHeight - (overlap * 2) - copyHeight
        let destRow = destinationHeight - (coreY * 2) - copyHeight
        let byteCount = copyWidth * 4
        for row in 0..<copyHeight {
            let src = modelOutput.advanced(by: (outputRow + row) * outputStride + overlap * 2 * 4)
            let dst = destination.advanced(by: (destRow + row) * destinationStride + coreX * 2 * 4)
            dst.update(from: src, count: byteCount)
        }
    }
}
