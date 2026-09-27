import Foundation
import CoreVideo
import RifeMetal

/// Runs RIFE HQ on overlapping horizontal bands so peak Metal memory stays
/// well below a full-frame HQ graph. Each band uses the same full-quality HQ
/// network; only the spatial working set is reduced. Overlap is discarded when
/// stitching, avoiding visible seams while preserving context near boundaries.
final class TiledHQInterpolator {
    private let interpolator: RifeInterpolator
    private let width: Int
    private let height: Int
    private let bandCount: Int
    private let overlap: Int
    private let coreHeight: Int
    private let tileHeight: Int

    init(interpolator: RifeInterpolator,
         width: Int,
         height: Int,
         bandCount: Int = 3,
         overlap: Int = 64) {
        self.interpolator = interpolator
        self.width = width
        self.height = height
        self.bandCount = max(1, bandCount)
        self.overlap = max(0, overlap)
        self.coreHeight = Int(ceil(Double(height) / Double(max(1, bandCount))))
        self.tileHeight = self.coreHeight + self.overlap * 2
    }

    func interpolate(previous: CVPixelBuffer,
                     current: CVPixelBuffer,
                     timesteps: [Float]) throws -> [CVPixelBuffer] {
        guard !timesteps.isEmpty else { return [] }

        var fullOutputs: [CVPixelBuffer] = []
        fullOutputs.reserveCapacity(timesteps.count)
        for _ in timesteps {
            fullOutputs.append(try makeBuffer(width: width, height: height))
        }

        for band in 0..<bandCount {
            let coreStart = band * coreHeight
            guard coreStart < height else { break }
            let coreCount = min(coreHeight, height - coreStart)

            let previousTile = try extractBand(
                previous,
                coreStart: coreStart
            )
            let currentTile = try extractBand(
                current,
                coreStart: coreStart
            )

            let tileOutputs = try autoreleasepool {
                try interpolator.interpolate(
                    previous: previousTile,
                    current: currentTile,
                    timesteps: timesteps
                )
            }

            guard tileOutputs.count == fullOutputs.count else {
                throw ProcessorError.conversionFailed("Tiled HQ RIFE returned an unexpected frame count")
            }

            for index in tileOutputs.indices {
                try copyCore(
                    from: tileOutputs[index],
                    to: fullOutputs[index],
                    coreStart: coreStart,
                    coreCount: coreCount
                )
            }
        }

        return fullOutputs
    }

    private func makeBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let buffer else {
            throw ProcessorError.conversionFailed("could not allocate tiled HQ frame")
        }
        return buffer
    }

    private func extractBand(_ source: CVPixelBuffer,
                             coreStart: Int) throws -> CVPixelBuffer {
        let format = CVPixelBufferGetPixelFormatType(source)
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA else {
            throw ProcessorError.conversionFailed("tiled HQ requires BGRA/RGBA input")
        }

        let tile = try makeBuffer(width: width, height: tileHeight)

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(tile, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(tile, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(source),
              let dstBase = CVPixelBufferGetBaseAddress(tile) else {
            throw ProcessorError.conversionFailed("tiled HQ band base address unavailable")
        }

        let srcRow = CVPixelBufferGetBytesPerRow(source)
        let dstRow = CVPixelBufferGetBytesPerRow(tile)
        let rowBytes = width * 4

        for y in 0..<tileHeight {
            let sourceY = min(max(coreStart + y - overlap, 0), height - 1)
            let s = srcBase.advanced(by: sourceY * srcRow)
            let d = dstBase.advanced(by: y * dstRow)
            memcpy(d, s, rowBytes)
        }

        return tile
    }

    private func copyCore(from tile: CVPixelBuffer,
                          to full: CVPixelBuffer,
                          coreStart: Int,
                          coreCount: Int) throws {
        CVPixelBufferLockBaseAddress(tile, .readOnly)
        CVPixelBufferLockBaseAddress(full, [])
        defer {
            CVPixelBufferUnlockBaseAddress(tile, .readOnly)
            CVPixelBufferUnlockBaseAddress(full, [])
        }

        guard let tileBase = CVPixelBufferGetBaseAddress(tile),
              let fullBase = CVPixelBufferGetBaseAddress(full) else {
            throw ProcessorError.conversionFailed("tiled HQ stitch base address unavailable")
        }

        let tileRow = CVPixelBufferGetBytesPerRow(tile)
        let fullRow = CVPixelBufferGetBytesPerRow(full)
        let rowBytes = width * 4

        for y in 0..<coreCount {
            let s = tileBase.advanced(by: (overlap + y) * tileRow)
            let d = fullBase.advanced(by: (coreStart + y) * fullRow)
            memcpy(d, s, rowBytes)
        }
    }
}
