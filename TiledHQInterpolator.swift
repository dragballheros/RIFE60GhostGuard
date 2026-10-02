import Foundation
import CoreVideo
import RifeMetal

/// Runs true RIFE HQ using the smallest spatial split needed for memory safety.
/// 1080p-class video uses one persistent full-frame RifeStream. For larger
/// sources, automatic Performance Mode uses two larger bands while Nominal/Fair;
/// a render that starts in Thermal Safe Mode uses the conservative 3-band layout.
/// Model quality and timesteps are identical in both modes.
final class TiledHQInterpolator {
    private let width: Int
    private let height: Int
    private var bandCount: Int
    private let overlap: Int
    private var coreHeight: Int
    private var tileHeight: Int
    private let streams: [RifeStream]
    private var seeded = false
    private let memoryAdaptive: Bool
    private let sharedInterpolator: RifeInterpolator?
    private var sharedStream: RifeStream?
    private var previousFrame: CVPixelBuffer?

    init(interpolator: RifeInterpolator,
         width: Int,
         height: Int,
         bandCount: Int = 3,
         overlap: Int = 32,
         memoryAdaptive: Bool = false) throws {
        self.memoryAdaptive = memoryAdaptive
        self.sharedInterpolator = memoryAdaptive ? interpolator : nil
        self.width = width
        self.height = height

        // 1080p-class remains the fastest/lowest-overhead full-frame path. Above
        // that, use fewer/larger bands when thermal headroom exists. We never
        // rebuild a stateful RIFE stream in the middle of an active frame.
        let performanceMode = automaticPerformanceModeEnabled()
        let adaptiveBands: Int
        if memoryAdaptive {
            let state = currentRenderPerformanceSnapshot()
            adaptiveBands = RIFEBandPlan.bands(width: width, height: height, availableMB: state.availableMemoryMB, underPressure: state.tier == .safe)
        } else if height <= 1200 {
            adaptiveBands = 1
        } else if performanceMode {
            adaptiveBands = 2
        } else {
            adaptiveBands = max(3, bandCount)
        }

        self.bandCount = adaptiveBands
        self.overlap = adaptiveBands == 1 ? 0 : (memoryAdaptive ? max(0, overlap) : (performanceMode ? min(max(0, overlap), 48) : max(0, overlap)))
        self.coreHeight = Int(ceil(Double(height) / Double(adaptiveBands)))
        self.tileHeight = adaptiveBands == 1 ? height : self.coreHeight + self.overlap * 2

        DiagnosticsLogger.shared.log("RIFE HQ layout • \(width)x\(height) • bands=\(adaptiveBands) • overlap=\(self.overlap) • \(currentThermalStateName())")

        var sessions: [RifeStream] = []
        sessions.reserveCapacity(adaptiveBands)
        for _ in 0..<(memoryAdaptive ? 0 : adaptiveBands) {
            sessions.append(try interpolator.makeStream(width: width, height: tileHeight))
        }
        self.streams = sessions
    }

    func seed(_ frame: CVPixelBuffer) throws {
        if memoryAdaptive {
            previousFrame = frame
            seeded = true
            return
        }
        if bandCount == 1 {
            let outputs = try autoreleasepool {
                try streams[0].push(frame, timesteps: [])
            }
            guard outputs.isEmpty else {
                throw ProcessorError.conversionFailed("HQ stream seed unexpectedly returned output")
            }
            seeded = true
            return
        }

        for band in 0..<bandCount {
            let coreStart = band * coreHeight
            guard coreStart < height else { break }
            let tile = try extractBand(frame, coreStart: coreStart)
            let outputs = try autoreleasepool {
                try streams[band].push(tile, timesteps: [])
            }
            guard outputs.isEmpty else {
                throw ProcessorError.conversionFailed("HQ stream seed unexpectedly returned output")
            }
        }
        seeded = true
    }

    func interpolate(current: CVPixelBuffer,
                     timesteps: [Float]) throws -> [CVPixelBuffer] {
        guard seeded else {
            throw ProcessorError.conversionFailed("HQ streams were not seeded")
        }

        if memoryAdaptive { return try interpolateShared(current: current, timesteps: timesteps) }

        // Fast path: no extraction, stitching, overlap, or second RIFE stream.
        if bandCount == 1 {
            return try autoreleasepool {
                try streams[0].push(current, timesteps: timesteps)
            }
        }

        var fullOutputs: [CVPixelBuffer] = []
        fullOutputs.reserveCapacity(timesteps.count)
        for _ in timesteps {
            fullOutputs.append(try makeBuffer(width: width, height: height))
        }

        for band in 0..<bandCount {
            let coreStart = band * coreHeight
            guard coreStart < height else { break }
            let coreCount = min(coreHeight, height - coreStart)
            let currentTile = try extractBand(current, coreStart: coreStart)

            let tileOutputs = try autoreleasepool {
                try streams[band].push(currentTile, timesteps: timesteps)
            }

            guard tileOutputs.count == timesteps.count else {
                throw ProcessorError.conversionFailed("Streaming tiled HQ RIFE returned an unexpected frame count")
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

    /// Every push waits for GPU completion. Rebase one stream to the previous
    /// band before each current band, so states from different bands never mix.
    private func interpolateShared(current: CVPixelBuffer, timesteps: [Float]) throws -> [CVPixelBuffer] {
        guard let previousFrame, let sharedInterpolator else { throw ProcessorError.conversionFailed("adaptive RIFE has no previous frame") }
        try adaptAtFrameBoundary()
        if timesteps.isEmpty { self.previousFrame = current; return [] }
        var outputs: [CVPixelBuffer] = []
        for _ in timesteps { outputs.append(try makeBuffer(width: width, height: height)) }
        if sharedStream == nil { sharedStream = try sharedInterpolator.makeStream(width: width, height: tileHeight) }
        guard let stream = sharedStream else { throw ProcessorError.conversionFailed("adaptive RIFE stream unavailable") }
        for band in 0..<bandCount {
            try Task.checkCancellation()
            let start = band * coreHeight
            guard start < height else { break }
            let count = min(coreHeight, height - start)
            try autoreleasepool {
                let previousTile = try extractBand(previousFrame, coreStart: start)
                let seedOutputs = try stream.push(previousTile, timesteps: [])
                guard seedOutputs.isEmpty else { throw ProcessorError.conversionFailed("adaptive RIFE rebase unexpectedly produced output") }
                let currentTile = try extractBand(current, coreStart: start)
                let generated = try stream.push(currentTile, timesteps: timesteps)
                guard generated.count == outputs.count else { throw ProcessorError.conversionFailed("adaptive RIFE output count mismatch") }
                for index in generated.indices { try copyCore(from: generated[index], to: outputs[index], coreStart: start, coreCount: count) }
            }
        }
        self.previousFrame = current
        return outputs
    }

    private func adaptAtFrameBoundary() throws {
        guard let sharedInterpolator else { return }
        let state = currentRenderPerformanceSnapshot()
        let requested = RIFEBandPlan.bands(width: width, height: height, availableMB: state.availableMemoryMB, underPressure: state.tier == .safe)
        // Shrink only. Growing again would churn graphs while memory fluctuates.
        if requested > bandCount || state.availableMemoryMB < 700 {
            sharedStream = nil
            // No live streams remain and synchronous push has completed its GPU
            // work. Release the old graph BEFORE allocating a smaller one.
            sharedInterpolator.releaseIdleStreamGraph()
            if requested > bandCount {
                bandCount = requested
                coreHeight = Int(ceil(Double(height) / Double(bandCount)))
                tileHeight = coreHeight + overlap * 2
            }
            DiagnosticsLogger.shared.log("Adaptive RIFE reclaimed idle graph • bands=\(bandCount) • tile=\(width)x\(tileHeight) • headroom=\(Int(currentRenderPerformanceSnapshot().availableMemoryMB)) MB")
        }
        guard currentRenderPerformanceSnapshot().availableMemoryMB >= 700 else {
            throw ProcessorError.conversionFailed("Adaptive RIFE stopped at the memory governor's 700 MB emergency floor after releasing idle graph resources. Completed checkpoints are retained.")
        }
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
