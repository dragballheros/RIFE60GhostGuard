import AVFoundation
import CoreVideo
import Foundation

struct AnimeRestorePassResult: Sendable {
    let url: URL
    let frames: Int
    let millisecondsPerFrame: Double
}

/// First pipeline stage. Runs only when a model is registered.
/// Output size matches the source. Checkpoint stage id is `anime-restore`.
final class AnimeRestorePass {
    private let strength: Double
    private let quality: AnimeRestoreQuality

    init(strength: Double, quality: AnimeRestoreQuality) {
        self.strength = min(max(strength, 0.1), 1)
        self.quality = quality
    }

    func run(
        sourceURL: URL,
        progress: @escaping @Sendable (Double, String) -> Void,
        recoveryDirectory: URL? = nil
    ) async throws -> AnimeRestorePassResult {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ProcessorError.missingVideoTrack
        }
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())
        guard AnimeRestoreRegistry.isRegistered else {
            throw ProcessorError.conversionFailed("Anime Restore model is not registered")
        }
        let probeFrame = try await Self.firstFrame(sourceURL: sourceURL)
        let variant = try AnimeRestoreProbe.resolve(quality: quality, width: width, height: height, frame: probeFrame)
        let model = try AnimeRestoreTiledModel(variant: variant)
        defer { model.release() }
        AnimeRestoreRegistry.displayName = variant.title

        let snapshot = currentRenderPerformanceSnapshot()
        DiagnosticsLogger.shared.log("Anime Restore start • \(width)x\(height) • strength=\(String(format: "%.2f", strength)) • model=\(AnimeRestoreRegistry.displayName) • \(snapshot.thermalAndMode) • raw=\(Int(snapshot.availableMemoryMB)) MB • effective=\(Int(snapshot.effectiveMemoryMB)) MB")
        CrashCapture.step("Anime Restore start • \(width)x\(height) • \(AnimeRestoreRegistry.displayName)")

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot attach anime restore output") }
        reader.add(output)

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.proRes422HQ,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ]
        let checkpoint = try IncrementalVideoCheckpointWriter(
            recoveryDirectory: recoveryDirectory,
            stageID: "anime-restore",
            outputSettings: outputSettings,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ],
            transform: transform,
            segmentDuration: 5.0
        )
        guard reader.startReading() else {
            throw ProcessorError.reader(reader.error?.localizedDescription ?? "anime restore reader failed")
        }

        let blendPool: CVPixelBufferPool? = try Self.makePool(width: width, height: height)
        if let blendPool {
            ReclaimableMemoryRegistry.shared.trackPixelPool(blendPool, name: "anime-restore-blend", width: width, height: height, slots: 2)
        }
        defer { ReclaimableMemoryRegistry.shared.remove(name: "anime-restore-blend") }

        let tiles = AnimeRestoreChoice.tileCount(length: width) * AnimeRestoreChoice.tileCount(length: height)
        DiagnosticsLogger.shared.log("Anime Restore tiles • \(width)x\(height) • \(tiles) source tiles • each tile is restored at 256 then stitched, with no full 2× frame")
        var frames = 0
        var seconds = 0.0
        var active = model
        var triedSPAN = variant == .span
        var passthrough = false
        let ahead = FrameAheadReader(output: output, depth: 4)
        let started = CFAbsoluteTimeGetCurrent()
        while let item = ahead.next() {
            try await checkpoint.checkCancellation()
            try await adaptiveFrameBoundaryPacing()
            DiagnosticsLogger.shared.log("Anime Restore frame \(frames) • tiles=\(active.tileCount) • last=\(active.lastStep)")
            let began = CFAbsoluteTimeGetCurrent()
            let frame: CVPixelBuffer
            if passthrough {
                frame = try Self.copyFrame(item.buffer)
            } else {
                do {
                    frame = try autoreleasepool {
                        try Self.restoreFrame(model: active, source: item.buffer, strength: strength, pool: blendPool, deadline: CFAbsoluteTimeGetCurrent() + 15) { done, total in
                            progress(min(0.99, Double(frames) / 300), "Anime Restore • \(frames) source frames • tile \(done)/\(total)")
                        }
                    }
                } catch let stall as AnimeRestoreStall {
                    let line = "\(stall) • in-flight tile stalled • source frames=\(frames)"
                    DiagnosticsLogger.shared.log(line)
                    CrashCapture.step(line)
                    if !triedSPAN {
                        triedSPAN = true
                        DiagnosticsLogger.shared.log("Anime Restore falling back to SPAN")
                        CrashCapture.step("Anime Restore falling back to SPAN")
                        active = try AnimeRestoreTiledModel(variant: .span)
                        AnimeRestoreRegistry.displayName = AnimeRestoreVariant.span.title
                        do {
                            frame = try Self.restoreFrame(model: active, source: item.buffer, strength: strength, pool: blendPool, deadline: CFAbsoluteTimeGetCurrent() + 15, onTile: nil)
                        } catch {
                            let skip = "Anime Restore skipped for this render • SPAN also stalled • \(error.localizedDescription)"
                            DiagnosticsLogger.shared.log(skip)
                            CrashCapture.step(skip)
                            passthrough = true
                            frame = try Self.copyFrame(item.buffer)
                        }
                    } else {
                        passthrough = true
                        let skip = "Anime Restore skipped for this render • SPAN also stalled • frames unchanged from here"
                        DiagnosticsLogger.shared.log(skip)
                        CrashCapture.step(skip)
                        frame = try Self.copyFrame(item.buffer)
                    }
                } catch {
                    let line = "Anime Restore failed • \(error.localizedDescription)"
                    DiagnosticsLogger.shared.log(line)
                    CrashCapture.step(line)
                    throw error
                }
            }
            seconds += CFAbsoluteTimeGetCurrent() - began
            try await checkpoint.append(frame, at: item.time)
            frames += 1
            progress(min(0.99, Double(frames) / 300), "Anime Restore • \(frames) source frames")
        }
        let url = try await checkpoint.finish()
        let ms = frames > 0 ? seconds * 1000 / Double(frames) : 0
        PipelineStageTimer.current?.record("animeRestoreMs", milliseconds: ms)
        let wall = String(format: "Stage wall • animeRestore=%.0fms • frames=%d • temporalReuse=off • pass=%.1fs • model=%@", ms, frames, CFAbsoluteTimeGetCurrent() - started, AnimeRestoreRegistry.displayName)
        DiagnosticsLogger.shared.log(wall)
        CrashCapture.step(wall)
        return AnimeRestorePassResult(url: url, frames: frames, millisecondsPerFrame: ms)
    }

    private static func firstFrame(sourceURL: URL) async throws -> CVPixelBuffer {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ProcessorError.missingVideoTrack }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = true
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot read an Anime Restore probe frame") }
        reader.add(output)
        guard reader.startReading(), let sample = output.copyNextSampleBuffer(), let buffer = CMSampleBufferGetImageBuffer(sample) else {
            throw ProcessorError.reader("Anime Restore probe frame is missing")
        }
        return try copyFrame(buffer)
    }

    private static func copyFrame(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        var copy: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &copy) == kCVReturnSuccess, let copy else {
            throw ProcessorError.conversionFailed("Anime Restore probe copy failed")
        }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(copy, [])
        if let from = CVPixelBufferGetBaseAddress(source), let to = CVPixelBufferGetBaseAddress(copy) {
            let rows = min(CVPixelBufferGetBytesPerRow(source), CVPixelBufferGetBytesPerRow(copy))
            for y in 0..<height {
                to.advanced(by: y * CVPixelBufferGetBytesPerRow(copy)).copyMemory(from: from.advanced(by: y * CVPixelBufferGetBytesPerRow(source)), byteCount: rows)
            }
        }
        CVPixelBufferUnlockBaseAddress(copy, [])
        CVPixelBufferUnlockBaseAddress(source, .readOnly)
        return copy
    }

    /// Every source frame is restored independently. Temporal pixel/block holding is
    /// intentionally prohibited because missed motion creates stale rectangular regions.
    private static func restoreFrame(model: AnimeRestoreTiledModel, source: CVPixelBuffer, strength: Double, pool: CVPixelBufferPool?, deadline: CFAbsoluteTime, onTile: ((Int, Int) -> Void)?) throws -> CVPixelBuffer {
        let restored = try model.restore(source, deadline: deadline, onTile: onTile)
        return try apply(model: model, source: source, restored: restored, strength: strength, pool: pool)
    }

    private static func apply(model: any AnimeRestoreModel, source: CVPixelBuffer, restored: CVPixelBuffer, strength: Double, pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        _ = model
        guard let pool else { return restored }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess, let blended = buffer else {
            throw ProcessorError.conversionFailed("Anime Restore blend pool is empty")
        }
        try blend(source: source, restored: restored, into: blended, strength: Float(strength))
        return blended
    }

    private static func blend(source: CVPixelBuffer, restored: CVPixelBuffer, into destination: CVPixelBuffer, strength: Float) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(restored, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(restored, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }
        let width = min(CVPixelBufferGetWidth(source), CVPixelBufferGetWidth(restored), CVPixelBufferGetWidth(destination))
        let height = min(CVPixelBufferGetHeight(source), CVPixelBufferGetHeight(restored), CVPixelBufferGetHeight(destination))
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let restoredBase = CVPixelBufferGetBaseAddress(restored),
              let destBase = CVPixelBufferGetBaseAddress(destination),
              width > 0, height > 0 else {
            throw ProcessorError.conversionFailed("Anime Restore blend buffers are unreadable")
        }
        let sourceStride = CVPixelBufferGetBytesPerRow(source)
        let restoredStride = CVPixelBufferGetBytesPerRow(restored)
        let destStride = CVPixelBufferGetBytesPerRow(destination)
        let keep = 1 - strength
        for y in 0..<height {
            let sourceRow = sourceBase.advanced(by: y * sourceStride).assumingMemoryBound(to: UInt8.self)
            let restoredRow = restoredBase.advanced(by: y * restoredStride).assumingMemoryBound(to: UInt8.self)
            let destinationRow = destBase.advanced(by: y * destStride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let pixel = x * 4
                for channel in 0..<3 {
                    destinationRow[pixel + channel] = UInt8(min(255, max(0, keep * Float(sourceRow[pixel + channel]) + strength * Float(restoredRow[pixel + channel]) + 0.5)))
                }
                destinationRow[pixel + 3] = 255
            }
        }
    }

    private static func makePool(width: Int, height: Int) throws -> CVPixelBufferPool {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess, let pool else {
            throw ProcessorError.conversionFailed("Anime Restore blend pool could not be created")
        }
        return pool
    }
}
