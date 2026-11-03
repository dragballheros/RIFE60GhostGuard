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

    init(strength: Double) {
        self.strength = min(max(strength, 0.1), 1)
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
        let duration = try await asset.load(.duration)
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())
        guard let factory = AnimeRestoreRegistry.factory else {
            throw ProcessorError.conversionFailed("Anime Restore model is not registered")
        }
        let model = try factory()
        defer { model.release() }

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
        if checkpoint.resumeTime > .zero {
            reader.timeRange = CMTimeRange(start: checkpoint.resumeTime, duration: CMTimeMaximum(.zero, CMTimeSubtract(duration, checkpoint.resumeTime)))
            DiagnosticsLogger.shared.log("Anime Restore incremental resume • from \(String(format: "%.3f", CMTimeGetSeconds(checkpoint.resumeTime)))s")
        }
        guard reader.startReading() else {
            throw ProcessorError.reader(reader.error?.localizedDescription ?? "anime restore reader failed")
        }

        let blendPool: CVPixelBufferPool? = strength < 0.999 ? try Self.makePool(width: width, height: height) : nil
        if let blendPool {
            ReclaimableMemoryRegistry.shared.trackPixelPool(blendPool, name: "anime-restore-blend", width: width, height: height, slots: 2)
        }
        defer { ReclaimableMemoryRegistry.shared.remove(name: "anime-restore-blend") }

        var frames = 0
        var seconds = 0.0
        let ahead = FrameAheadReader(output: output, depth: 2)
        let started = CFAbsoluteTimeGetCurrent()
        while let item = ahead.next() {
            try await checkpoint.checkCancellation()
            try await adaptiveFrameBoundaryPacing()
            if CMTimeCompare(item.time, checkpoint.resumeTime) <= 0 { continue }
            let frame: CVPixelBuffer
            let began = CFAbsoluteTimeGetCurrent()
            do {
                frame = try autoreleasepool { try Self.apply(model: model, source: item.buffer, strength: strength, pool: blendPool) }
            } catch {
                let line = "Anime Restore failed • \(error.localizedDescription)"
                DiagnosticsLogger.shared.log(line)
                CrashCapture.step(line)
                throw error
            }
            seconds += CFAbsoluteTimeGetCurrent() - began
            try await checkpoint.append(frame, at: item.time)
            frames += 1
            if frames == 1 || frames % 30 == 0 {
                progress(min(0.99, Double(frames) / 300), "Anime Restore • \(frames) frames")
            }
        }
        let url = try await checkpoint.finish()
        let ms = frames > 0 ? seconds * 1000 / Double(frames) : 0
        PipelineStageTimer.current?.record("animeRestoreMs", milliseconds: ms)
        let wall = String(format: "Stage wall • animeRestore=%.0fms • frames=%d • pass=%.1fs • model=%@", ms, frames, CFAbsoluteTimeGetCurrent() - started, AnimeRestoreRegistry.displayName)
        DiagnosticsLogger.shared.log(wall)
        CrashCapture.step(wall)
        return AnimeRestorePassResult(url: url, frames: frames, millisecondsPerFrame: ms)
    }

    private static func apply(model: any AnimeRestoreModel, source: CVPixelBuffer, strength: Double, pool: CVPixelBufferPool?) throws -> CVPixelBuffer {
        let restored = try model.restore(source)
        guard strength < 0.999, let pool else { return restored }
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
            let s = sourceBase.advanced(by: y * sourceStride).assumingMemoryBound(to: UInt8.self)
            let r = restoredBase.advanced(by: y * restoredStride).assumingMemoryBound(to: UInt8.self)
            let d = destBase.advanced(by: y * destStride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<(width * 4) {
                d[x] = UInt8(min(255, max(0, keep * Float(s[x]) + strength * Float(r[x]) + 0.5)))
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
