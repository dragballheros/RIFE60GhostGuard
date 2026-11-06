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
        var reused = 0
        var seconds = 0.0
        var previousSource: CVPixelBuffer?
        var previousRestored: CVPixelBuffer?
        var active = model
        var triedSPAN = variant == .span
        var passthrough = false
        let ahead = FrameAheadReader(output: output, depth: 2)
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
                        try Self.restoreFrame(model: active, source: item.buffer, strength: strength, pool: blendPool, deadline: CFAbsoluteTimeGetCurrent() + 15, previousSource: &previousSource, previousRestored: &previousRestored, reused: &reused) { done, total in
                            progress(min(0.99, Double(frames) / 300), "Anime Restore • \(frames) source frames • tile \(done)/\(total)")
                        }
                    }
                } catch let stall as AnimeRestoreStall {
                    let line = "\(stall) • in-flight tile stalled • source frames=\(frames)"
                    DiagnosticsLogger.shared.log(line)
                    CrashCapture.step(line)
                    previousSource = nil
                    previousRestored = nil
                    if !triedSPAN {
                        triedSPAN = true
                        DiagnosticsLogger.shared.log("Anime Restore falling back to SPAN")
                        CrashCapture.step("Anime Restore falling back to SPAN")
                        active = try AnimeRestoreTiledModel(variant: .span)
                        AnimeRestoreRegistry.displayName = AnimeRestoreVariant.span.title
                        do {
                            frame = try Self.restoreFrame(model: active, source: item.buffer, strength: strength, pool: blendPool, deadline: CFAbsoluteTimeGetCurrent() + 15, previousSource: &previousSource, previousRestored: &previousRestored, reused: &reused, onTile: nil)
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
        let wall = String(format: "Stage wall • animeRestore=%.0fms • frames=%d • reused=%d • pass=%.1fs • model=%@", ms, frames, reused, CFAbsoluteTimeGetCurrent() - started, AnimeRestoreRegistry.displayName)
        DiagnosticsLogger.shared.log(wall)
        CrashCapture.step(wall)
        return AnimeRestorePassResult(url: url, frames: frames, millisecondsPerFrame: ms)
    }

    private static func firstFrame(sourceURL: URL) async throws -> CVPixelBuffer {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ProcessorError.missingVideoTrack }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
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

    /// Exact and near-duplicate anime frames reuse the restored picture.
    /// Static blocks keep the previous restoration. A scene cut drops that hold.
    private static func restoreFrame(model: AnimeRestoreTiledModel, source: CVPixelBuffer, strength: Double, pool: CVPixelBufferPool?, deadline: CFAbsoluteTime, previousSource: inout CVPixelBuffer?, previousRestored: inout CVPixelBuffer?, reused: inout Int, onTile: ((Int, Int) -> Void)?) throws -> CVPixelBuffer {
        if let priorSource = previousSource, let priorRestored = previousRestored {
            let difference = blockDifference(priorSource, source)
            if difference.mean >= 0.20 {
                previousSource = nil
                previousRestored = nil
            } else if difference.mean <= 0.012 {
                reused += 1
                return try apply(model: model, source: source, restored: priorRestored, strength: strength, pool: pool, hold: nil)
            } else {
                let restored = try model.restore(source, deadline: deadline, onTile: onTile)
                let held = try apply(model: model, source: source, restored: restored, strength: strength, pool: pool, hold: (priorRestored, difference.quiet))
                previousSource = source
                previousRestored = held
                return held
            }
        }
        let restored = try model.restore(source, deadline: deadline, onTile: onTile)
        let blended = try apply(model: model, source: source, restored: restored, strength: strength, pool: pool, hold: nil)
        previousSource = source
        previousRestored = blended
        return blended
    }

    private static func blockDifference(_ previous: CVPixelBuffer, _ current: CVPixelBuffer) -> (mean: Float, quiet: [Bool]) {
        let blocksX = 16
        let blocksY = 16
        var quiet = [Bool](repeating: false, count: blocksX * blocksY)
        var total: Float = 0
        var count: Float = 0
        CVPixelBufferLockBaseAddress(previous, .readOnly)
        CVPixelBufferLockBaseAddress(current, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(previous, .readOnly)
            CVPixelBufferUnlockBaseAddress(current, .readOnly)
        }
        let width = min(CVPixelBufferGetWidth(previous), CVPixelBufferGetWidth(current))
        let height = min(CVPixelBufferGetHeight(previous), CVPixelBufferGetHeight(current))
        guard let aBase = CVPixelBufferGetBaseAddress(previous), let bBase = CVPixelBufferGetBaseAddress(current), width > 0, height > 0 else {
            return (1, quiet)
        }
        let aStride = CVPixelBufferGetBytesPerRow(previous)
        let bStride = CVPixelBufferGetBytesPerRow(current)
        for by in 0..<blocksY {
            for bx in 0..<blocksX {
                let x0 = bx * width / blocksX
                let x1 = (bx + 1) * width / blocksX
                let y0 = by * height / blocksY
                let y1 = (by + 1) * height / blocksY
                var sum: Float = 0
                var samples: Float = 0
                var y = y0
                while y < y1 {
                    let a = aBase.advanced(by: y * aStride).assumingMemoryBound(to: UInt8.self)
                    let b = bBase.advanced(by: y * bStride).assumingMemoryBound(to: UInt8.self)
                    var x = x0
                    while x < x1 {
                        let lumaA = 0.114 * Float(a[x * 4]) + 0.587 * Float(a[x * 4 + 1]) + 0.299 * Float(a[x * 4 + 2])
                        let lumaB = 0.114 * Float(b[x * 4]) + 0.587 * Float(b[x * 4 + 1]) + 0.299 * Float(b[x * 4 + 2])
                        sum += abs(lumaA - lumaB) / 255
                        samples += 1
                        x += 4
                    }
                    y += 4
                }
                let mean = samples > 0 ? sum / samples : 1
                quiet[by * blocksX + bx] = mean <= 0.01
                total += mean
                count += 1
            }
        }
        return (count > 0 ? total / count : 1, quiet)
    }

    private static func apply(model: any AnimeRestoreModel, source: CVPixelBuffer, restored: CVPixelBuffer, strength: Double, pool: CVPixelBufferPool?, hold: (CVPixelBuffer, [Bool])?) throws -> CVPixelBuffer {
        _ = model
        guard let pool else { return restored }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess, let blended = buffer else {
            throw ProcessorError.conversionFailed("Anime Restore blend pool is empty")
        }
        try blend(source: source, restored: restored, into: blended, strength: Float(strength), hold: hold)
        return blended
    }

    private static func blend(source: CVPixelBuffer, restored: CVPixelBuffer, into destination: CVPixelBuffer, strength: Float, hold: (CVPixelBuffer, [Bool])?) throws {
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
        var holdBase: UnsafeMutableRawPointer?
        var holdStride = 0
        if let hold {
            CVPixelBufferLockBaseAddress(hold.0, .readOnly)
            holdBase = CVPixelBufferGetBaseAddress(hold.0)
            holdStride = CVPixelBufferGetBytesPerRow(hold.0)
        }
        defer { if hold != nil { CVPixelBufferUnlockBaseAddress(hold!.0, .readOnly) } }
        for y in 0..<height {
            let s = sourceBase.advanced(by: y * sourceStride).assumingMemoryBound(to: UInt8.self)
            let r = restoredBase.advanced(by: y * restoredStride).assumingMemoryBound(to: UInt8.self)
            let d = destBase.advanced(by: y * destStride).assumingMemoryBound(to: UInt8.self)
            let held = holdBase?.advanced(by: y * holdStride).assumingMemoryBound(to: UInt8.self)
            let blockY = min(15, y * 16 / height)
            for x in 0..<width {
                let blockX = min(15, x * 16 / width)
                let quiet = hold?.1[blockY * 16 + blockX] == true
                let pixel = x * 4
                if quiet, let held {
                    d[pixel] = held[pixel]
                    d[pixel + 1] = held[pixel + 1]
                    d[pixel + 2] = held[pixel + 2]
                    d[pixel + 3] = held[pixel + 3]
                } else {
                    for channel in 0..<4 {
                        d[pixel + channel] = UInt8(min(255, max(0, keep * Float(s[pixel + channel]) + strength * Float(r[pixel + channel]) + 0.5)))
                    }
                }
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
