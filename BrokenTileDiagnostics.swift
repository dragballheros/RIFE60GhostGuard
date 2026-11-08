import Foundation
import AVFoundation
import CoreImage
import CoreVideo
import ImageIO
import UIKit

/// A retained defect snapshot produced only when the live detector sees a frame that is materially more suspicious than the stage baseline.
struct LiveDiagnosticDefect: Identifiable, Sendable {
    let id: UUID
    let stage: String
    let frame: Int
    let seconds: Double
    let grid: Int
    let ratio: Double
    let seam: Double
    let score: Double
    let imageURL: URL
}

/// Opt-in, read-only diagnostics that compare the same processing stages without changing pixels.
final class BrokenTileDiagnostics: @unchecked Sendable {
    static let shared = BrokenTileDiagnostics()

    private struct Candidate {
        let score: Double
        let ratio: Double
        let meanSeam: Double
        let frame: Int
        let seconds: Double
        let grid: Int
        let url: URL
    }

    private let lock = NSLock()
    private let context = CIContext(options: [.useSoftwareRenderer: false])
    private let maximumSnapshotsPerStage = 5
    private var enabled = false
    private var runID = ""
    private var candidates: [String: [Candidate]] = [:]
    private var sampleCounts: [String: Int] = [:]
    private var outputDirectory: URL?
    private var liveDefectHandler: ((LiveDiagnosticDefect) -> Void)?
    private var recentScores: [String: [Double]] = [:]

    private init() {}

    func setLiveDefectHandler(_ handler: ((LiveDiagnosticDefect) -> Void)?) {
        lock.lock()
        liveDefectHandler = handler
        lock.unlock()
    }

    func beginRun(enabled: Bool) {
        lock.lock()
        defer { lock.unlock() }
        self.enabled = enabled
        guard enabled else { return }
        runID = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
        candidates.removeAll()
        sampleCounts.removeAll()
        recentScores.removeAll()
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Diagnostics", isDirectory: true)
        let folder = root.appendingPathComponent("TileFrames-\(runID)", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        outputDirectory = folder
        DiagnosticsLogger.shared.logSync("BROKEN-TILE DIAGNOSTICS START • run=\(runID) • sampling every 15 frames • top 5 seam candidates per stage")
    }

    var isEnabled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return enabled
    }

    /// Called on decoded/intermediate pixel buffers. Samples at 4 Hz for 60 FPS video.
    func inspect(_ pixelBuffer: CVPixelBuffer, stage: String, frame: Int, seconds: Double, force: Bool = false) {
        lock.lock()
        let active = enabled
        lock.unlock()
        guard active, force || frame % 15 == 0 else { return }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let planar = CVPixelBufferGetPlaneCount(pixelBuffer) > 0
        let base = planar ? CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) : CVPixelBufferGetBaseAddress(pixelBuffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(pixelBuffer, 0) : CVPixelBufferGetWidth(pixelBuffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(pixelBuffer, 0) : CVPixelBufferGetHeight(pixelBuffer)
        let rowBytes = planar ? CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0) : CVPixelBufferGetBytesPerRow(pixelBuffer)
        let isTenBit = format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            || format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        let isBGRA = format == kCVPixelFormatType_32BGRA
        let score = base.flatMap { seamScore(base: $0, width: width, height: height, rowBytes: rowBytes, tenBit: isTenBit, bgra: isBGRA) }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        guard let score else { return }
        consider(
            stage: stage,
            frame: frame,
            seconds: seconds,
            score: score.score,
            ratio: score.ratio,
            meanSeam: score.meanSeam,
            grid: score.grid,
            pixelBuffer: pixelBuffer,
            force: force
        )
    }

    /// Scans completed stage movies at 4 Hz and retains the five strongest tile-grid candidates.
    /// Compares visible pixel bytes before and after the app-owned snapshot copy.
    /// This isolates copy/stride corruption from inference output corruption.
    func comparePixelBuffers(_ lhs: CVPixelBuffer,
                             _ rhs: CVPixelBuffer,
                             frame: Int,
                             seconds: Double) {
        lock.lock()
        let active = enabled
        lock.unlock()
        guard active, frame % 15 == 0 else { return }

        let width = CVPixelBufferGetWidth(lhs)
        let height = CVPixelBufferGetHeight(lhs)
        guard width == CVPixelBufferGetWidth(rhs),
              height == CVPixelBufferGetHeight(rhs),
              CVPixelBufferGetPixelFormatType(lhs) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(rhs) == kCVPixelFormatType_32BGRA else {
            DiagnosticsLogger.shared.logSync(
                "RIFE_COPY_PARITY frame=\\(frame) t=\\(String(format: "%.3f", seconds)) comparable=false"
            )
            return
        }

        CVPixelBufferLockBaseAddress(lhs, .readOnly)
        CVPixelBufferLockBaseAddress(rhs, .readOnly)
        guard let lhsBase = CVPixelBufferGetBaseAddress(lhs),
              let rhsBase = CVPixelBufferGetBaseAddress(rhs) else {
            CVPixelBufferUnlockBaseAddress(rhs, .readOnly)
            CVPixelBufferUnlockBaseAddress(lhs, .readOnly)
            DiagnosticsLogger.shared.logSync("RIFE_COPY_PARITY frame=\\(frame) baseAddress=unavailable")
            return
        }

        let lhsStride = CVPixelBufferGetBytesPerRow(lhs)
        let rhsStride = CVPixelBufferGetBytesPerRow(rhs)
        var mismatchedPixels = 0
        var maxDelta = 0
        for y in 0..<height {
            let a = lhsBase.advanced(by: y * lhsStride).assumingMemoryBound(to: UInt8.self)
            let b = rhsBase.advanced(by: y * rhsStride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let offset = x * 4
                var pixelDiffers = false
                for channel in 0..<4 {
                    let delta = abs(Int(a[offset + channel]) - Int(b[offset + channel]))
                    if delta != 0 { pixelDiffers = true }
                    maxDelta = max(maxDelta, delta)
                }
                if pixelDiffers { mismatchedPixels += 1 }
            }
        }
        CVPixelBufferUnlockBaseAddress(rhs, .readOnly)
        CVPixelBufferUnlockBaseAddress(lhs, .readOnly)

        DiagnosticsLogger.shared.logSync(
            "RIFE_COPY_PARITY frame=\\(frame) t=\\(String(format: "%.3f", seconds)) " +
            "mismatchedPixels=\\(mismatchedPixels)/\\(width * height) maxChannelDelta=\\(maxDelta) " +
            "sourceStride=\\(lhsStride) destinationStride=\\(rhsStride)"
        )
    }

    func scanVideo(url: URL, stage: String) async {
        guard isEnabled else { return }
        do {
            let asset = AVURLAsset(url: url)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let track = tracks.first else {
                DiagnosticsLogger.shared.log("Tile diagnostics • \(stage) • no video track")
                return
            }
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
            )
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else { return }
            reader.add(output)
            guard reader.startReading() else { throw reader.error ?? NSError(domain: "TileDiagnostics", code: 1) }
            var frame = 0
            while let sample = output.copyNextSampleBuffer() {
                if frame % 15 == 0, let pixelBuffer = CMSampleBufferGetImageBuffer(sample) {
                    let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                    inspect(pixelBuffer, stage: stage, frame: frame, seconds: CMTimeGetSeconds(pts))
                }
                frame += 1
            }
            if reader.status == .failed { throw reader.error ?? NSError(domain: "TileDiagnostics", code: 2) }
            DiagnosticsLogger.shared.log("Tile diagnostics scan complete • stage=\(stage) • decodedFrames=\(frame) • sampled=\((frame + 14) / 15)")
        } catch {
            DiagnosticsLogger.shared.log("Tile diagnostics scan failed • stage=\(stage) • \(error.localizedDescription)")
        }
    }

    func finishRun() {
        lock.lock()
        guard enabled else { lock.unlock(); return }
        let report = sampleCounts.keys.sorted().map { stage -> String in
            let values = candidates[stage] ?? []
            let strongest = values.max(by: { $0.score < $1.score })
            if let strongest {
                return String(format: "%@ samples=%d retained=%d strongestFrame=%d t=%.3fs grid=%d ratio=%.3f seam=%.2f",
                              stage, sampleCounts[stage] ?? 0, values.count, strongest.frame,
                              strongest.seconds, strongest.grid, strongest.ratio, strongest.meanSeam)
            }
            return "\(stage) samples=\(sampleCounts[stage] ?? 0) retained=0"
        }.joined(separator: " | ")
        enabled = false
        // Keep the UI callback installed across runs. The view model owns the lifecycle\n        // of the callback, while diagnostics only owns whether detection is enabled.\n        lock.unlock()
        let detail = report.isEmpty ? "no pixel-buffer samples" : report
        DiagnosticsLogger.shared.logSync("BROKEN-TILE DIAGNOSTICS COMPLETE • run=\(runID) • \(detail)")
    }

    func exportFiles() -> [URL] {
        lock.lock()
        defer { lock.unlock() }
        guard let outputDirectory,
              let urls = try? FileManager.default.contentsOfDirectory(at: outputDirectory, includingPropertiesForKeys: [.contentModificationDateKey]) else { return [] }
        return urls.filter { $0.pathExtension.lowercased() == "jpg" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func consider(stage: String, frame: Int, seconds: Double, score: Double, ratio: Double, meanSeam: Double, grid: Int, pixelBuffer: CVPixelBuffer, force: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard enabled, let outputDirectory else { return }
        let existing = candidates[stage] ?? []
        if existing.contains(where: { $0.frame == frame }) { return }
        sampleCounts[stage, default: 0] += 1

        // Do not turn the carousel into a slideshow of normal frames. A live defect
        // must either be explicitly forced by a guard, or be a strong outlier against
        // the recent score history for this stage. The first few samples establish a
        // baseline without writing screenshots.
        let history = recentScores[stage] ?? []
        let baseline = robustBaseline(history)
        let adaptiveThreshold = baseline.map { max(2.25, $0 + 4.0 * robustMAD(history, around: $0)) } ?? 2.25
        let isDefect = force || score >= adaptiveThreshold
        recentScores[stage] = Array((history + [score]).suffix(24))
        guard isDefect else { return }

        if !force,
           existing.count >= maximumSnapshotsPerStage,
           let weakest = existing.min(by: { $0.score < $1.score }),
           score <= weakest.score { return }

        guard let image = context.createCGImage(CIImage(cvPixelBuffer: pixelBuffer), from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))) else { return }
        let safeStage = stage.replacingOccurrences(of: "[^A-Za-z0-9-]", with: "-", options: .regularExpression)
        let name = String(format: "%@-f%06d-t%07.2fs-r%.2f.jpg", safeStage, frame, seconds, ratio)
        let destination = outputDirectory.appendingPathComponent(name)
        let data = NSMutableData()
        guard let writer = CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(writer, image, [kCGImageDestinationLossyCompressionQuality: 0.62] as CFDictionary)
        guard CGImageDestinationFinalize(writer) else { return }
        do { try (data as Data).write(to: destination, options: .atomic) } catch { return }

        var updated = existing
        if updated.count >= maximumSnapshotsPerStage,
           let weakestIndex = updated.indices.min(by: { updated[$0].score < updated[$1].score }) {
            try? FileManager.default.removeItem(at: updated[weakestIndex].url)
            updated.remove(at: weakestIndex)
        }
        updated.append(Candidate(score: score, ratio: ratio, meanSeam: meanSeam, frame: frame, seconds: seconds, grid: grid, url: destination))
        candidates[stage] = updated
        let live = LiveDiagnosticDefect(
            id: UUID(), stage: stage, frame: frame, seconds: seconds,
            grid: grid, ratio: ratio, seam: meanSeam, score: score, imageURL: destination
        )
        let handler = liveDefectHandler
        DiagnosticsLogger.shared.logSync(String(
            format: "TILE SAMPLE • stage=%@ • frame=%d • t=%.3fs • grid=%dpx • seamRatio=%.3f • seamMean=%.2f • score=%.3f • retained=%d",
            stage, frame, seconds, grid, ratio, meanSeam, score, updated.count
        ))
        if let handler {
            Task { @MainActor in handler(live) }
        }
    }

    private func robustBaseline(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private func robustMAD(_ values: [Double], around center: Double) -> Double {
        guard !values.isEmpty else { return 0.0 }
        let deviations = values.map { abs($0 - center) }.sorted()
        return deviations[deviations.count / 2]
    }

    private func seamScore(base: UnsafeMutableRawPointer, width: Int, height: Int, rowBytes: Int, tenBit: Bool, bgra: Bool) -> (score: Double, ratio: Double, meanSeam: Double, grid: Int)? {
        guard width > 256, height > 256 else { return nil }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let words = base.assumingMemoryBound(to: UInt16.self)
        func value(_ x: Int, _ y: Int) -> Double {
            if bgra {
                let offset = y * rowBytes + x * 4
                let blue = Double(bytes[offset])
                let green = Double(bytes[offset + 1])
                let red = Double(bytes[offset + 2])
                return 0.114 * blue + 0.587 * green + 0.299 * red
            }
            let offset = y * rowBytes + x * (tenBit ? 2 : 1)
            if tenBit { return Double(words[offset / 2] >> 6) / 4.0 }
            return Double(bytes[offset])
        }

        var best: (score: Double, ratio: Double, meanSeam: Double, grid: Int)?
        func updateBest(seamTotal: Double, baselineTotal: Double, count: Int, grid: Int) {
            guard count > 0 else { return }
            let seam = seamTotal / Double(count)
            let baseline = baselineTotal / Double(count)
            let ratio = seam / max(baseline, 1.0)
            let score = ratio * log1p(seam)
            if best == nil || score > best!.score {
                best = (score, ratio, seam, grid)
            }
        }

        // Score short local segments, not a whole-frame average. A single damaged
        // tile should remain visible in the metric instead of being diluted by the
        // many unaffected grid boundaries elsewhere in the frame.
        for grid in [16, 32, 64, 128, 256] where grid < min(width, height) / 2 {
            for x in stride(from: grid, to: width - 4, by: grid) {
                for segmentY in stride(from: 4, to: height - 4, by: 64) {
                    let endY = min(segmentY + 64, height - 4)
                    var seam = 0.0
                    var baseline = 0.0
                    var count = 0
                    for y in stride(from: segmentY, to: endY, by: 2) {
                        seam += abs(value(x, y) - value(x - 1, y))
                        baseline += (abs(value(x - 3, y) - value(x - 4, y)) + abs(value(x + 3, y) - value(x + 2, y))) * 0.5
                        count += 1
                    }
                    updateBest(seamTotal: seam, baselineTotal: baseline, count: count, grid: grid)
                }
            }
            for y in stride(from: grid, to: height - 4, by: grid) {
                for segmentX in stride(from: 4, to: width - 4, by: 64) {
                    let endX = min(segmentX + 64, width - 4)
                    var seam = 0.0
                    var baseline = 0.0
                    var count = 0
                    for x in stride(from: segmentX, to: endX, by: 2) {
                        seam += abs(value(x, y) - value(x, y - 1))
                        baseline += (abs(value(x, y - 3) - value(x, y - 4)) + abs(value(x, y + 3) - value(x, y + 2))) * 0.5
                        count += 1
                    }
                    updateBest(seamTotal: seam, baselineTotal: baseline, count: count, grid: grid)
                }
            }
        }
        return best
    }}
