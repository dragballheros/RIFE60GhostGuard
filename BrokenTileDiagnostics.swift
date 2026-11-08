import Foundation
import AVFoundation
import CoreImage
import CoreVideo
import ImageIO
import UIKit

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

    private init() {}

    func beginRun(enabled: Bool) {
        lock.lock()
        defer { lock.unlock() }
        self.enabled = enabled
        guard enabled else { return }
        runID = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
        candidates.removeAll()
        sampleCounts.removeAll()
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
    func inspect(_ pixelBuffer: CVPixelBuffer, stage: String, frame: Int, seconds: Double) {
        lock.lock()
        let active = enabled
        lock.unlock()
        guard active, frame % 15 == 0 else { return }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let isTenBit = format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            || format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        let score = base.flatMap { seamScore(base: $0, width: width, height: height, rowBytes: rowBytes, tenBit: isTenBit) }
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
            pixelBuffer: pixelBuffer
        )
    }

    /// Scans completed stage movies at 4 Hz and retains the five strongest tile-grid candidates.
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
        lock.unlock()
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

    private func consider(stage: String, frame: Int, seconds: Double, score: Double, ratio: Double, meanSeam: Double, grid: Int, pixelBuffer: CVPixelBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard enabled, let outputDirectory else { return }
        sampleCounts[stage, default: 0] += 1
        let existing = candidates[stage] ?? []
        if existing.count >= maximumSnapshotsPerStage,
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
        DiagnosticsLogger.shared.logSync(String(
            format: "TILE SAMPLE • stage=%@ • frame=%d • t=%.3fs • grid=%dpx • seamRatio=%.3f • seamMean=%.2f • score=%.3f • retained=%d",
            stage, frame, seconds, grid, ratio, meanSeam, score, updated.count
        ))
    }

    private func seamScore(base: UnsafeMutableRawPointer, width: Int, height: Int, rowBytes: Int, tenBit: Bool) -> (score: Double, ratio: Double, meanSeam: Double, grid: Int)? {
        guard width > 256, height > 256 else { return nil }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let words = base.assumingMemoryBound(to: UInt16.self)
        func value(_ x: Int, _ y: Int) -> Double {
            let offset = y * rowBytes + x * (tenBit ? 2 : 1)
            if tenBit { return Double(words[offset / 2] >> 6) / 4.0 }
            return Double(bytes[offset])
        }
        var best: (score: Double, ratio: Double, meanSeam: Double, grid: Int)?
        for grid in [32, 64, 128, 256] where grid < min(width, height) / 2 {
            var seamTotal = 0.0
            var baseTotal = 0.0
            var seamCount = 0
            var baseCount = 0
            for x in stride(from: grid, to: width - 1, by: grid) {
                for y in stride(from: 4, to: height - 4, by: 8) {
                    seamTotal += abs(value(x, y) - value(x - 1, y))
                    baseTotal += (abs(value(x - 3, y) - value(x - 4, y)) + abs(value(x + 3, y) - value(x + 2, y))) * 0.5
                    seamCount += 1
                    baseCount += 1
                }
            }
            for y in stride(from: grid, to: height - 1, by: grid) {
                for x in stride(from: 4, to: width - 4, by: 8) {
                    seamTotal += abs(value(x, y) - value(x, y - 1))
                    baseTotal += (abs(value(x, y - 3) - value(x, y - 4)) + abs(value(x, y + 3) - value(x, y + 2))) * 0.5
                    seamCount += 1
                    baseCount += 1
                }
            }
            guard seamCount > 0, baseCount > 0 else { continue }
            let seam = seamTotal / Double(seamCount)
            let baseline = baseTotal / Double(baseCount)
            let ratio = seam / max(baseline, 1.0)
            let score = ratio * log1p(seam)
            if best == nil || score > best!.score {
                best = (score, ratio, seam, grid)
            }
        }
        return best
    }
}
