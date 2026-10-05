import Foundation
import AVFoundation
import VideoToolbox
import CoreVideo

/// Pause-only checkpoint writer.
///
/// Normal rendering uses one AVAssetWriter-backed segment for the entire stage.
/// No manifest is written per frame and no rolling segment rotation occurs.
/// When the user presses Pause, cancellation is intercepted at a safe frame
/// boundary, the current writer is finalized, and only then is a small manifest
/// written. Resume starts a new segment from that saved presentation timestamp.
final class PauseCheckpointCoordinator {
    static let shared = PauseCheckpointCoordinator()
    private let lock = NSLock()
    private var requested = false

    func request() {
        lock.lock(); requested = true; lock.unlock()
    }

    func clear() {
        lock.lock(); requested = false; lock.unlock()
    }

    var isRequested: Bool {
        lock.lock(); defer { lock.unlock() }
        return requested
    }
}

final class IncrementalVideoCheckpointWriter {
    private struct Manifest: Codable {
        let version: Int
        let stageID: String
        var segments: [String]
        var lastPTSSeconds: Double
    }

    let resumeTime: CMTime

    private let fm = FileManager.default
    private let directory: URL
    private let stageID: String
    private let outputSettings: [String: Any]
    private let sourcePixelBufferAttributes: [String: Any]
    private let transform: CGAffineTransform
    private let fileType: AVFileType

    private var manifest: Manifest
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var segmentStartPTS: CMTime?
    private var segmentURL: URL?
    private var lastWrittenPTS: CMTime?
    private var transferSession: VTPixelTransferSession?

    private var manifestURL: URL { directory.appendingPathComponent("manifest.json") }

    init(
        recoveryDirectory: URL?,
        stageID: String,
        outputSettings: [String: Any],
        sourcePixelBufferAttributes: [String: Any],
        transform: CGAffineTransform,
        fileType: AVFileType = .mov,
        segmentDuration: Double = 0
    ) throws {
        self.stageID = stageID
        self.outputSettings = outputSettings
        self.sourcePixelBufferAttributes = sourcePixelBufferAttributes
        self.transform = transform
        self.fileType = fileType

        let root = recoveryDirectory ?? fm.temporaryDirectory
        self.directory = root.appendingPathComponent("incremental-\(stageID)", isDirectory: true)
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        let manifestURL = directory.appendingPathComponent("manifest.json")
        if let data = try? Data(contentsOf: manifestURL),
           let existing = try? JSONDecoder().decode(Manifest.self, from: data),
           existing.version == 1,
           existing.stageID == stageID {
            self.manifest = existing
        } else {
            let initial = Manifest(version: 1, stageID: stageID, segments: [], lastPTSSeconds: -1)
            self.manifest = initial
            if let data = try? JSONEncoder().encode(initial) {
                try? data.write(to: manifestURL, options: .atomic)
            }
        }

        self.resumeTime = self.manifest.lastPTSSeconds >= 0
            ? CMTime(seconds: self.manifest.lastPTSSeconds, preferredTimescale: 600)
            : .zero
    }

    var hasCheckpointedMedia: Bool { !manifest.segments.isEmpty }

    /// Normal path: exactly the same cancellation check the old stages used.
    /// Only the cancellation caused by the Pause button is converted into a
    /// durable checkpoint; Cancel and system/task cancellation remain ordinary
    /// cancellation with no incremental checkpoint save.
    func checkCancellation() async throws {
        do {
            try Task.checkCancellation()
        } catch {
            if PauseCheckpointCoordinator.shared.isRequested {
                try await finishCurrentSegment(persistManifest: true)
            }
            throw error
        }
    }

    func append(_ pixelBuffer: CVPixelBuffer, at presentationTime: CMTime) async throws {
        try await checkCancellation()
        try await ensureWriter(startingAt: presentationTime)
        guard let input, let adaptor else {
            throw ProcessorError.writer("pause checkpoint writer unavailable")
        }

        while !input.isReadyForMoreMediaData {
            try await checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000)
        }

        guard adaptor.append(pixelBuffer, withPresentationTime: localTime(presentationTime)) else {
            throw ProcessorError.writer("failed writing pause checkpoint frame")
        }
        lastWrittenPTS = presentationTime
    }

    func append10Bit(_ source: CVPixelBuffer, at presentationTime: CMTime) async throws {
        try await checkCancellation()
        try await ensureWriter(startingAt: presentationTime)

        guard let input, let adaptor, let pool = adaptor.pixelBufferPool else {
            throw ProcessorError.writer("pause checkpoint 10-bit pool unavailable")
        }

        while !input.isReadyForMoreMediaData {
            try await checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000)
        }

        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess,
              let destination else {
            throw ProcessorError.conversionFailed("could not allocate pause checkpoint 10-bit frame")
        }

        if transferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(
                allocator: kCFAllocatorDefault,
                pixelTransferSessionOut: &session
            ) == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create pause checkpoint pixel transfer session")
            }
            transferSession = session
        }

        guard let transferSession,
              VTPixelTransferSessionTransferImage(
                transferSession,
                from: source,
                to: destination
              ) == noErr else {
            throw ProcessorError.conversionFailed("BGRA→P010 conversion failed for pause checkpoint")
        }

        guard adaptor.append(destination, withPresentationTime: localTime(presentationTime)) else {
            throw ProcessorError.writer("failed writing pause checkpoint 10-bit frame")
        }
        lastWrittenPTS = presentationTime
    }

    /// Finishes the active writer. If this is a normal completion with one segment,
    /// return that file directly so the fast path does not add a composition/export.
    /// If a pause/resume produced multiple segments, concatenate them once at the end.
    func finish() async throws -> URL {
        try await finishCurrentSegment(persistManifest: false)
        guard !manifest.segments.isEmpty else { throw ProcessorError.noOutput }

        if manifest.segments.count == 1,
           let only = manifest.segments.first {
            let url = directory.appendingPathComponent(only)
            guard fm.fileExists(atPath: url.path) else { throw ProcessorError.noOutput }
            return url
        }

        return try await composeSegments()
    }

    func discardCheckpointMedia() {
        try? fm.removeItem(at: directory)
    }

    static func cleanup(recoveryDirectory: URL?, stageID: String) {
        let root = recoveryDirectory ?? FileManager.default.temporaryDirectory
        try? FileManager.default.removeItem(
            at: root.appendingPathComponent("incremental-(stageID)", isDirectory: true)
        )
    }

    private func ensureWriter(startingAt pts: CMTime) async throws {
        guard writer == nil else { return }

        let url = directory.appendingPathComponent(
            String(format: "%05d.mov", manifest.segments.count)
        )
        try? fm.removeItem(at: url)

        let writer = try AVAssetWriter(outputURL: url, fileType: fileType)
        writer.movieFragmentInterval = CMTime(seconds: 2, preferredTimescale: 600)

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        input.expectsMediaDataInRealTime = false
        input.transform = transform
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: sourcePixelBufferAttributes
        )

        guard writer.canAdd(input) else {
            throw ProcessorError.writer("cannot attach pause checkpoint writer input")
        }
        writer.add(input)

        guard writer.startWriting() else {
            throw ProcessorError.writer(writer.error?.localizedDescription ?? "pause checkpoint writer failed")
        }
        writer.startSession(atSourceTime: .zero)

        self.writer = writer
        self.input = input
        self.adaptor = adaptor
        self.segmentStartPTS = pts
        self.segmentURL = url
    }

    private func localTime(_ global: CMTime) -> CMTime {
        guard let start = segmentStartPTS else { return global }
        return CMTimeSubtract(global, start)
    }

    private func finishCurrentSegment(persistManifest: Bool) async throws {
        guard let writer, let input, let url = segmentURL else {
            if persistManifest, let lastWrittenPTS {
                manifest.lastPTSSeconds = CMTimeGetSeconds(lastWrittenPTS)
                try writeManifest()
            }
            return
        }

        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }

        guard writer.status == .completed else {
            throw ProcessorError.writer(
                writer.error?.localizedDescription ?? "pause checkpoint writer failed to finish"
            )
        }

        if !manifest.segments.contains(url.lastPathComponent) {
            manifest.segments.append(url.lastPathComponent)
        }
        if let lastWrittenPTS {
            manifest.lastPTSSeconds = CMTimeGetSeconds(lastWrittenPTS)
        }

        if persistManifest {
            try writeManifest()
            DiagnosticsLogger.shared.log(
                "Pause checkpoint saved • stage=\(stageID) • pts=\(String(format: "%.3f", manifest.lastPTSSeconds))s • segments=\(manifest.segments.count)"
            )
        }

        self.writer = nil
        self.input = nil
        self.adaptor = nil
        self.segmentStartPTS = nil
        self.segmentURL = nil
        self.lastWrittenPTS = nil
    }

    private func composeSegments() async throws -> URL {
        let assets = try await manifest.segments.map { name -> AVURLAsset in
            let url = directory.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else { throw ProcessorError.noOutput }
            return AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
        }

        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            throw ProcessorError.noOutput
        }

        var cursor = CMTime.zero
        for asset in assets {
            guard let sourceTrack = try await asset.loadTracks(withMediaType: .video).first else {
                throw ProcessorError.missingVideoTrack
            }
            let range = try await sourceTrack.load(.timeRange)
            try videoTrack.insertTimeRange(range, of: sourceTrack, at: cursor)
            cursor = CMTimeAdd(cursor, range.duration)
        }

        if let first = assets.first,
           let firstTrack = try await first.loadTracks(withMediaType: .video).first {
            videoTrack.preferredTransform = try await firstTrack.load(.preferredTransform)
        }

        let finalURL = fm.temporaryDirectory
            .appendingPathComponent("pause-checkpoint-complete-\(stageID)-\(UUID().uuidString).mov")
        try? fm.removeItem(at: finalURL)

        guard let exporter = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetPassthrough
        ) else {
            throw ProcessorError.writer("could not create pause checkpoint compositor")
        }

        exporter.outputURL = finalURL
        exporter.outputFileType = .mov
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            exporter.exportAsynchronously { continuation.resume() }
        }

        guard exporter.status == .completed else {
            try? fm.removeItem(at: finalURL)
            throw ProcessorError.writer(
                exporter.error?.localizedDescription ?? "pause checkpoint composition failed"
            )
        }
        return finalURL
    }

    private func writeManifest(_ value: Manifest? = nil) throws {
        let data = try JSONEncoder().encode(value ?? manifest)
        try data.write(to: manifestURL, options: .atomic)
    }
}
