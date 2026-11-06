import Foundation
import AVFoundation
import VideoToolbox
import CoreVideo

final class IncrementalVideoCheckpointWriter {
    private let fm = FileManager.default
    private let directory: URL
    private let removesDirectoryOnFinish: Bool
    private let outputSettings: [String: Any]
    private let sourcePixelBufferAttributes: [String: Any]
    private let transform: CGAffineTransform
    private let fileType: AVFileType

    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var segmentStartPTS: CMTime?
    private var segmentURL: URL?
    private var lastWrittenPTS: CMTime?
    private var transferSession: VTPixelTransferSession?
    private(set) var readyWaitSeconds = 0.0
    private(set) var transferSeconds = 0.0
    private(set) var appendCallSeconds = 0.0

    init(
        recoveryDirectory: URL?,
        stageID: String,
        outputSettings: [String: Any],
        sourcePixelBufferAttributes: [String: Any],
        transform: CGAffineTransform,
        fileType: AVFileType = .mov,
        segmentDuration: Double = 0
    ) throws {
        self.outputSettings = outputSettings
        self.sourcePixelBufferAttributes = sourcePixelBufferAttributes
        self.transform = transform
        self.fileType = fileType

        if let recoveryDirectory {
            self.directory = recoveryDirectory.appendingPathComponent("incremental-\(stageID)", isDirectory: true)
            self.removesDirectoryOnFinish = false
        } else {
            self.directory = fm.temporaryDirectory.appendingPathComponent("rife-stage-\(UUID().uuidString)", isDirectory: true)
            self.removesDirectoryOnFinish = true
        }
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)


    }

    func checkCancellation() async throws {
        try Task.checkCancellation()
    }

    private func waitUntilReady(_ input: AVAssetWriterInput) async throws {
        if input.isReadyForMoreMediaData { return }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let queue = DispatchQueue(label: "RIFE60.media-ready")
            input.requestMediaDataWhenReady(on: queue) {
                if input.isReadyForMoreMediaData {
                    continuation.resume()
                }
            }
        }
        try await checkCancellation()
    }


    func append(_ pixelBuffer: CVPixelBuffer, at presentationTime: CMTime) async throws {
        try await checkCancellation()
        try await ensureWriter(startingAt: presentationTime)
        guard let input, let adaptor else {
            throw ProcessorError.writer("video stage writer unavailable")
        }

        try await waitUntilReady(input)

        guard adaptor.append(pixelBuffer, withPresentationTime: localTime(presentationTime)) else {
            throw ProcessorError.writer("failed writing video stage frame")
        }
        lastWrittenPTS = presentationTime
    }

    func append10Bit(_ source: CVPixelBuffer, at presentationTime: CMTime) async throws {
        try await checkCancellation()
        try await ensureWriter(startingAt: presentationTime)

        guard let input, let adaptor, let pool = adaptor.pixelBufferPool else {
            throw ProcessorError.writer("video stage 10-bit pool unavailable")
        }

        let readyStart = CFAbsoluteTimeGetCurrent()
        try await waitUntilReady(input)
        readyWaitSeconds += CFAbsoluteTimeGetCurrent() - readyStart

        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess,
              let destination else {
            throw ProcessorError.conversionFailed("could not allocate video stage 10-bit frame")
        }

        if transferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(
                allocator: kCFAllocatorDefault,
                pixelTransferSessionOut: &session
            ) == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create video stage pixel transfer session")
            }
            // Same VideoToolbox transfer into the writer's 10-bit pool as before.
            // The session runs off the inference thread so the encoder can overlap it.
            transferSession = session
        }

        let transferStart = CFAbsoluteTimeGetCurrent()
        guard let transferSession,
              VTPixelTransferSessionTransferImage(
                transferSession,
                from: source,
                to: destination
              ) == noErr else {
            throw ProcessorError.conversionFailed("BGRA→P010 conversion failed for video stage")
        }
        transferSeconds += CFAbsoluteTimeGetCurrent() - transferStart

        let appendStart = CFAbsoluteTimeGetCurrent()
        guard adaptor.append(destination, withPresentationTime: localTime(presentationTime)) else {
            throw ProcessorError.writer("failed writing video stage 10-bit frame")
        }
        appendCallSeconds += CFAbsoluteTimeGetCurrent() - appendStart
        lastWrittenPTS = presentationTime
    }

    func finish() async throws -> URL {
        let url = directory.appendingPathComponent("00000.mov")
        try await finishCurrentSegment()
        guard fm.fileExists(atPath: url.path) else { throw ProcessorError.noOutput }
        if removesDirectoryOnFinish {
            let output = fm.temporaryDirectory.appendingPathComponent("rife-stage-output-\(UUID().uuidString).mov")
            try fm.moveItem(at: url, to: output)
            try? fm.removeItem(at: directory)
            return output
        }
        return url
    }

    func discardTemporaryMedia() {
        try? fm.removeItem(at: directory)
    }

    static func cleanup(recoveryDirectory: URL?, stageID: String) {
        guard let recoveryDirectory else { return }
        try? FileManager.default.removeItem(
            at: recoveryDirectory.appendingPathComponent("incremental-\(stageID)", isDirectory: true)
        )
    }

    private func ensureWriter(startingAt pts: CMTime) async throws {
        guard writer == nil else { return }

        let url = directory.appendingPathComponent(
            "00000.mov"
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
            throw ProcessorError.writer("cannot attach video stage writer input")
        }
        writer.add(input)

        guard writer.startWriting() else {
            throw ProcessorError.writer(writer.error?.localizedDescription ?? "video stage writer failed")
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

    private func finishCurrentSegment() async throws {
        guard let writer, let input else { return }
        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw ProcessorError.writer(writer.error?.localizedDescription ?? "video stage writer failed to finish")
        }
        self.writer = nil
        self.input = nil
        self.adaptor = nil
        self.segmentStartPTS = nil
        self.segmentURL = nil
        self.lastWrittenPTS = nil
    }


}
