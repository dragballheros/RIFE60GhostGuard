import AVFoundation
import Foundation

/// Copies the reference audio track onto an H3 video without re-encoding either stream.
enum H3OriginalAudio {
    static func attach(reference: URL, generated: URL) async throws {
        let videoAsset = AVURLAsset(url: generated)
        let audioAsset = AVURLAsset(url: reference)
        guard let videoTrack = try await videoAsset.loadTracks(withMediaType: .video).first else {
            throw Failure.missingVideo
        }
        guard let audioTrack = try await audioAsset.loadTracks(withMediaType: .audio).first else {
            CrashCapture.step("H3 keep original audio: source has no audio track; output stays silent")
            return
        }
        let videoDuration = try await videoAsset.load(.duration)
        let audioRange = try await audioTrack.load(.timeRange)
        let duration = CMTimeMinimum(videoDuration, audioRange.duration)
        guard duration.seconds > 0 else {
            CrashCapture.step("H3 keep original audio: source audio duration is empty; output stays silent")
            return
        }

        let videoReader = try AVAssetReader(asset: videoAsset)
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: nil)
        videoOutput.alwaysCopiesSampleData = false
        videoReader.add(videoOutput)
        let audioReader = try AVAssetReader(asset: audioAsset)
        audioReader.timeRange = CMTimeRange(start: .zero, duration: duration)
        let audioOutput = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
        audioOutput.alwaysCopiesSampleData = false
        audioReader.add(audioOutput)

        let temporary = generated.deletingLastPathComponent()
            .appendingPathComponent("h3-audio-\(UUID().uuidString).mov")
        let writer = try AVAssetWriter(outputURL: temporary, fileType: .mov)
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: nil)
        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
        videoInput.expectsMediaDataInRealTime = false
        audioInput.expectsMediaDataInRealTime = false
        guard writer.canAdd(videoInput), writer.canAdd(audioInput) else { throw Failure.writer }
        writer.add(videoInput)
        writer.add(audioInput)
        guard videoReader.startReading(), audioReader.startReading() else {
            throw Failure.reader(videoReader.error?.localizedDescription ?? audioReader.error?.localizedDescription ?? "reader failed")
        }
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        try await pump(videoOutput, into: videoInput)
        try await pump(audioOutput, into: audioInput)
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        if writer.status != .completed {
            try? FileManager.default.removeItem(at: temporary)
            throw Failure.writer
        }
        _ = try FileManager.default.replaceItemAt(generated, withItemAt: temporary)
        CrashCapture.step("H3 muxed original audio passthrough duration=\(duration.seconds)")
    }

    private static func pump(_ output: AVAssetReaderTrackOutput, into input: AVAssetWriterInput) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let gate = NSLock()
            var finished = false
            let queue = DispatchQueue(label: "h3.audio.passthrough")
            input.requestMediaDataWhenReady(on: queue) {
                while input.isReadyForMoreMediaData {
                    gate.lock()
                    if finished {
                        gate.unlock()
                        return
                    }
                    gate.unlock()
                    if let sample = output.copyNextSampleBuffer() {
                        if !input.append(sample) {
                            gate.lock()
                            finished = true
                            gate.unlock()
                            input.markAsFinished()
                            continuation.resume(throwing: Failure.writer)
                            return
                        }
                    } else {
                        gate.lock()
                        finished = true
                        gate.unlock()
                        input.markAsFinished()
                        continuation.resume()
                        return
                    }
                }
            }
        }
    }

    enum Failure: LocalizedError {
        case missingVideo
        case reader(String)
        case writer

        var errorDescription: String? {
            switch self {
            case .missingVideo:
                return "H3 output has no video track to attach the original audio to."
            case .reader(let message):
                return "Could not read the original audio track: \(message)"
            case .writer:
                return "Could not mux the original audio without re-encoding."
            }
        }
    }
}
