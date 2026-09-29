import Foundation
import AVFoundation
import VideoToolbox

/// Delivery-only compression pass. The AI pipeline creates its high-quality master
/// first; only the finished master is size-limited. On iOS AVAssetWriter's HEVC
/// encoder specification key is unavailable, so we verify that a hardware HEVC
/// encoder exists before starting the system HEVC Main10 writer path.
final class FinalSizeOptimizer {
    static let hardLimitBytes: Int64 = 1_000_000_000
    static let targetBytes: Int64 = 900_000_000

    struct Result: Sendable {
        let url: URL
        let optimized: Bool
        let originalBytes: Int64
        let finalBytes: Int64
        let targetVideoBitrate: Int
    }

    enum OptimizerError: LocalizedError {
        case missingVideo
        case invalidDuration
        case bitrateTooLow
        case hardwareHEVCUnavailable
        case cannotAddTrack(String)
        case reader(String)
        case writer(String)
        case emptyOutput
        case couldNotMeetLimit

        var errorDescription: String? {
            switch self {
            case .missingVideo: return "Final size optimizer could not find the video track."
            case .invalidDuration: return "Final size optimizer received an invalid video duration."
            case .bitrateTooLow: return "This video is too long to fit safely below 1 GB at the minimum supported quality."
            case .hardwareHEVCUnavailable: return "A hardware-accelerated HEVC encoder is not available. The high-quality master was kept intact."
            case .cannotAddTrack(let name): return "Final size optimizer could not add the \(name) track."
            case .reader(let message): return "Final size optimizer reader failed: \(message)"
            case .writer(let message): return "Final size optimizer writer failed: \(message)"
            case .emptyOutput: return "Final size optimizer produced an empty file."
            case .couldNotMeetLimit: return "Final size optimizer could not keep the completed video below 1 GB."
            }
        }
    }

    func optimizeIfNeeded(sourceURL: URL, progress: @escaping @Sendable (Double, String) -> Void) async throws -> Result {
        let originalBytes = fileSize(sourceURL)
        guard originalBytes >= Self.hardLimitBytes else {
            progress(1.0, "Final size pass skipped • already under 1 GB")
            return Result(url: sourceURL, optimized: false, originalBytes: originalBytes, finalBytes: originalBytes, targetVideoBitrate: 0)
        }

        guard hardwareHEVCEncoderAvailable() else { throw OptimizerError.hardwareHEVCUnavailable }

        let asset = AVURLAsset(url: sourceURL)
        let duration = try await asset.load(.duration)
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds > 0 else { throw OptimizerError.invalidDuration }
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else { throw OptimizerError.missingVideo }
        let fps = max(1.0, Double(try await videoTrack.load(.nominalFrameRate)))
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)

        let audioBps = audioTracks.isEmpty ? 0.0 : 256_000.0
        let muxReserveBps = 128_000.0
        let totalBudgetBps = (Double(Self.targetBytes) * 8.0 / seconds) * 0.975
        let calculated = totalBudgetBps - audioBps - muxReserveBps
        guard calculated >= 1_000_000 else { throw OptimizerError.bitrateTooLow }
        let bitrate = Int(calculated.rounded(.down))

        DiagnosticsLogger.shared.log("Final size optimizer begin • master=\(formatGB(originalBytes)) GB • duration=\(String(format: "%.3f", seconds))s • target video bitrate=\(String(format: "%.2f", Double(bitrate) / 1_000_000.0)) Mbps • HEVC Main10 • hardware HEVC verified")
        progress(0.0, "Final delivery compression • Hardware HEVC Main10 • target < 1 GB")

        let first = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-FinalUnder1GB-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: first)
        try await transcode(asset: asset, videoTrack: videoTrack, audioTrack: audioTracks.first, outputURL: first, bitrate: bitrate, fps: fps, durationSeconds: seconds, progress: progress)

        var finalURL = first
        var finalBytes = fileSize(first)
        var usedBitrate = bitrate
        guard finalBytes > 0 else { throw OptimizerError.emptyOutput }

        if finalBytes >= Self.hardLimitBytes {
            let ratio = Double(Self.targetBytes) / Double(finalBytes)
            let retryBitrate = max(1_000_000, Int(Double(bitrate) * ratio * 0.94))
            let retry = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-FinalUnder1GB-Retry-\(UUID().uuidString).mp4")
            try? FileManager.default.removeItem(at: retry)
            DiagnosticsLogger.shared.log("Final size optimizer overshoot • \(formatGB(finalBytes)) GB • retry bitrate=\(String(format: "%.2f", Double(retryBitrate) / 1_000_000.0)) Mbps")
            progress(0.02, "Final delivery compression • size overshoot • hardware retry")
            try await transcode(asset: asset, videoTrack: videoTrack, audioTrack: audioTracks.first, outputURL: retry, bitrate: retryBitrate, fps: fps, durationSeconds: seconds, progress: progress)
            let retryBytes = fileSize(retry)
            guard retryBytes > 0, retryBytes < Self.hardLimitBytes else {
                try? FileManager.default.removeItem(at: retry)
                throw OptimizerError.couldNotMeetLimit
            }
            try? FileManager.default.removeItem(at: first)
            finalURL = retry
            finalBytes = retryBytes
            usedBitrate = retryBitrate
        }

        DiagnosticsLogger.shared.log("Final size optimizer complete • output=\(formatGB(finalBytes)) GB • bitrate=\(String(format: "%.2f", Double(usedBitrate) / 1_000_000.0)) Mbps • hardware HEVC path")
        progress(1.0, "Final hardware compression complete • \(formatMB(finalBytes)) MB")
        return Result(url: finalURL, optimized: true, originalBytes: originalBytes, finalBytes: finalBytes, targetVideoBitrate: usedBitrate)
    }

    private func hardwareHEVCEncoderAvailable() -> Bool {
        var list: CFArray?
        guard VTCopyVideoEncoderList(nil, &list) == noErr,
              let encoders = list as? [[String: Any]] else { return false }
        let codecKey = kVTVideoEncoderList_CodecType as String
        let hardwareKey = kVTVideoEncoderList_IsHardwareAccelerated as String
        for encoder in encoders {
            guard let codecNumber = encoder[codecKey] as? NSNumber else { continue }
            let codec = CMVideoCodecType(codecNumber.uint32Value)
            let hardware = (encoder[hardwareKey] as? NSNumber)?.boolValue ?? (encoder[hardwareKey] as? Bool ?? false)
            if codec == kCMVideoCodecType_HEVC && hardware { return true }
        }
        return false
    }

    private func transcode(asset: AVURLAsset, videoTrack: AVAssetTrack, audioTrack: AVAssetTrack?, outputURL: URL, bitrate: Int, fps: Double, durationSeconds: Double, progress: @escaping @Sendable (Double, String) -> Void) async throws {
        let reader = try AVAssetReader(asset: asset)
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        writer.shouldOptimizeForNetworkUse = true

        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
        ])
        videoOutput.alwaysCopiesSampleData = false
        guard reader.canAdd(videoOutput) else { throw OptimizerError.cannotAddTrack("video reader") }
        reader.add(videoOutput)

        let naturalSize = try await videoTrack.load(.naturalSize)
        let transform = try await videoTrack.load(.preferredTransform)
        let width = max(2, Int(abs(naturalSize.width).rounded()))
        let height = max(2, Int(abs(naturalSize.height).rounded()))

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: bitrate,
            AVVideoExpectedSourceFrameRateKey: Int(fps.rounded()),
            AVVideoMaxKeyFrameIntervalKey: max(1, Int(fps.rounded() * 2.0)),
            AVVideoAllowFrameReorderingKey: true,
            AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String
        ]
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression
        ]
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput.expectsMediaDataInRealTime = false
        videoInput.transform = transform
        guard writer.canAdd(videoInput) else { throw OptimizerError.cannotAddTrack("HEVC Main10 video writer") }
        writer.add(videoInput)

        var audioOutput: AVAssetReaderTrackOutput?
        var audioInput: AVAssetWriterInput?
        if let audioTrack {
            let sourceFormat = try await audioTrack.load(.formatDescriptions).first
            let basic = sourceFormat.flatMap { CMAudioFormatDescriptionGetStreamBasicDescription($0)?.pointee }
            let sampleRate = basic.map { max(8_000.0, $0.mSampleRate) } ?? 48_000.0
            let channels = basic.map { max(1, min(2, Int($0.mChannelsPerFrame))) } ?? 2
            let ao = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsNonInterleaved: false
            ])
            guard reader.canAdd(ao) else { throw OptimizerError.cannotAddTrack("audio reader") }
            reader.add(ao)
            let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: channels,
                AVEncoderBitRateKey: 256_000
            ])
            ai.expectsMediaDataInRealTime = false
            guard writer.canAdd(ai) else { throw OptimizerError.cannotAddTrack("audio writer") }
            writer.add(ai)
            audioOutput = ao
            audioInput = ai
        }

        guard writer.startWriting() else { throw OptimizerError.writer(writer.error?.localizedDescription ?? "could not start HEVC Main10 encoder") }
        guard reader.startReading() else { throw OptimizerError.reader(reader.error?.localizedDescription ?? "could not start") }
        writer.startSession(atSourceTime: .zero)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let group = DispatchGroup()
            let videoQueue = DispatchQueue(label: "RIFE60.final-size.video", qos: .userInitiated)
            let audioQueue = DispatchQueue(label: "RIFE60.final-size.audio", qos: .userInitiated)
            let stateQueue = DispatchQueue(label: "RIFE60.final-size.state")
            var firstError: Error?
            func record(_ error: Error) { stateQueue.sync { if firstError == nil { firstError = error } }; reader.cancelReading() }

            group.enter()
            videoInput.requestMediaDataWhenReady(on: videoQueue) {
                while videoInput.isReadyForMoreMediaData {
                    if reader.status == .failed { record(OptimizerError.reader(reader.error?.localizedDescription ?? "video read failed")); videoInput.markAsFinished(); group.leave(); return }
                    guard let sample = videoOutput.copyNextSampleBuffer() else { videoInput.markAsFinished(); group.leave(); return }
                    if !videoInput.append(sample) { record(OptimizerError.writer(writer.error?.localizedDescription ?? "video append failed")); videoInput.markAsFinished(); group.leave(); return }
                    let s = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
                    if s.isFinite, durationSeconds > 0 {
                        let p = min(max(s / durationSeconds, 0), 0.999)
                        progress(p, "Final delivery compression • \(Int(p * 100))% • Hardware HEVC Main10")
                    }
                }
            }

            if let audioOutput, let audioInput {
                group.enter()
                audioInput.requestMediaDataWhenReady(on: audioQueue) {
                    while audioInput.isReadyForMoreMediaData {
                        if reader.status == .failed { record(OptimizerError.reader(reader.error?.localizedDescription ?? "audio read failed")); audioInput.markAsFinished(); group.leave(); return }
                        guard let sample = audioOutput.copyNextSampleBuffer() else { audioInput.markAsFinished(); group.leave(); return }
                        if !audioInput.append(sample) { record(OptimizerError.writer(writer.error?.localizedDescription ?? "audio append failed")); audioInput.markAsFinished(); group.leave(); return }
                    }
                }
            }

            group.notify(queue: videoQueue) {
                let capturedError = stateQueue.sync { firstError }
                if let capturedError { writer.cancelWriting(); continuation.resume(throwing: capturedError); return }
                if reader.status == .failed { writer.cancelWriting(); continuation.resume(throwing: OptimizerError.reader(reader.error?.localizedDescription ?? "read failed")); return }
                writer.finishWriting {
                    if writer.status == .completed { continuation.resume() }
                    else { continuation.resume(throwing: OptimizerError.writer(writer.error?.localizedDescription ?? "finish failed")) }
                }
            }
        }
    }

    private func fileSize(_ url: URL) -> Int64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? NSNumber)?.int64Value ?? 0
    }
    private func formatMB(_ bytes: Int64) -> String { String(format: "%.0f", Double(bytes) / 1_000_000.0) }
    private func formatGB(_ bytes: Int64) -> String { String(format: "%.2f", Double(bytes) / 1_000_000_000.0) }
}
