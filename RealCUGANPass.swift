import Foundation
import AVFoundation
import CoreML
import VideoToolbox

final class RealCUGANPass {
    private let intensity: Double
    private var transferSession: VTPixelTransferSession?

    init(intensity: Double = 1.30) { self.intensity = intensity }

    func run(
        sourceURL: URL,
        finalAudioBitrate: Double,
        progress: @escaping @Sendable (Double, String) -> Void,
        telemetry: @escaping @Sendable (PerformanceTelemetry) -> Void
    ) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ProcessorError.missingVideoTrack }
        let duration = try await asset.load(.duration)
        let seconds = max(CMTimeGetSeconds(duration), 0.001)
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())
        guard width == 1920, height == 1080 else {
            throw ProcessorError.conversionFailed("The optimized Real-CUGAN path currently expects 1920×1080 input. This source is \(width)×\(height).")
        }

        progress(0.001, "Pass 3/3 • Loading Real-CUGAN Anime 2x Noise 3…")
        let model = try loadModel()
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot attach Real-CUGAN reader") }
        reader.add(output)

        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-CUGAN-4K-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: outURL)
        let writer = try AVAssetWriter(outputURL: outURL, fileType: .mov)

        let targetTotalBytes = 950_000_000.0
        let containerReserveBytes = 12_000_000.0
        let usableBits = max((targetTotalBytes - containerReserveBytes) * 8.0, 8_000_000.0)
        let audioBits = max(finalAudioBitrate, 0) * seconds
        let videoBitrate = Int(max((usableBits - audioBits) / seconds * 0.97, 500_000.0))
        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: videoBitrate,
            AVVideoQualityKey: 1.0,
            AVVideoExpectedSourceFrameRateKey: 60,
            AVVideoMaxKeyFrameIntervalKey: 120,
            AVVideoAllowFrameReorderingKey: true,
            AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 3840,
            AVVideoHeightKey: 2160,
            AVVideoCompressionPropertiesKey: compression,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ])
        input.expectsMediaDataInRealTime = false
        input.transform = transform
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelBufferWidthKey as String: 3840,
            kCVPixelBufferHeightKey as String: 2160,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ])
        guard writer.canAdd(input) else { throw ProcessorError.writer("cannot attach 4K HEVC Main10 writer") }
        writer.add(input)
        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN reader failed") }
        guard writer.startWriting() else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "4K writer failed") }
        writer.startSession(atSourceTime: .zero)
        guard let pool = adaptor.pixelBufferPool else { throw ProcessorError.writer("4K P010 pixel buffer pool unavailable") }

        let alphaValue = Float(1.0 / max(intensity, 0.01))
        let alpha = try MLMultiArray(shape: [1], dataType: .float16)
        alpha[0] = NSNumber(value: alphaValue)
        var frames = 0
        var inferenceSeconds = 0.0
        var encodeSeconds = 0.0
        let passStart = CFAbsoluteTimeGetCurrent()

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            let started = CFAbsoluteTimeGetCurrent()
            let provider = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: frame), "alpha": MLFeatureValue(multiArray: alpha)])
            let prediction = try autoreleasepool { try model.prediction(from: provider) }
            guard let upscaled = prediction.featureValue(for: "output")?.imageBufferValue else {
                throw ProcessorError.conversionFailed("Real-CUGAN did not return a 4K image buffer")
            }
            inferenceSeconds += CFAbsoluteTimeGetCurrent() - started
            let encodeStart = CFAbsoluteTimeGetCurrent()
            try await append10Bit(upscaled, at: pts, input: input, adaptor: adaptor, pool: pool)
            encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStart
            frames += 1

            if frames % 3 == 0 {
                let elapsed = max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)
                var t = PerformanceTelemetry()
                t.upscaledFrames = frames
                t.cuganMsPerFrame = inferenceSeconds * 1000.0 / Double(frames)
                t.encodeMsPerOutputFrame = encodeSeconds * 1000.0 / Double(frames)
                t.upscaleFPS = Double(frames) / elapsed
                t.thermalState = currentThermalStateName()
                telemetry(t)
            }
            if frames % 6 == 0 {
                let frac = min(max(CMTimeGetSeconds(pts) / seconds, 0), 1)
                let fps = Double(frames) / max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)
                let remaining = fps > 0 ? (seconds * 60.0 - Double(frames)) / fps : 0
                progress(frac, "Pass 3/3 • Real-CUGAN 4K • \(frames) frames • ETA \(formatDuration(remaining))")
                await Task.yield()
            }
        }

        if reader.status == .failed { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN decode failed") }
        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in writer.finishWriting { continuation.resume() } }
        guard writer.status == .completed else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "4K finish failed") }
        return outURL
    }

    private func loadModel() throws -> MLModel {
        guard let url = Bundle.main.url(forResource: "RealCUGAN2xNoise3_1080p", withExtension: "mlmodelc") else {
            throw ProcessorError.conversionFailed("Bundled Real-CUGAN model is missing")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.allowLowPrecisionAccumulationOnGPU = true
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private func append10Bit(_ source: CVPixelBuffer, at time: CMTime, input: AVAssetWriterInput, adaptor: AVAssetWriterInputPixelBufferAdaptor, pool: CVPixelBufferPool) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess, let destination else {
            throw ProcessorError.conversionFailed("could not allocate 4K P010 frame")
        }
        if transferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create 4K pixel transfer session")
            }
            transferSession = session
        }
        guard let transferSession, VTPixelTransferSessionTransferImage(transferSession, from: source, to: destination) == noErr else {
            throw ProcessorError.conversionFailed("4K BGRA→P010 conversion failed")
        }
        guard adaptor.append(destination, withPresentationTime: time) else { throw ProcessorError.writer("failed appending 4K frame") }
    }
}

func formatDuration(_ seconds: Double) -> String {
    guard seconds.isFinite, seconds > 0 else { return "--:--" }
    let total = Int(seconds.rounded())
    let h = total / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    return h > 0 ? String(format: "%d:%02d:%02d", h, m, s) : String(format: "%02d:%02d", m, s)
}
