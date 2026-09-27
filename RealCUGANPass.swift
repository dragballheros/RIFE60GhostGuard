import Foundation
import AVFoundation
import CoreML
import VideoToolbox

final class RealCUGANPass {
    private let intensity: Double
    private var outputTransferSession: VTPixelTransferSession?
    private var modelInputTransferSession: VTPixelTransferSession?
    private var modelInputPool: CVPixelBufferPool?

    private let modelInputWidth = 1920
    private let modelInputHeight = 1080
    private let modelOutputWidth = 3840
    private let modelOutputHeight = 2160

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

        DiagnosticsLogger.shared.log("Real-CUGAN entered • source=\(width)x\(height) • duration=\(String(format: "%.3f", seconds))s • thermal=\(currentThermalStateName())")
        if width != modelInputWidth || height != modelInputHeight {
            DiagnosticsLogger.shared.log("Real-CUGAN source normalization enabled • \(width)x\(height) -> \(modelInputWidth)x\(modelInputHeight) using aspect-preserving letterbox")
            progress(0.001, "Pass 3/3 • Normalizing \(width)×\(height) for Real-CUGAN…")
            try prepareModelInputScaler()
        }

        progress(0.002, "Pass 3/3 • Loading Real-CUGAN Anime 2x Noise 3…")
        DiagnosticsLogger.shared.log("Real-CUGAN model load begin")
        let model = try loadModel()
        DiagnosticsLogger.shared.log("Real-CUGAN model load complete")

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
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
            AVVideoWidthKey: modelOutputWidth,
            AVVideoHeightKey: modelOutputHeight,
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
            kCVPixelBufferWidthKey as String: modelOutputWidth,
            kCVPixelBufferHeightKey as String: modelOutputHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ])
        guard writer.canAdd(input) else { throw ProcessorError.writer("cannot attach 4K HEVC Main10 writer") }
        writer.add(input)
        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN reader failed") }
        guard writer.startWriting() else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "4K writer failed") }
        writer.startSession(atSourceTime: .zero)
        guard let pool = adaptor.pixelBufferPool else { throw ProcessorError.writer("4K P010 pixel buffer pool unavailable") }
        DiagnosticsLogger.shared.log("Real-CUGAN reader/writer started • target=3840x2160 HEVC Main10 • bitrate=\(videoBitrate)")

        let alphaValue = Float(1.0 / max(intensity, 0.01))
        let alpha = try MLMultiArray(shape: [1], dataType: .float16)
        alpha[0] = NSNumber(value: alphaValue)
        var frames = 0
        var inferenceSeconds = 0.0
        var encodeSeconds = 0.0
        let passStart = CFAbsoluteTimeGetCurrent()

        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let decodedFrame = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            let frameNumber = frames + 1

            if frameNumber == 1 {
                DiagnosticsLogger.shared.log("Real-CUGAN first frame decoded • pixel=\(CVPixelBufferGetWidth(decodedFrame))x\(CVPixelBufferGetHeight(decodedFrame))")
            }

            let modelFrame = try normalizedModelInput(decodedFrame)
            if frameNumber == 1 {
                DiagnosticsLogger.shared.log("Real-CUGAN first model input ready • pixel=\(CVPixelBufferGetWidth(modelFrame))x\(CVPixelBufferGetHeight(modelFrame)) • prediction begin")
            }

            let started = CFAbsoluteTimeGetCurrent()
            let upscaled: CVPixelBuffer = try autoreleasepool {
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(pixelBuffer: modelFrame),
                    "alpha": MLFeatureValue(multiArray: alpha)
                ])
                let prediction = try model.prediction(from: provider)
                guard let buffer = prediction.featureValue(for: "output")?.imageBufferValue else {
                    throw ProcessorError.conversionFailed("Real-CUGAN did not return a 4K image buffer")
                }
                return buffer
            }
            inferenceSeconds += CFAbsoluteTimeGetCurrent() - started

            if frameNumber == 1 {
                DiagnosticsLogger.shared.log("Real-CUGAN first prediction complete • output=\(CVPixelBufferGetWidth(upscaled))x\(CVPixelBufferGetHeight(upscaled))")
            }

            let encodeStart = CFAbsoluteTimeGetCurrent()
            try await append10Bit(upscaled, at: pts, input: input, adaptor: adaptor, pool: pool)
            encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStart
            frames += 1

            if frames == 1 || frames % 10 == 0 {
                DiagnosticsLogger.shared.log("Real-CUGAN frame \(frames) complete • inference=\(String(format: "%.1f", inferenceSeconds * 1000.0 / Double(frames)))ms/frame • thermal=\(currentThermalStateName())")
            }

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
        guard frames > 0 else { throw ProcessorError.conversionFailed("Real-CUGAN received zero decoded frames") }
        DiagnosticsLogger.shared.log("Real-CUGAN inference loop complete • frames=\(frames) • finishing writer")
        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in writer.finishWriting { continuation.resume() } }
        guard writer.status == .completed else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "4K finish failed") }
        DiagnosticsLogger.shared.log("Real-CUGAN writer completed • frames=\(frames)")
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

    private func prepareModelInputScaler() throws {
        if modelInputPool == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: modelInputWidth,
                kCVPixelBufferHeightKey as String: modelInputHeight,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
            var pool: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
            guard status == kCVReturnSuccess, let pool else {
                throw ProcessorError.conversionFailed("could not create 1920×1080 Real-CUGAN input pool")
            }
            modelInputPool = pool
        }
        if modelInputTransferSession == nil {
            var session: VTPixelTransferSession?
            let status = VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session)
            guard status == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create Real-CUGAN input scaler")
            }
            let propertyStatus = VTSessionSetProperty(session, key: kVTPixelTransferPropertyKey_ScalingMode, value: kVTScalingMode_Letterbox)
            guard propertyStatus == noErr else {
                throw ProcessorError.conversionFailed("could not configure Real-CUGAN aspect-preserving scaler")
            }
            modelInputTransferSession = session
        }
    }

    private func normalizedModelInput(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        if width == modelInputWidth && height == modelInputHeight { return source }

        try prepareModelInputScaler()
        guard let modelInputPool else {
            throw ProcessorError.conversionFailed("Real-CUGAN input pool unavailable")
        }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, modelInputPool, &destination) == kCVReturnSuccess, let destination else {
            throw ProcessorError.conversionFailed("could not allocate normalized 1920×1080 Real-CUGAN frame")
        }
        guard let modelInputTransferSession,
              VTPixelTransferSessionTransferImage(modelInputTransferSession, from: source, to: destination) == noErr else {
            throw ProcessorError.conversionFailed("Real-CUGAN input normalization failed for \(width)×\(height)")
        }
        return destination
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
        if outputTransferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create 4K pixel transfer session")
            }
            outputTransferSession = session
        }
        guard let outputTransferSession,
              VTPixelTransferSessionTransferImage(outputTransferSession, from: source, to: destination) == noErr else {
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
