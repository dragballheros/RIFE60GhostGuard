import Foundation
import AVFoundation
import CoreML
import CoreImage
import VideoToolbox

final class RealCUGANPass {
    private let intensity: Double
    private var outputTransferSession: VTPixelTransferSession?

    // The Core ML model operates on fixed 512px tiles to keep peak memory low.
    // We keep 32px of context on every side and only stitch the 448px center.
    // This means the FINAL frame is always exactly 2x the ORIGINAL frame size.
    private let tileSize = 512
    private let tileOverlap = 32
    private var tileStep: Int { tileSize - tileOverlap * 2 }
    private var tileInputPool: CVPixelBufferPool?
    private var stitchedFramePool: CVPixelBufferPool?
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

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
        let targetWidth = width * 2
        let targetHeight = height * 2
        let tilesAcross = Int(ceil(Double(width) / Double(tileStep)))
        let tilesDown = Int(ceil(Double(height) / Double(tileStep)))
        let tilesPerFrame = tilesAcross * tilesDown

        DiagnosticsLogger.shared.log("Real-CUGAN entered • native source=\(width)x\(height) • native 2x target=\(targetWidth)x\(targetHeight) • tiles/frame=\(tilesPerFrame) • duration=\(String(format: "%.3f", seconds))s • thermal=\(currentThermalStateName())")
        progress(0.001, "Pass 3/3 • Real-CUGAN native 2× • \(width)×\(height) → \(targetWidth)×\(targetHeight)…")

        try preparePools(targetWidth: targetWidth, targetHeight: targetHeight)
        DiagnosticsLogger.shared.log("Real-CUGAN tile pools ready • tile=\(tileSize) • overlap=\(tileOverlap) • core=\(tileStep)")

        progress(0.002, "Pass 3/3 • Loading Real-CUGAN Anime 2x Noise 3…")
        DiagnosticsLogger.shared.log("Real-CUGAN tiled model load begin")
        let model = try loadModel()
        DiagnosticsLogger.shared.log("Real-CUGAN tiled model load complete")

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ProcessorError.reader("cannot attach Real-CUGAN reader") }
        reader.add(output)

        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("RIFE60-CUGAN-2X-\(UUID().uuidString).mov")
        try? FileManager.default.removeItem(at: outURL)
        let writer = try AVAssetWriter(outputURL: outURL, fileType: .mov)

        // Keep the final file below the requested ~1 GB ceiling while preserving
        // the highest practical HEVC Main10 quality inside that size budget.
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
            AVVideoWidthKey: targetWidth,
            AVVideoHeightKey: targetHeight,
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
            kCVPixelBufferWidthKey as String: targetWidth,
            kCVPixelBufferHeightKey as String: targetHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ])
        guard writer.canAdd(input) else { throw ProcessorError.writer("cannot attach native 2x HEVC Main10 writer") }
        writer.add(input)
        guard reader.startReading() else { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN reader failed") }
        guard writer.startWriting() else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "native 2x writer failed") }
        writer.startSession(atSourceTime: .zero)
        guard let writerPool = adaptor.pixelBufferPool else { throw ProcessorError.writer("native 2x P010 pixel buffer pool unavailable") }
        DiagnosticsLogger.shared.log("Real-CUGAN reader/writer started • target=\(targetWidth)x\(targetHeight) HEVC Main10 • bitrate=\(videoBitrate)")

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
                DiagnosticsLogger.shared.log("Real-CUGAN first frame decoded • pixel=\(CVPixelBufferGetWidth(decodedFrame))x\(CVPixelBufferGetHeight(decodedFrame)) • tiled prediction begin")
            }

            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let upscaled = try upscaleNative2x(
                decodedFrame,
                model: model,
                alpha: alpha,
                frameNumber: frameNumber
            )
            inferenceSeconds += CFAbsoluteTimeGetCurrent() - inferenceStart

            if frameNumber == 1 {
                DiagnosticsLogger.shared.log("Real-CUGAN first tiled frame complete • output=\(CVPixelBufferGetWidth(upscaled))x\(CVPixelBufferGetHeight(upscaled))")
            }

            let encodeStart = CFAbsoluteTimeGetCurrent()
            try await append10Bit(upscaled, at: pts, input: input, adaptor: adaptor, pool: writerPool)
            encodeSeconds += CFAbsoluteTimeGetCurrent() - encodeStart
            frames += 1

            if frames == 1 || frames % 10 == 0 {
                DiagnosticsLogger.shared.log("Real-CUGAN frame \(frames) complete • \(tilesPerFrame) tiles/frame • inference=\(String(format: "%.1f", inferenceSeconds * 1000.0 / Double(frames)))ms/frame • thermal=\(currentThermalStateName())")
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
            if frames % 3 == 0 {
                let frac = min(max(CMTimeGetSeconds(pts) / seconds, 0), 1)
                let fps = Double(frames) / max(CFAbsoluteTimeGetCurrent() - passStart, 0.001)
                let remaining = fps > 0 ? (seconds * 60.0 - Double(frames)) / fps : 0
                progress(frac, "Pass 3/3 • Real-CUGAN native 2× • \(frames) frames • ETA \(formatDuration(remaining))")
                await Task.yield()
            }
        }

        if reader.status == .failed { throw ProcessorError.reader(reader.error?.localizedDescription ?? "Real-CUGAN decode failed") }
        guard frames > 0 else { throw ProcessorError.conversionFailed("Real-CUGAN received zero decoded frames") }
        DiagnosticsLogger.shared.log("Real-CUGAN tiled inference complete • frames=\(frames) • finishing writer")
        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in writer.finishWriting { continuation.resume() } }
        guard writer.status == .completed else { throw ProcessorError.writer(writer.error?.localizedDescription ?? "native 2x finish failed") }
        DiagnosticsLogger.shared.log("Real-CUGAN writer completed • native 2x=\(targetWidth)x\(targetHeight) • frames=\(frames)")
        return outURL
    }

    private func loadModel() throws -> MLModel {
        guard let url = Bundle.main.url(forResource: "RealCUGAN2xNoise3_Tile512", withExtension: "mlmodelc") else {
            throw ProcessorError.conversionFailed("Bundled tiled Real-CUGAN model is missing")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.allowLowPrecisionAccumulationOnGPU = true
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private func preparePools(targetWidth: Int, targetHeight: Int) throws {
        if tileInputPool == nil {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: tileSize,
                kCVPixelBufferHeightKey as String: tileSize,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
            var pool: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs as CFDictionary, &pool) == kCVReturnSuccess,
                  let pool else {
                throw ProcessorError.conversionFailed("could not create Real-CUGAN 512px tile pool")
            }
            tileInputPool = pool
        }

        let stitchedAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: targetWidth,
            kCVPixelBufferHeightKey as String: targetHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        var stitched: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, stitchedAttrs as CFDictionary, &stitched) == kCVReturnSuccess,
              let stitched else {
            throw ProcessorError.conversionFailed("could not create native 2x Real-CUGAN frame pool")
        }
        stitchedFramePool = stitched
    }

    private func upscaleNative2x(
        _ source: CVPixelBuffer,
        model: MLModel,
        alpha: MLMultiArray,
        frameNumber: Int
    ) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        let targetWidth = sourceWidth * 2
        let targetHeight = sourceHeight * 2

        guard let tileInputPool, let stitchedFramePool else {
            throw ProcessorError.conversionFailed("Real-CUGAN tile pools are unavailable")
        }
        var stitched: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, stitchedFramePool, &stitched) == kCVReturnSuccess,
              let stitched else {
            throw ProcessorError.conversionFailed("could not allocate native 2x stitched frame")
        }

        let sourceImage = CIImage(cvPixelBuffer: source).clampedToExtent()
        var tileIndex = 0
        let totalTiles = Int(ceil(Double(sourceWidth) / Double(tileStep))) * Int(ceil(Double(sourceHeight) / Double(tileStep)))

        var y = 0
        while y < sourceHeight {
            let coreHeight = min(tileStep, sourceHeight - y)
            var x = 0
            while x < sourceWidth {
                try Task.checkCancellation()
                let coreWidth = min(tileStep, sourceWidth - x)
                tileIndex += 1

                var tileBuffer: CVPixelBuffer?
                guard CVPixelBufferPoolCreatePixelBuffer(nil, tileInputPool, &tileBuffer) == kCVReturnSuccess,
                      let tileBuffer else {
                    throw ProcessorError.conversionFailed("could not allocate Real-CUGAN input tile")
                }

                let inputRect = CGRect(
                    x: x - tileOverlap,
                    y: y - tileOverlap,
                    width: tileSize,
                    height: tileSize
                )
                let tileImage = sourceImage
                    .cropped(to: inputRect)
                    .transformed(by: CGAffineTransform(
                        translationX: -inputRect.origin.x,
                        y: -inputRect.origin.y
                    ))
                ciContext.render(
                    tileImage,
                    to: tileBuffer,
                    bounds: CGRect(x: 0, y: 0, width: tileSize, height: tileSize),
                    colorSpace: colorSpace
                )

                try autoreleasepool {
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "image": MLFeatureValue(pixelBuffer: tileBuffer),
                        "alpha": MLFeatureValue(multiArray: alpha)
                    ])
                    let prediction = try model.prediction(from: provider)
                    guard let modelOutput = prediction.featureValue(for: "output")?.imageBufferValue else {
                        throw ProcessorError.conversionFailed("Real-CUGAN tile did not return an image buffer")
                    }

                    let outputImage = CIImage(cvPixelBuffer: modelOutput)
                    let cropOrigin = tileOverlap * 2
                    let cropRect = CGRect(
                        x: cropOrigin,
                        y: cropOrigin,
                        width: coreWidth * 2,
                        height: coreHeight * 2
                    )
                    let destinationRect = CGRect(
                        x: x * 2,
                        y: y * 2,
                        width: coreWidth * 2,
                        height: coreHeight * 2
                    )
                    let translated = outputImage
                        .cropped(to: cropRect)
                        .transformed(by: CGAffineTransform(
                            translationX: destinationRect.origin.x - cropRect.origin.x,
                            y: destinationRect.origin.y - cropRect.origin.y
                        ))
                    ciContext.render(
                        translated,
                        to: stitched,
                        bounds: destinationRect,
                        colorSpace: colorSpace
                    )
                }

                if frameNumber == 1 && (tileIndex == 1 || tileIndex == totalTiles) {
                    DiagnosticsLogger.shared.log("Real-CUGAN first-frame tile \(tileIndex)/\(totalTiles) complete • core=\(coreWidth)x\(coreHeight)")
                }
                x += tileStep
            }
            y += tileStep
        }

        guard CVPixelBufferGetWidth(stitched) == targetWidth,
              CVPixelBufferGetHeight(stitched) == targetHeight else {
            throw ProcessorError.conversionFailed("Real-CUGAN stitched frame size mismatch")
        }
        return stitched
    }

    private func append10Bit(_ source: CVPixelBuffer, at time: CMTime, input: AVAssetWriterInput, adaptor: AVAssetWriterInputPixelBufferAdaptor, pool: CVPixelBufferPool) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        var destination: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess, let destination else {
            throw ProcessorError.conversionFailed("could not allocate native 2x P010 frame")
        }
        if outputTransferSession == nil {
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) == noErr, let session else {
                throw ProcessorError.conversionFailed("could not create native 2x pixel transfer session")
            }
            outputTransferSession = session
        }
        guard let outputTransferSession,
              VTPixelTransferSessionTransferImage(outputTransferSession, from: source, to: destination) == noErr else {
            throw ProcessorError.conversionFailed("native 2x BGRA→P010 conversion failed")
        }
        guard adaptor.append(destination, withPresentationTime: time) else {
            throw ProcessorError.writer("failed appending native 2x frame")
        }
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
