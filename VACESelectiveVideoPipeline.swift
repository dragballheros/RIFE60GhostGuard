import AVFoundation
import CoreVideo
import Foundation

enum VACEVideoPipelineError: LocalizedError {
    case noVideoTrack
    case couldNotCreateReader
    case couldNotCreateWriter
    case readerStartFailed(String)
    case writerStartFailed(String)
    case invalidPixelBuffer
    case maskDimensionsMismatch(frame: Int)
    case reconstructorReturnedNoFrame(frame: Int, crop: Int)
    case reconstructorReturnedInvalidDimensions(frame: Int, crop: Int)
    case writerAppendFailed(String)
    case outputExportFailed(String)
    case emptyVideo
    case unsupportedOutputURL
    case invalidWindowOptions

    var errorDescription: String? {
        switch self {
        case .noVideoTrack: return "The selected source has no readable video track."
        case .couldNotCreateReader: return "Could not create an AVAssetReader for the source video."
        case .couldNotCreateWriter: return "Could not create an AVAssetWriter for the temporary output."
        case .readerStartFailed(let message): return "Video reader could not start: \(message)"
        case .writerStartFailed(let message): return "Video writer could not start: \(message)"
        case .invalidPixelBuffer: return "Video decoder returned a pixel buffer that could not be read."
        case .maskDimensionsMismatch(let frame): return "Replacement mask dimensions do not match source frame \(frame)."
        case .reconstructorReturnedNoFrame(let frame, let crop): return "The local VACE reconstruction engine returned no output for frame \(frame), crop \(crop)."
        case .reconstructorReturnedInvalidDimensions(let frame, let crop): return "The local VACE engine returned invalid output dimensions for frame \(frame), crop \(crop)."
        case .writerAppendFailed(let message): return "Could not append a reconstructed video frame: \(message)"
        case .outputExportFailed(let message): return "Could not mux the finished video: \(message)"
        case .emptyVideo: return "The source video contains no decodable frames."
        case .unsupportedOutputURL: return "Choose a destination ending in .mov."
        case .invalidWindowOptions: return "VACE temporal window length and overlap must be valid."
        }
    }
}

struct VACEVideoFrameRecord {
    let index: Int
    let presentationTime: CMTime
    let source: VACERGBAFrame
    let mask: VACEMaskPlane
}

struct VACERegionFrameInput {
    /// Position within the fixed-size temporal model window.
    let windowPosition: Int
    /// Index in the original source video. Padded tail inputs repeat the last
    /// source index and are marked isPadding=true.
    let sourceFrameIndex: Int
    let sourceCrop: VACERGBAFrame
    let maskCrop: VACEMaskPlane
    let isPadding: Bool
}

struct VACERegionReconstructionRequest {
    let prompt: String
    let negativePrompt: String
    let cropPlan: VACERegionCropPlan
    let frames: [VACERegionFrameInput]
    let temporalWindowLength: Int
    let validFrameCount: Int
}

struct VACERegionFrameOutput {
    let windowPosition: Int
    /// RGB content in source crop coordinates and at source crop dimensions.
    let reconstructedCrop: VACERGBAFrame
}

/// Adapter seam for the actual local Core ML VACE inference engine. Implementors
/// must return one generated source-sized patch per valid input window position.
/// All model work stays on-device; this protocol defines no network fallback.
protocol VACERegionReconstructionEngine {
    func reconstruct(_ request: VACERegionReconstructionRequest) async throws -> [VACERegionFrameOutput]
}

/// Supplies one full-frame mask for each decoded source frame. A concrete
/// provider may use a manually drawn mask, imported mask sequence, local
/// segmentation, or motion propagation. Nil masks are not silently interpreted
/// as editable; provider must return a matching all-zero plane to preserve frame.
protocol VACEFrameMaskProvider {
    func mask(for frameIndex: Int, presentationTime: CMTime, source: VACERGBAFrame) async throws -> VACEMaskPlane
}

struct VACESelectiveVideoOptions {
    var temporalWindowLength: Int = 5
    var temporalOverlap: Int = 2
    var maximumCropWidth: Int = 832
    var maximumCropHeight: Int = 480
    var contextPaddingPixels: Int = 32
    var minimumActivePixels: Int = 4
    var tileOverlapPixels: Int = 64
    var maximumCropCount: Int = 32
    var preserveSourceAudio: Bool = true
    var averageBitRate: Int = 200_000_000
    var writerFileType: AVFileType = .mov
}

/// Streams original frames through bounded temporal windows. It does not buffer
/// the whole movie. The caller injects a local mask provider and a real Core ML
/// reconstruction engine; if that engine is absent or fails to return patches,
/// export fails instead of silently generating an unchanged "success" result.
final class VACESelectiveVideoPipeline {
    typealias Progress = (_ fraction: Double, _ message: String) -> Void

    private let maskProvider: VACEFrameMaskProvider
    private let reconstructionEngine: VACERegionReconstructionEngine
    private let options: VACESelectiveVideoOptions

    init(maskProvider: VACEFrameMaskProvider,
         reconstructionEngine: VACERegionReconstructionEngine,
         options: VACESelectiveVideoOptions = VACESelectiveVideoOptions()) {
        self.maskProvider = maskProvider
        self.reconstructionEngine = reconstructionEngine
        self.options = options
    }

    func process(sourceURL: URL, destinationURL: URL, prompt: String,
                 negativePrompt: String = "",
                 progress: Progress? = nil) async throws {
        guard destinationURL.pathExtension.lowercased() == "mov" else {
            throw VACEVideoPipelineError.unsupportedOutputURL
        }
        guard options.temporalWindowLength == 5, options.temporalOverlap >= 0,
              options.temporalOverlap < options.temporalWindowLength else {
            // The current adapter contract explicitly targets Wan VACE's common
            // five-frame window; expanding it is a separate model compatibility test.
            throw VACEVideoPipelineError.invalidWindowOptions
        }
        try Task.checkCancellation()

        let asset = AVURLAsset(url: sourceURL)
        let sourceVideoTrack = try await firstVideoTrack(in: asset)
        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: sourceVideoTrack,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        readerOutput.alwaysCopiesSampleData = false
        guard reader.canAdd(readerOutput) else { throw VACEVideoPipelineError.couldNotCreateReader }
        reader.add(readerOutput)
        guard reader.startReading() else {
            throw VACEVideoPipelineError.readerStartFailed(reader.error?.localizedDescription ?? "unknown reader error")
        }
        defer {
            if reader.status == .reading { reader.cancelReading() }
        }

        guard let firstSample = readerOutput.copyNextSampleBuffer() else {
            throw VACEVideoPipelineError.emptyVideo
        }
        guard let firstBuffer = CMSampleBufferGetImageBuffer(firstSample) else {
            throw VACEVideoPipelineError.invalidPixelBuffer
        }
        let width = CVPixelBufferGetWidth(firstBuffer)
        let height = CVPixelBufferGetHeight(firstBuffer)
        let firstTime = CMSampleBufferGetPresentationTimeStamp(firstSample)
        let firstRGBA = try Self.rgbaFrame(fromBGRA: firstBuffer)
        let firstMask = try await maskProvider.mask(for: 0, presentationTime: firstTime, source: firstRGBA)
        guard firstMask.width == width, firstMask.height == height else {
            throw VACEVideoPipelineError.maskDimensionsMismatch(frame: 0)
        }
        var pending = [VACEVideoFrameRecord(index: 0, presentationTime: firstTime, source: firstRGBA, mask: firstMask)]

        let temporaryDirectory = destinationURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        let videoOnlyURL = temporaryDirectory.appendingPathComponent("vace-video-\(UUID().uuidString).mov")
        let muxedURL = temporaryDirectory.appendingPathComponent("vace-muxed-\(UUID().uuidString).mov")
        defer {
            try? FileManager.default.removeItem(at: videoOnlyURL)
            try? FileManager.default.removeItem(at: muxedURL)
        }

        let writer = try AVAssetWriter(outputURL: videoOnlyURL, fileType: options.writerFileType)
        defer {
            if writer.status == .writing || writer.status == .unknown { writer.cancelWriting() }
        }
        let frameRate = max(1, Int(sourceVideoTrack.nominalFrameRate.rounded()))
        let estimatedFrameCount: Int? = {
            let seconds = asset.duration.seconds
            guard seconds.isFinite, seconds > 0 else { return nil }
            return max(1, Int(ceil(seconds * Double(frameRate))))
        }()
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: max(1_000_000, options.averageBitRate),
                AVVideoExpectedSourceFrameRateKey: frameRate,
                AVVideoMaxKeyFrameIntervalDurationKey: 2
            ]
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        writerInput.expectsMediaDataInRealTime = false
        writerInput.transform = sourceVideoTrack.preferredTransform
        let pixelAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: pixelAttributes
        )
        guard writer.canAdd(writerInput) else { throw VACEVideoPipelineError.couldNotCreateWriter }
        writer.add(writerInput)
        guard writer.startWriting() else {
            throw VACEVideoPipelineError.writerStartFailed(writer.error?.localizedDescription ?? "unknown writer error")
        }
        writer.startSession(atSourceTime: firstTime)

        let stride = options.temporalWindowLength - options.temporalOverlap
        var framesRead = 1
        var framesWritten = 0
        func append(_ record: VACEVideoFrameRecord) async throws {
            try Task.checkCancellation()
            while !writerInput.isReadyForMoreMediaData {
                if writer.status == .failed {
                    throw VACEVideoPipelineError.writerAppendFailed(writer.error?.localizedDescription ?? "writer failed")
                }
                try await Task.sleep(nanoseconds: 2_000_000)
            }
            let pixelBuffer = try Self.bgraPixelBuffer(from: record.source)
            guard adaptor.append(pixelBuffer, withPresentationTime: record.presentationTime) else {
                throw VACEVideoPipelineError.writerAppendFailed(writer.error?.localizedDescription ?? "pixel-buffer adaptor rejected frame")
            }
            framesWritten += 1
            progress?(0, "Reconstructed and encoded frame \(framesWritten)")
        }

        func runWindow(_ realRecords: [VACEVideoFrameRecord], outputCount: Int) async throws {
            guard !realRecords.isEmpty, outputCount > 0, outputCount <= realRecords.count else { return }
            var windowRecords = realRecords
            if windowRecords.count < options.temporalWindowLength, let last = windowRecords.last {
                windowRecords.append(contentsOf: repeatElement(last, count: options.temporalWindowLength - windowRecords.count))
            }
            let unionPlans = try VACESelectiveRegionPlanner.plan(
                masks: windowRecords.map(\.mask),
                options: VACERegionPlanningOptions(
                    contextPaddingPixels: options.contextPaddingPixels,
                    cropAlignment: 16,
                    minimumActivePixels: options.minimumActivePixels,
                    maximumCropWidth: options.maximumCropWidth,
                    maximumCropHeight: options.maximumCropHeight,
                    tileOverlapPixels: options.tileOverlapPixels,
                    maximumCropCount: options.maximumCropCount,
                    foregroundThreshold: 0
                )
            )
            var processedFrames = Dictionary(uniqueKeysWithValues: windowRecords.prefix(realRecords.count).map { ($0.index, $0.source) })

            for cropPlan in unionPlans {
                try Task.checkCancellation()
                var inputs: [VACERegionFrameInput] = []
                inputs.reserveCapacity(options.temporalWindowLength)
                for (position, record) in windowRecords.enumerated() {
                    let frameCrop = try record.source.cropped(to: cropPlan.cropRect)
                    let maskCrop = try record.mask.cropped(to: cropPlan.cropRect)
                    inputs.append(VACERegionFrameInput(
                        windowPosition: position,
                        sourceFrameIndex: record.index,
                        sourceCrop: frameCrop,
                        maskCrop: maskCrop,
                        isPadding: position >= realRecords.count
                    ))
                }
                let request = VACERegionReconstructionRequest(
                    prompt: prompt,
                    negativePrompt: negativePrompt,
                    cropPlan: cropPlan,
                    frames: inputs,
                    temporalWindowLength: options.temporalWindowLength,
                    validFrameCount: realRecords.count
                )
                let outputs = try await reconstructionEngine.reconstruct(request)
                var outputsByPosition: [Int: VACERGBAFrame] = [:]
                for output in outputs {
                    guard output.windowPosition >= 0, output.windowPosition < realRecords.count else { continue }
                    guard output.reconstructedCrop.width == cropPlan.cropRect.width,
                          output.reconstructedCrop.height == cropPlan.cropRect.height else {
                        throw VACEVideoPipelineError.reconstructorReturnedInvalidDimensions(
                            frame: realRecords[output.windowPosition].index, crop: cropPlan.regionID
                        )
                    }
                    outputsByPosition[output.windowPosition] = output.reconstructedCrop
                }
                for position in 0..<realRecords.count {
                    guard let patchFrame = outputsByPosition[position] else {
                        let record = realRecords[position]
                        if inputs[position].maskCrop.activePixelCount == 0 { continue }
                        throw VACEVideoPipelineError.reconstructorReturnedNoFrame(
                            frame: record.index, crop: cropPlan.regionID
                        )
                    }
                    let record = realRecords[position]
                    let current = processedFrames[record.index] ?? record.source
                    let composite = try VACESelectiveCompositor.composite(
                        original: current,
                        mask: record.mask,
                        patches: [VACEReconstructedPatch(sourceRect: cropPlan.cropRect, frame: patchFrame)]
                    )
                    processedFrames[record.index] = composite
                }
            }

            for record in realRecords.prefix(outputCount) {
                try await append(
                    VACEVideoFrameRecord(index: record.index, presentationTime: record.presentationTime,
                                         source: processedFrames[record.index] ?? record.source, mask: record.mask)
                )
            }
        }

        while let sample = readerOutput.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
                throw VACEVideoPipelineError.invalidPixelBuffer
            }
            let rgba = try Self.rgbaFrame(fromBGRA: pixelBuffer)
            guard rgba.width == width, rgba.height == height else {
                throw VACEVideoPipelineError.maskDimensionsMismatch(frame: framesRead)
            }
            let timestamp = CMSampleBufferGetPresentationTimeStamp(sample)
            let mask = try await maskProvider.mask(for: framesRead, presentationTime: timestamp, source: rgba)
            guard mask.width == width, mask.height == height else {
                throw VACEVideoPipelineError.maskDimensionsMismatch(frame: framesRead)
            }
            pending.append(VACEVideoFrameRecord(index: framesRead, presentationTime: timestamp, source: rgba, mask: mask))
            framesRead += 1

            // One-frame lookahead lets us stream windows while keeping at most
            // windowLength+1 source frames in memory. Each frame is emitted once.
            while pending.count > options.temporalWindowLength {
                let realWindow = Array(pending.prefix(options.temporalWindowLength))
                try await runWindow(realWindow, outputCount: stride)
                pending.removeFirst(stride)
            }
        }
        guard reader.status == .completed else {
            throw VACEVideoPipelineError.readerStartFailed(reader.error?.localizedDescription ?? "reader stopped before EOF")
        }
        if !pending.isEmpty {
            try await runWindow(pending, outputCount: pending.count)
        }
        guard framesWritten > 0 else { throw VACEVideoPipelineError.emptyVideo }

        writerInput.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw VACEVideoPipelineError.writerAppendFailed(writer.error?.localizedDescription ?? "finishWriting failed")
        }
        try Task.checkCancellation()

        if options.preserveSourceAudio {
            try await muxSourceAudio(source: asset, videoURL: videoOnlyURL, destinationURL: muxedURL)
            try replaceDestination(from: muxedURL, to: destinationURL)
        } else {
            try replaceDestination(from: videoOnlyURL, to: destinationURL)
        }
        progress?(1, "VACE selective reconstruction export complete")
    }

    private func firstVideoTrack(in asset: AVAsset) async throws -> AVAssetTrack {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw VACEVideoPipelineError.noVideoTrack }
        return track
    }

    private static func rgbaFrame(fromBGRA buffer: CVPixelBuffer) throws -> VACERGBAFrame {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw VACEVideoPipelineError.invalidPixelBuffer }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let sourceRowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let (pixelCount, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, pixelCount <= 16_777_216 else {
            throw VACERegionPlanningError.imageTooLarge(pixelCount: overflow ? Int.max : pixelCount)
        }
        var rgba = [UInt8](repeating: 0, count: pixelCount * 4)
        let source = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            let sourceRow = source.advanced(by: y * sourceRowBytes)
            let destinationRow = y * width * 4
            for x in 0..<width {
                let s = x * 4, d = destinationRow + s
                rgba[d] = sourceRow[s + 2]
                rgba[d + 1] = sourceRow[s + 1]
                rgba[d + 2] = sourceRow[s]
                rgba[d + 3] = sourceRow[s + 3]
            }
        }
        return try VACERGBAFrame(width: width, height: height, bytes: rgba)
    }

    private static func bgraPixelBuffer(from frame: VACERGBAFrame) throws -> CVPixelBuffer {
        var optionalBuffer: CVPixelBuffer?
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: frame.width,
            kCVPixelBufferHeightKey as String: frame.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, frame.width, frame.height,
                                         kCVPixelFormatType_32BGRA, attributes as CFDictionary, &optionalBuffer)
        guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
            throw VACEVideoPipelineError.invalidPixelBuffer
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw VACEVideoPipelineError.invalidPixelBuffer }
        let destination = base.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<frame.height {
            let destinationRow = destination.advanced(by: y * rowBytes)
            let sourceRow = y * frame.width * 4
            for x in 0..<frame.width {
                let s = sourceRow + x * 4, d = x * 4
                destinationRow[d] = frame.bytes[s + 2]
                destinationRow[d + 1] = frame.bytes[s + 1]
                destinationRow[d + 2] = frame.bytes[s]
                destinationRow[d + 3] = frame.bytes[s + 3]
            }
        }
        return buffer
    }

    private func muxSourceAudio(source: AVAsset, videoURL: URL, destinationURL: URL) async throws {
        let editedVideo = AVURLAsset(url: videoURL)
        let composition = AVMutableComposition()
        let outputVideoTracks = try await editedVideo.loadTracks(withMediaType: .video)
        guard let editedVideoTrack = outputVideoTracks.first,
              let compositionVideoTrack = composition.addMutableTrack(
                withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw VACEVideoPipelineError.noVideoTrack
        }
        let videoRange = try await editedVideoTrack.load(.timeRange)
        try compositionVideoTrack.insertTimeRange(videoRange, of: editedVideoTrack, at: videoRange.start)
        compositionVideoTrack.preferredTransform = try await editedVideoTrack.load(.preferredTransform)

        let audioTracks = try await source.loadTracks(withMediaType: .audio)
        for audio in audioTracks {
            guard let compositionAudioTrack = composition.addMutableTrack(
                withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else { continue }
            let audioRange = try await audio.load(.timeRange)
            let duration = CMTimeMinimum(audioRange.duration, videoRange.duration)
            guard duration.isNumeric, duration > .zero else { continue }
            try compositionAudioTrack.insertTimeRange(
                CMTimeRange(start: audioRange.start, duration: duration),
                of: audio, at: audioRange.start
            )
        }
        guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
            throw VACEVideoPipelineError.outputExportFailed("Could not create passthrough export session.")
        }
        try? FileManager.default.removeItem(at: destinationURL)
        exporter.outputURL = destinationURL
        exporter.outputFileType = .mov
        exporter.shouldOptimizeForNetworkUse = true
        await withCheckedContinuation { continuation in
            exporter.exportAsynchronously { continuation.resume() }
        }
        guard exporter.status == .completed else {
            throw VACEVideoPipelineError.outputExportFailed(exporter.error?.localizedDescription ?? "unknown exporter error")
        }
    }

    private func replaceDestination(from source: URL, to destination: URL) throws {
        let manager = FileManager.default
        if manager.fileExists(atPath: destination.path) { try manager.removeItem(at: destination) }
        try manager.moveItem(at: source, to: destination)
    }
}
