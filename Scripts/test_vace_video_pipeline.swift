import AVFoundation
import CoreVideo
import Foundation

private struct SolidTestMaskProvider: VACEFrameMaskProvider {
    let width: Int
    let height: Int
    func mask(for frameIndex: Int, presentationTime: CMTime, source: VACERGBAFrame) async throws -> VACEMaskPlane {
        _ = frameIndex
        _ = presentationTime
        guard source.width == width, source.height == height else {
            throw VACEVideoPipelineError.maskDimensionsMismatch(frame: frameIndex)
        }
        var pixels = [UInt8](repeating: 0, count: width * height)
        for y in 16..<32 {
            for x in 24..<40 { pixels[y * width + x] = 255 }
        }
        return try VACEMaskPlane(width: width, height: height, pixels: pixels)
    }
}

private struct SolidTestReconstructionEngine: VACERegionReconstructionEngine {
    func reconstruct(_ request: VACERegionReconstructionRequest) async throws -> [VACERegionFrameOutput] {
        var output: [VACERegionFrameOutput] = []
        for input in request.frames where !input.isPadding {
            var bytes = input.sourceCrop.bytes
            for pixel in 0..<(input.sourceCrop.width * input.sourceCrop.height) {
                let offset = pixel * 4
                bytes[offset] = 245
                bytes[offset + 1] = 12
                bytes[offset + 2] = 18
                // Original alpha is preserved by the production compositor.
            }
            let patch = try VACERGBAFrame(width: input.sourceCrop.width, height: input.sourceCrop.height, bytes: bytes)
            output.append(VACERegionFrameOutput(windowPosition: input.windowPosition, reconstructedCrop: patch))
        }
        return output
    }
}

@main
enum VACEVideoPipelineIntegrationTest {
    static func main() async throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("vace-video-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let sourceURL = directory.appendingPathComponent("source.mov")
        let destinationURL = directory.appendingPathComponent("reconstructed.mov")
        try await writeSyntheticVideo(to: sourceURL, frameCount: 7, width: 64, height: 64)

        let pipeline = VACESelectiveVideoPipeline(
            maskProvider: SolidTestMaskProvider(width: 64, height: 64),
            reconstructionEngine: SolidTestReconstructionEngine(),
            options: VACESelectiveVideoOptions(
                temporalWindowLength: 5, temporalOverlap: 2,
                maximumCropWidth: 832, maximumCropHeight: 480,
                contextPaddingPixels: 32, minimumActivePixels: 4,
                tileOverlapPixels: 64, maximumCropCount: 32,
                preserveSourceAudio: false, averageBitRate: 2_000_000
            )
        )
        try await pipeline.process(sourceURL: sourceURL, destinationURL: destinationURL, prompt: "synthetic red patch")
        let sourcePixels = try await readFirstFrame(from: sourceURL)
        let outputPixels = try await readFirstFrame(from: destinationURL)
        try checkFrameCount(destinationURL, expected: 7)

        let outside = (5, 5)
        let sourceOutside = rgb(sourcePixels, x: outside.0, y: outside.1, width: 64)
        let outputOutside = rgb(outputPixels, x: outside.0, y: outside.1, width: 64)
        check(near(sourceOutside.0, outputOutside.0, tolerance: 18)
              && near(sourceOutside.1, outputOutside.1, tolerance: 18)
              && near(sourceOutside.2, outputOutside.2, tolerance: 18),
              "pixel outside replacement mask should remain visually unchanged after codec round-trip")

        let inside = rgb(outputPixels, x: 30, y: 24, width: 64)
        check(inside.0 > 180 && inside.1 < 80 && inside.2 < 80,
              "pixel inside replacement mask should come from reconstruction patch")
        print("PASS: VACE video integration decoded 7 frames, masked patch composite, encoded at source dimensions, and preserved unmasked appearance")
    }

    static func check(_ condition: Bool, _ message: String) {
        if !condition { fatalError("FAIL: \(message)") }
    }

    static func near(_ lhs: UInt8, _ rhs: UInt8, tolerance: Int) -> Bool {
        abs(Int(lhs) - Int(rhs)) <= tolerance
    }

    static func rgb(_ frame: VACERGBAFrame, x: Int, y: Int, width: Int) -> (UInt8, UInt8, UInt8) {
        let offset = (y * width + x) * 4
        return (frame.bytes[offset], frame.bytes[offset + 1], frame.bytes[offset + 2])
    }

    static func writeSyntheticVideo(to url: URL, frameCount: Int, width: Int, height: Int) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 2_000_000,
                AVVideoExpectedSourceFrameRateKey: 10
            ]
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                            sourcePixelBufferAttributes: attributes)
        guard writer.canAdd(input) else { throw VACEVideoPipelineError.couldNotCreateWriter }
        writer.add(input)
        guard writer.startWriting() else {
            throw VACEVideoPipelineError.writerStartFailed(writer.error?.localizedDescription ?? "test writer start failed")
        }
        writer.startSession(atSourceTime: .zero)
        for index in 0..<frameCount {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
            let pixelBuffer = try solidBGRA(width: width, height: height, red: 20, green: 90, blue: 150)
            let time = CMTime(value: Int64(index), timescale: 10)
            guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                throw VACEVideoPipelineError.writerAppendFailed(writer.error?.localizedDescription ?? "test append failed")
            }
        }
        input.markAsFinished()
        await withCheckedContinuation { continuation in writer.finishWriting { continuation.resume() } }
        guard writer.status == .completed else {
            throw VACEVideoPipelineError.writerAppendFailed(writer.error?.localizedDescription ?? "test writer finish failed")
        }
    }

    static func solidBGRA(width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8) throws -> CVPixelBuffer {
        var optionalBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                         [kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary] as CFDictionary,
                                         &optionalBuffer)
        guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
            throw VACEVideoPipelineError.invalidPixelBuffer
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw VACEVideoPipelineError.invalidPixelBuffer }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * rowBytes + x * 4
                bytes[offset] = blue; bytes[offset + 1] = green; bytes[offset + 2] = red; bytes[offset + 3] = 255
            }
        }
        return buffer
    }

    static func readFirstFrame(from url: URL) async throws -> VACERGBAFrame {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw VACEVideoPipelineError.noVideoTrack }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        guard reader.canAdd(output) else { throw VACEVideoPipelineError.couldNotCreateReader }
        reader.add(output)
        guard reader.startReading() else { throw VACEVideoPipelineError.readerStartFailed(reader.error?.localizedDescription ?? "test read failed") }
        guard let sample = output.copyNextSampleBuffer(), let buffer = CMSampleBufferGetImageBuffer(sample) else {
            throw VACEVideoPipelineError.emptyVideo
        }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw VACEVideoPipelineError.invalidPixelBuffer }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let source = base.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let s = y * rowBytes + x * 4, d = (y * width + x) * 4
                rgba[d] = source[s + 2]; rgba[d + 1] = source[s + 1]
                rgba[d + 2] = source[s]; rgba[d + 3] = source[s + 3]
            }
        }
        return try VACERGBAFrame(width: width, height: height, bytes: rgba)
    }

    static func checkFrameCount(_ url: URL, expected: Int) throws {
        let asset = AVURLAsset(url: url)
        let videoTracks = asset.tracks(withMediaType: .video)
        guard let track = videoTracks.first else { throw VACEVideoPipelineError.noVideoTrack }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        guard reader.canAdd(output) else { throw VACEVideoPipelineError.couldNotCreateReader }
        reader.add(output)
        guard reader.startReading() else { throw VACEVideoPipelineError.readerStartFailed(reader.error?.localizedDescription ?? "count reader start failed") }
        var count = 0
        while output.copyNextSampleBuffer() != nil { count += 1 }
        check(count == expected, "expected \(expected) output frames but decoded \(count)")
    }
}
