import AVFoundation
import CoreVideo
import Darwin
import Foundation

final class DiagnosticsLogger {
    static let shared = DiagnosticsLogger()
    func log(_ text: String) { print(text) }
}

enum CrashCapture {
    static func step(_ text: String) { print(text) }
}

enum ProcessorError: Error {
    case missingVideoTrack
    case reader(String)
    case writer(String)
    case conversionFailed(String)
    case noOutput
}

struct RenderPerformanceSnapshot {
    var availableMemoryMB = 4000.0
    var effectiveMemoryMB = 4000.0
    var thermalAndMode = "test"
}

func currentRenderPerformanceSnapshot() -> RenderPerformanceSnapshot { RenderPerformanceSnapshot() }
func adaptiveFrameBoundaryPacing(_: RenderPerformanceSnapshot? = nil) async throws {}

func require(_ condition: Bool, _ message: String) {
    if !condition { fatalError(message) }
}

let modelDirectory = URL(fileURLWithPath: CommandLine.arguments[1])
AnimeRestoreFiles.cacheDirectoryOverride = modelDirectory
AnimeRestoreRegistry.installDefault()

let source = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("anime-restore-720p.mov")
try? FileManager.default.removeItem(at: source)
let writer = try AVAssetWriter(outputURL: source, fileType: .mov)
let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: 1280,
    AVVideoHeightKey: 720
])
let adaptor = AVAssetWriterInputPixelBufferAdaptor(input: input, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String: 1280,
    kCVPixelBufferHeightKey as String: 720
])
writer.add(input)
writer.startWriting()
writer.startSession(atSourceTime: .zero)
for frame in 0..<2 {
    while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.01) }
    guard let pool = adaptor.pixelBufferPool else { fatalError("synthetic pixel pool") }
    var buffer: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
    if let buffer {
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, frame == 0 ? 40 : 180, CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer))
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        require(adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 30)), "append synthetic frame")
    }
}
input.markAsFinished()
await writer.finishWriting()
require(writer.status == .completed, "synthetic 720p clip \(writer.error?.localizedDescription ?? "")")

let result = try await AnimeRestorePass(strength: 0.8, quality: .apisr).run(sourceURL: source) { _, message in
    print("progress \(message)")
}
require(result.frames == 2, "restore pass finished \(result.frames) frames")
print("ANIME_RESTORE_PASS \(result.frames) frames \(String(format: "%.1f", result.millisecondsPerFrame)) ms")
