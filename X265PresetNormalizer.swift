import Foundation
import UIKit

/// Re-encodes one imported video with the fixed DJATOM x265 preset before RIFE.
/// The encode writes a file and releases its frames before returning, so the
/// RIFE and Real-CUGAN stages do not share those buffers.
enum X265PresetNormalizer {
    static func normalize(
        source: URL,
        progress: @escaping @Sendable (Double, String) -> Void
    ) async throws -> URL {
        let digest = try SourceFileIdentity.sha256(of: source)
        let preset = String(cString: rife60_x265_preset_params())
        let build = rife60_x265_build_number()
        DiagnosticsLogger.shared.log("x265 preset stage • build=\(build) • before RIFE • source=\(source.lastPathComponent)")
        if build != 198 {
            throw ProcessorError.conversionFailed("The linked x265 build is \(build); the preset requires DJATOM build 198.")
        }
        if let cached = existingOutput(sourceSHA256: digest, preset: preset) {
            DiagnosticsLogger.shared.log("x265 preset reused • \(cached.lastPathComponent)")
            progress(1, "x265 preset already finished for this source")
            return cached
        }
        let folder = X265PresetCache.directory()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let partial = X265PresetCache.partialURL(sourceSHA256: digest)
        let final = X265PresetCache.outputURL(sourceSHA256: digest)
        try? FileManager.default.removeItem(at: partial)
        try? FileManager.default.removeItem(at: final)
        try? FileManager.default.removeItem(at: X265PresetCache.manifestURL(sourceSHA256: digest))

        let control = EncodeControl()
        let taskID = await MainActor.run {
            UIApplication.shared.beginBackgroundTask(withName: "x265-preset") {
                control.cancel()
            }
        }
        defer {
            let ending = taskID
            Task { @MainActor in
                if ending != .invalid { UIApplication.shared.endBackgroundTask(ending) }
            }
        }

        control.progress.handler = { frame, total in
            if control.isCancelled { return true }
            let fraction: Double
            if total > 0 {
                fraction = min(1, Double(frame) / Double(total))
            } else if frame == 0 {
                fraction = 0
            } else {
                fraction = min(0.95, Double(frame) / Double(frame + 24))
            }
            let detail = total > 0 ? "frame \(frame) of \(total)" : "frame \(frame)"
            progress(fraction, "x265 preset • \(detail)")
            return false
        }
        let code: Int32 = try await withTaskCancellationHandler(operation: {
            try await Task.detached(priority: .userInitiated) {
                control.transcode(source: source, output: partial)
            }.value
        }, onCancel: {
            control.cancel()
        })
        if code == 1 || Task.isCancelled {
            try? FileManager.default.removeItem(at: partial)
            throw CancellationError()
        }
        if code != 0 {
            try? FileManager.default.removeItem(at: partial)
            let message = control.errorMessage
            throw ProcessorError.conversionFailed(message.isEmpty ? "x265 preset encode failed." : message)
        }
        let handle = try FileHandle(forWritingTo: partial)
        try handle.synchronize()
        try handle.close()
        if FileManager.default.fileExists(atPath: final.path) {
            try FileManager.default.removeItem(at: final)
        }
        try FileManager.default.moveItem(at: partial, to: final)
        let bytes = X265PresetCache.fileSize(final) ?? 0
        let record = X265PresetCacheRecord(sourceSHA256: digest, preset: preset, complete: true, outputBytes: bytes)
        let data = try JSONEncoder().encode(record)
        try data.write(to: X265PresetCache.manifestURL(sourceSHA256: digest), options: .atomic)
        DiagnosticsLogger.shared.log("x265 preset complete • \(final.lastPathComponent) • \(bytes) bytes • build=\(build)")
        progress(1, "x265 preset complete")
        return final
    }

    static func discardCachedOutput(_ url: URL) {
        let folder = X265PresetCache.directory().standardizedFileURL
        let target = url.standardizedFileURL
        guard target.path.hasPrefix(folder.path + "/") else { return }
        try? FileManager.default.removeItem(at: target)
        let manifest = target.deletingPathExtension().appendingPathExtension("json")
        try? FileManager.default.removeItem(at: manifest)
    }

    private static func existingOutput(sourceSHA256: String, preset: String) -> URL? {
        let url = X265PresetCache.outputURL(sourceSHA256: sourceSHA256)
        guard let reusable = X265PresetCache.reusableOutput(
            sourceSHA256: sourceSHA256,
            preset: preset,
            record: X265PresetCache.loadRecord(sourceSHA256: sourceSHA256),
            outputBytes: X265PresetCache.fileSize(url)
        ), FileManager.default.fileExists(atPath: reusable.path) else { return nil }
        return reusable
    }
}

private final class EncodeControl: @unchecked Sendable {
    let cancelFlag = UnsafeMutablePointer<Int32>.allocate(capacity: 1)
    var error = [CChar](repeating: 0, count: 512)
    let progress = ProgressBox()

    init() { cancelFlag.initialize(to: 0) }
    deinit { cancelFlag.deallocate() }

    var isCancelled: Bool { cancelFlag.pointee != 0 }
    func cancel() { cancelFlag.pointee = 1 }
    var errorMessage: String { String(cString: error) }

    func transcode(source: URL, output: URL) -> Int32 {
        let userdata = Unmanaged.passUnretained(progress).toOpaque()
        return error.withUnsafeMutableBufferPointer { buffer in
            rife60_x265_preset_transcode(
                source.path,
                output.path,
                cancelFlag,
                progressThunk,
                userdata,
                buffer.baseAddress,
                buffer.count
            )
        }
    }
}

private final class ProgressBox: @unchecked Sendable {
    var handler: (Int64, Int64) -> Bool = { _, _ in false }
}

private let progressThunk: @convention(c) (Int64, Int64, UnsafeMutableRawPointer?) -> Int32 = { frame, total, userdata in
    guard let userdata else { return 0 }
    let box = Unmanaged<ProgressBox>.fromOpaque(userdata).takeUnretainedValue()
    return box.handler(frame, total) ? 1 : 0
}
