import Foundation

/// Thread-safe cancellation signal shared by SwiftUI and the native H3 progress callback.
final class H3CancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}

private final class H3ProgressContext {
    let cancellationToken: H3CancellationToken?
    let handler: ((String, Double) -> Void)?

    init(cancellationToken: H3CancellationToken?, handler: ((String, Double) -> Void)?) {
        self.cancellationToken = cancellationToken
        self.handler = handler
    }
}

private func h3NativeProgressCallback(
    _ phase: UnsafePointer<CChar>?,
    _ completed: Int32,
    _ total: Int32,
    _ opaque: UnsafeMutableRawPointer?
) -> Int32 {
    guard let opaque else { return 0 }
    let context = Unmanaged<H3ProgressContext>.fromOpaque(opaque).takeUnretainedValue()
    let fraction = total > 0 ? min(1, max(0, Double(completed) / Double(total))) : 0
    let phaseName = phase.map { String(cString: $0) } ?? "MiniMax H3"
    context.handler?(phaseName, fraction)
    return context.cancellationToken?.isCancelled == true ? 1 : 0
}

/// Owns a native H3 context. This is deliberately separate from the Wan VACE
/// engine; merely creating a context is not proof that generation will fit in
/// memory or that a particular model directory contains usable weights.
final class H3NativeRuntime {
    private var context: OpaquePointer?
    let modelDirectory: URL

    private init(context: OpaquePointer, modelDirectory: URL) {
        self.context = context
        self.modelDirectory = modelDirectory
    }

    deinit {
        if let context {
            h3_free(context)
        }
    }

    static func load(modelDirectory: URL) throws -> H3NativeRuntime {
        let preflight = H3RuntimePreflight.inspect(modelDirectory: modelDirectory)
        guard preflight.isOperatingSystemSupported else {
            throw RuntimeError.unsupportedOperatingSystem(preflight.message)
        }
        guard preflight.modelDirectoryExists else {
            throw RuntimeError.missingModelDirectory(preflight.message)
        }
        guard preflight.hasMinimumMemory else {
            throw RuntimeError.insufficientMemory(preflight.message)
        }

        // Native model loading may allocate substantial memory and Metal
        // resources. Keep it off the main thread at call sites and catch
        // failures at the UI boundary. The runtime does not promise that a
        // successful load can complete a generation on this device.
        let loadedContext = modelDirectory.path.withCString { path in
            h3_load_dir(path)
        }
        guard let context = loadedContext else {
            throw RuntimeError.nativeLoadFailed(
                "The native H3 runtime could not load the selected model directory. Check its files and available memory."
            )
        }
        return H3NativeRuntime(context: context, modelDirectory: modelDirectory)
    }

    /// Runs a real local H3 text-to-video request. This is intentionally a
    /// distinct workflow from Wan VACE mask editing: H3 does not implement
    /// VACE's white-edit/black-preserve mask semantics here.
    ///
    /// Call from a background task. H3's native generation call is synchronous;
    /// callers should expose cancellation only after a safe native cancel seam
    /// is added rather than pretending Swift task cancellation stops Metal work.
    func generateTextToVideo(
        prompt: String,
        outputURL: URL,
        width: Int = 448,
        height: Int = 256,
        frames: Int = 56,
        steps: Int = 20,
        seed: UInt64 = 42
    ) throws -> GenerationResult {
        guard let context else {
            throw RuntimeError.nativeLoadFailed("The H3 runtime context has already been released.")
        }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RuntimeError.invalidRequest("Enter a non-empty prompt.")
        }
        guard width >= 256, height >= 256, width % 32 == 0, height % 32 == 0,
              frames >= 1, steps >= 1 else {
            throw RuntimeError.invalidRequest("H3 dimensions must be multiples of 32 and frames/steps must be positive.")
        }

        var params = h3_params()
        params.width = Int32(width)
        params.height = Int32(height)
        params.frames = Int32(frames)
        params.steps = Int32(steps)
        params.seed = seed
        params.ssd_streaming = 1
        // Do not enable H3's optional conditioning/DiT/decoder caches: the
        // upstream documents substantial additional retained memory for them.
        params.output_path = nil
        params.first_frame = nil
        params.last_frame = nil
        params.references = nil
        params.reference_count = 0
        params.on_frame = nil
        params.on_progress = nil
        params.callback_opaque = nil

        let resultPointer = prompt.withCString { promptPointer in
            outputURL.path.withCString { outputPathPointer in
                params.output_path = outputPathPointer
                return h3_generate(context, promptPointer, &params)
            }
        }

        guard let resultPointer = resultPointer else {
            let message: String
            if let errorPointer = h3_last_error(context) {
                message = String(cString: errorPointer)
            } else {
                message = "H3 generation failed without a native diagnostic."
            }
            throw RuntimeError.generationFailed(message)
        }
        defer { h3_result_free(resultPointer) }

        return GenerationResult(
            outputURL: outputURL,
            width: Int(resultPointer.pointee.width),
            height: Int(resultPointer.pointee.height),
            frames: Int(resultPointer.pointee.frames),
            seed: resultPointer.pointee.seed
        )
    }

    /// Generates a new video conditioned on an input video reference. This is H3 reference-conditioned regeneration, not VACE-style masked inpainting.
    func generateReferenceConditionedVideo(
        prompt: String,
        referenceVideoURL: URL,
        outputURL: URL,
        width: Int = 448,
        height: Int = 256,
        frames: Int = 56,
        steps: Int = 20,
        seed: UInt64 = 42,
        cancellationToken: H3CancellationToken? = nil,
        onProgress: ((String, Double) -> Void)? = nil
    ) throws -> GenerationResult {
        guard let context else {
            throw RuntimeError.nativeLoadFailed("The H3 runtime context has already been released.")
        }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RuntimeError.invalidRequest("Enter a non-empty prompt.")
        }
        guard FileManager.default.fileExists(atPath: referenceVideoURL.path) else {
            throw RuntimeError.invalidRequest("The selected reference video is no longer accessible.")
        }
        guard width >= 256, height >= 256, width % 32 == 0, height % 32 == 0,
              frames >= 1, steps >= 1 else {
            throw RuntimeError.invalidRequest("H3 dimensions must be multiples of 32 and frames/steps must be positive.")
        }

        var params = h3_params()
        params.width = Int32(width)
        params.height = Int32(height)
        params.frames = Int32(frames)
        params.steps = Int32(steps)
        params.seed = seed
        params.ssd_streaming = 1
        params.output_path = nil
        params.first_frame = nil
        params.last_frame = nil
        params.references = nil
        params.reference_count = 0
        params.on_frame = nil
        let progressContext = H3ProgressContext(cancellationToken: cancellationToken, handler: onProgress)
        let opaque = Unmanaged.passRetained(progressContext).toOpaque()
        defer { Unmanaged<H3ProgressContext>.fromOpaque(opaque).release() }
        params.on_progress = h3NativeProgressCallback
        params.callback_opaque = opaque

        let resultPointer = referenceVideoURL.path.withCString { referencePathPointer in
            prompt.withCString { promptPointer in
                outputURL.path.withCString { outputPathPointer in
                    var reference = h3_reference()
                    reference.kind = H3_REFERENCE_VIDEO
                    reference.path = referencePathPointer
                    reference.audio_path = nil
                    reference.include_embedded_audio = 1
                    return withUnsafePointer(to: &reference) { referencePointer in
                        params.references = referencePointer
                        params.reference_count = 1
                        params.output_path = outputPathPointer
                        return h3_generate(context, promptPointer, &params)
                    }
                }
            }
        }

        guard let resultPointer else {
            if cancellationToken?.isCancelled == true {
                throw RuntimeError.cancelled
            }
            let message: String
            if let errorPointer = h3_last_error(context) {
                message = String(cString: errorPointer)
            } else {
                message = "H3 reference-conditioned generation failed without a native diagnostic."
            }
            throw RuntimeError.generationFailed(message)
        }
        defer { h3_result_free(resultPointer) }
        return GenerationResult(
            outputURL: outputURL,
            width: Int(resultPointer.pointee.width),
            height: Int(resultPointer.pointee.height),
            frames: Int(resultPointer.pointee.frames),
            seed: resultPointer.pointee.seed
        )
    }

    struct GenerationResult {
        let outputURL: URL
        let width: Int
        let height: Int
        let frames: Int
        let seed: UInt64
    }

    enum RuntimeError: LocalizedError {
        case unsupportedOperatingSystem(String)
        case missingModelDirectory(String)
        case insufficientMemory(String)
        case nativeLoadFailed(String)
        case invalidRequest(String)
        case generationFailed(String)
        case cancelled

        var errorDescription: String? {
            switch self {
            case .unsupportedOperatingSystem(let message),
                 .missingModelDirectory(let message),
                 .insufficientMemory(let message),
                 .nativeLoadFailed(let message),
                 .invalidRequest(let message),
                 .generationFailed(let message):
                return message
            case .cancelled:
                return "MiniMax H3 generation was cancelled."
            }
        }
    }
}
