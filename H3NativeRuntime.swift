import Foundation

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
        case nativeLoadFailed(String)
        case invalidRequest(String)
        case generationFailed(String)

        var errorDescription: String? {
            switch self {
            case .unsupportedOperatingSystem(let message),
                 .missingModelDirectory(let message),
                 .nativeLoadFailed(let message),
                 .invalidRequest(let message),
                 .generationFailed(let message):
                return message
            }
        }
    }
}
