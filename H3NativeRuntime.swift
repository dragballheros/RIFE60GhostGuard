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

    enum RuntimeError: LocalizedError {
        case unsupportedOperatingSystem(String)
        case missingModelDirectory(String)
        case nativeLoadFailed(String)

        var errorDescription: String? {
            switch self {
            case .unsupportedOperatingSystem(let message),
                 .missingModelDirectory(let message),
                 .nativeLoadFailed(let message):
                return message
            }
        }
    }
}
