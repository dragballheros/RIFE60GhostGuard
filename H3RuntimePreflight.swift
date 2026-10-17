import Foundation
import Darwin

/// A fail-closed capability check for the separate MiniMax H3 backend.
///
/// This deliberately does not load native H3, allocate Metal resources, or
/// imply that the model fits on a device. Call it immediately before any
/// H3-specific model loading; keep the existing Wan VACE path independent.
enum H3RuntimePreflight {
    /// Conservative launch floor only. Passing it does not guarantee that model loading or inference fits.
    static let minimumAvailableMemoryBytes: UInt64 = 2 * 1024 * 1024 * 1024

    struct Report {
        let isOperatingSystemSupported: Bool
        let modelDirectoryExists: Bool
        let modelLayoutIsValid: Bool
        let availableMemoryBytes: UInt64
        let modelDirectory: URL

        var hasMinimumMemory: Bool {
            availableMemoryBytes >= H3RuntimePreflight.minimumAvailableMemoryBytes
        }

        var canAttemptInitialization: Bool {
            isOperatingSystemSupported && modelDirectoryExists && modelLayoutIsValid && hasMinimumMemory
        }

        var message: String {
            if !isOperatingSystemSupported {
                return "MiniMax H3 requires iOS 18.0 or newer. Wan VACE remains available."
            }
            if !modelDirectoryExists {
                return "MiniMax H3 model files are not installed. Select or install a compatible local H3 model directory first."
            }
            if !modelLayoutIsValid {
                return "The selected folder is not a complete H3 Ref2VA Diffusers checkpoint. Expected model_index.json plus processor, tokenizer, text_encoder, transformer, visual_vae, and audio_vae. A partial download or a GGUF-only file is not compatible with this native runtime."
            }
            if !hasMinimumMemory {
                let availableGiB = Double(availableMemoryBytes) / 1_073_741_824
                return String(format: "MiniMax H3 needs at least 2 GiB of currently available memory to attempt loading; only %.2f GiB is available. Close other apps and retry. This floor does not guarantee inference will fit.", availableGiB)
            }
            return "Basic preflight passed. This does not guarantee sufficient memory or successful inference; H3 must still fail safely if native model loading or Metal allocation fails."
        }
    }

    static func inspect(modelDirectory: URL) -> Report {
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: modelDirectory.path,
            isDirectory: &isDirectory
        ) && isDirectory.boolValue
        let requiredPaths = ["model_index.json", "processor", "tokenizer", "text_encoder", "transformer", "visual_vae", "audio_vae"]
        let layoutIsValid = exists && requiredPaths.allSatisfy { component in
            let candidate = modelDirectory.appendingPathComponent(component)
            var componentIsDirectory: ObjCBool = false
            let present = FileManager.default.fileExists(atPath: candidate.path, isDirectory: &componentIsDirectory)
            if component == "model_index.json" { return present && !componentIsDirectory.boolValue }
            return present && componentIsDirectory.boolValue
        }

        return Report(
            isOperatingSystemSupported: ProcessInfo.processInfo.isOperatingSystemAtLeast(
                OperatingSystemVersion(majorVersion: 18, minorVersion: 0, patchVersion: 0)
            ),
            modelDirectoryExists: exists,
            modelLayoutIsValid: layoutIsValid,
            availableMemoryBytes: UInt64(os_proc_available_memory()),
            modelDirectory: modelDirectory
        )
    }
}

