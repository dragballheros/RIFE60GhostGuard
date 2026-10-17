import Foundation
import Darwin

/// A fail-closed capability check for the separate MiniMax H3 backend.
///
/// This deliberately does not load native H3, allocate Metal resources, or
/// imply that the model fits on a device. Call it immediately before any
/// H3-specific model loading; keep the existing Wan VACE path independent.
enum H3RuntimePreflight {
    struct Report {
        let isOperatingSystemSupported: Bool
        let modelDirectoryExists: Bool
        let availableMemoryBytes: UInt64
        let modelDirectory: URL

        var canAttemptInitialization: Bool {
            isOperatingSystemSupported && modelDirectoryExists
        }

        var message: String {
            if !isOperatingSystemSupported {
                return "MiniMax H3 requires iOS 18.0 or newer. Wan VACE remains available."
            }
            if !modelDirectoryExists {
                return "MiniMax H3 model files are not installed. Select or install a compatible local H3 model directory first."
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

        return Report(
            isOperatingSystemSupported: ProcessInfo.processInfo.isOperatingSystemAtLeast(
                OperatingSystemVersion(majorVersion: 18, minorVersion: 0, patchVersion: 0)
            ),
            modelDirectoryExists: exists,
            availableMemoryBytes: UInt64(os_proc_available_memory()),
            modelDirectory: modelDirectory
        )
    }
}

