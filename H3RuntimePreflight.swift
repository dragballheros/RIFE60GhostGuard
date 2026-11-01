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

        var missingComponents: [String]

        var hasMinimumMemory: Bool {
            availableMemoryBytes >= H3RuntimePreflight.minimumAvailableMemoryBytes
        }

        var canAttemptInitialization: Bool {
            isOperatingSystemSupported && modelDirectoryExists && modelLayoutIsValid && hasMinimumMemory
        }

        var message: String {
            if !isOperatingSystemSupported {
                return "MiniMax H3 requires iOS 18.0 or newer."
            }
            if !modelDirectoryExists {
                return "MiniMax H3 model files are not installed. Select or install a compatible local H3 model directory first."
            }
            if !modelLayoutIsValid {
                let names = missingComponents.isEmpty ? "unknown component" : missingComponents.joined(separator: "; ")
                return "Missing H3 component: \(names)."
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
        // Every path h3_load_dir or generation fopen's after tier-1 setup.
        // Video VAE stats live in video_vae/config.json (the runtime opens
        // source/../config.json). Audio stats live in audio_vae/config.json.
        let runtimePaths = [
            "FL2VA/transformer/config.json",
            "FL2VA/transformer/minimax_h3_ref2va_pruned-Q8_0.gguf",
            "FL2VA/tokenizer/tokenizer.json",
            "FL2VA/text_encoder/config.json",
            "FL2VA/text_encoder/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            "FL2VA/video_vae/config.json",
            "FL2VA/video_vae/source/minimax_h3_video_vae_fp16.safetensors",
            "FL2VA/audio_vae/config.json",
            "FL2VA/audio_vae/minimax_h3_audio_vae_fp32.safetensors",
            "Ref2VA/transformer/config.json",
            "Ref2VA/transformer/model.safetensors.index.json",
            "Ref2VA/transformer/minimax_h3_ref2va_pruned-Q8_0.gguf",
            "Ref2VA/tokenizer/tokenizer.json",
            "Ref2VA/text_encoder/" + "config.json",
            "Ref2VA/text_encoder/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            "Ref2VA/video_vae/config.json",
            "Ref2VA/video_vae/source/minimax_h3_video_vae_fp16.safetensors",
            "Ref2VA/audio_vae/config.json",
            "Ref2VA/audio_vae/minimax_h3_audio_vae_fp32.safetensors"
        ]
        let weightFiles = [
            ("minimax_h3_ref2va_pruned-Q8_0.gguf", "Ref2VA transformer Q8_0 weights"),
            ("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "Qwen3-VL text encoder Q4_K_M weights"),
            ("vae_minimax_h3_video_vae_fp16.safetensors", "video VAE fp16 safetensors"),
            ("vae/minimax_h3_video_vae_fp16.safetensors", "video VAE fp16 safetensors"),
            ("vae_minimax_h3_audio_vae_fp32.safetensors", "audio VAE fp32 safetensors"),
            ("vae/minimax_h3_audio_vae_fp32.safetensors", "audio VAE fp32 safetensors")
        ]
        let roots = [modelDirectory, modelDirectory.appendingPathComponent("tier1-weights")]
        func present(_ relative: String) -> Bool {
            roots.contains { root in
                var isDirectory: ObjCBool = false
                let path = root.appendingPathComponent(relative).path
                return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) && !isDirectory.boolValue
            } || {
                var isDirectory: ObjCBool = false
                let path = modelDirectory.appendingPathComponent(relative).path
                return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) && !isDirectory.boolValue
            }()
        }
        var missing: [String] = []
        if !present("minimax_h3_ref2va_pruned-Q8_0.gguf") && !present("FL2VA/transformer/minimax_h3_ref2va_pruned-Q8_0.gguf") {
            missing.append("minimax_h3_ref2va_pruned-Q8_0.gguf (Ref2VA transformer Q8_0 weights)")
        }
        if !present("qwen3vl_32b_minimax_h3-Q4_K_M.gguf") && !present("FL2VA/text_encoder/qwen3vl_32b_minimax_h3-Q4_K_M.gguf") {
            missing.append("qwen3vl_32b_minimax_h3-Q4_K_M.gguf (Qwen3-VL text encoder Q4_K_M weights)")
        }
        let videoPresent = weightFiles.filter { $0.1 == "video VAE fp16 safetensors" }.contains { present($0.0) }
            || present("FL2VA/video_vae/source/minimax_h3_video_vae_fp16.safetensors")
        if !videoPresent {
            missing.append("vae/minimax_h3_video_vae_fp16.safetensors (video VAE)")
        }
        let audioPresent = weightFiles.filter { $0.1 == "audio VAE fp32 safetensors" }.contains { present($0.0) }
            || present("FL2VA/audio_vae/minimax_h3_audio_vae_fp32.safetensors")
        if !audioPresent {
            missing.append("vae/minimax_h3_audio_vae_fp32.safetensors (audio VAE)")
        }
        if !present("FL2VA/tokenizer/tokenizer.json") {
            missing.append("FL2VA/tokenizer/tokenizer.json (bundled tokenizer)")
        }
        for relative in runtimePaths where !present(relative) {
            if !missing.contains(where: { $0.hasPrefix(relative) }) {
                missing.append(relative)
            }
        }
        for relative in ["FL2VA/audio_vae/config.json", "Ref2VA/audio_vae/config.json", "FL2VA/video_vae/config.json", "Ref2VA/video_vae/config.json"] {
            let url = modelDirectory.appendingPathComponent(relative)
            guard FileManager.default.fileExists(atPath: url.path),
                  let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            if !text.contains("\"latents_mean\"") || !text.contains("\"latents_std\"") {
                missing.append("\(relative) (latents_mean and latents_std)")
            }
        }
        let layoutIsValid = exists && missing.isEmpty

        return Report(
            isOperatingSystemSupported: ProcessInfo.processInfo.isOperatingSystemAtLeast(
                OperatingSystemVersion(majorVersion: 18, minorVersion: 0, patchVersion: 0)
            ),
            modelDirectoryExists: exists,
            modelLayoutIsValid: layoutIsValid,
            availableMemoryBytes: UInt64(os_proc_available_memory()),
            modelDirectory: modelDirectory,
            missingComponents: missing
        )
    }
}

