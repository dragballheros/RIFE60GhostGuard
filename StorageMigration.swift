import Foundation

/// Launch cleanup. Wan VACE checkpoints stay on disk. Temporary frame folders are removed.
enum StorageMigration {
    private static let retiredNames = [
        "wan_v2.1_1.3b_480p_q8p.ckpt",
        "wan_v2.1_1.3b_vace_480p_q8p.ckpt",
        "umt5_xxl_encoder_q8p.ckpt",
        "wan_v2.1_video_vae_f16.ckpt"
    ]

    static func removeRetiredVACEWeights() {
        let fm = FileManager.default
        // Wan VACE is back in the app. Leave the 1.3B checkpoints in place.
        DiagnosticsLogger.shared.log("VACE weights kept: \(retiredNames.joined(separator: ", "))")
        let tmp = fm.temporaryDirectory
        if let items = try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil) {
            for item in items where item.lastPathComponent.hasPrefix("RIFE-VACE-frames-") {
                try? fm.removeItem(at: item)
            }
        }
        DiagnosticsLogger.shared.log("VACE weight migration finished. Documents/WanVACEExports and old Documents/MiniMax-H3 folders were not deleted.")
    }
}
