import Foundation

/// One-time cleanup of retired Wan VACE weight files. Export folders are left in place.
enum StorageMigration {
    private static let retiredNames = [
        "wan_v2.1_1.3b_480p_q8p.ckpt",
        "wan_v2.1_1.3b_vace_480p_q8p.ckpt",
        "umt5_xxl_encoder_q8p.ckpt",
        "wan_v2.1_video_vae_f16.ckpt"
    ]

    static func removeRetiredVACEWeights() {
        let fm = FileManager.default
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        var roots = [
            documents.appendingPathComponent("Models", isDirectory: true)
        ]
        if let support = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            roots.append(support.appendingPathComponent("RIFE60GhostGuard/Models", isDirectory: true))
        }
        for root in roots {
            for name in retiredNames {
                let url = root.appendingPathComponent(name)
                if fm.fileExists(atPath: url.path) {
                    try? fm.removeItem(at: url)
                    DiagnosticsLogger.shared.log("Removed retired VACE weight \(name)")
                }
            }
        }
        let tmp = fm.temporaryDirectory
        if let items = try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil) {
            for item in items where item.lastPathComponent.hasPrefix("RIFE-VACE-frames-") {
                try? fm.removeItem(at: item)
            }
        }
        DiagnosticsLogger.shared.log("VACE weight migration finished. Documents/WanVACEExports and old Documents/MiniMax-H3 folders were not deleted.")
    }
}
