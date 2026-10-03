import Foundation

struct GenerativeModelStore {
    static let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("AnimeGenerative", isDirectory: true)
    static var diffusion: URL { root.appendingPathComponent("diffusion.gguf") }
    static var textEncoder: URL { root.appendingPathComponent("text-encoder.gguf") }
    static var vae: URL { root.appendingPathComponent("vae.safetensors") }
    static var installed: Bool {
        [diffusion, textEncoder, vae].allSatisfy { FileManager.default.fileExists(atPath: $0.path) && ((try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 1_000_000 }
    }
    static var status: String {
        if installed { return "Wan 2.1 VACE 1.3B Q4 ready" }
        let n = [diffusion, textEncoder, vae].filter { FileManager.default.fileExists(atPath: $0.path) }.count
        return n == 0 ? "Wan edit model downloads automatically" : "Generative models installed: \(n)/3"
    }

    // Smallest Wan video-edit stack that stable-diffusion.cpp can run. Wan 2.2/2.7 14B does not fit in 6 GB.
    static let files: [(url: URL, destination: URL, label: String)] = [
        (URL(string: "https://huggingface.co/samuelchristlie/Wan2.1-VACE-1.3B-GGUF/resolve/main/Wan2.1-VACE-1.3B-Q4_K_S.gguf")!, diffusion, "VACE 1.3B"),
        (URL(string: "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q3_K_S.gguf")!, textEncoder, "UMT5 encoder"),
        (URL(string: "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors")!, vae, "Wan VAE")
    ]

    static func ensureInstalled(progress: @escaping (Double, String) -> Void) async throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let pending = files.filter { !FileManager.default.fileExists(atPath: $0.destination.path) || ((try? $0.destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) < 1_000_000 }
        if pending.isEmpty { progress(1, status); return }
        for (index, file) in pending.enumerated() {
            let partial = file.destination.appendingPathExtension("partial")
            let base = Double(index) / Double(pending.count)
            try await download(file.url, to: partial) { fraction in
                progress(base + fraction / Double(pending.count), "Downloading \(file.label) \(Int(fraction * 100))%")
            }
            try? FileManager.default.removeItem(at: file.destination)
            try FileManager.default.moveItem(at: partial, to: file.destination)
        }
        guard installed else { throw NSError(domain: "RIFE60Generative", code: 2, userInfo: [NSLocalizedDescriptionKey: "Wan edit model download did not finish."]) }
        progress(1, status)
    }

    private static func download(_ remote: URL, to destination: URL, fraction: @escaping (Double) -> Void) async throws {
        let delegate = ProgressDelegate(fraction: fraction)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let (temp, response) = try await session.download(from: remote)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw NSError(domain: "RIFE60Generative", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not download \(remote.lastPathComponent)."])
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temp, to: destination)
    }
}

private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate {
    let fraction: (Double) -> Void
    init(fraction: @escaping (Double) -> Void) { self.fraction = fraction }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        fraction(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
}
