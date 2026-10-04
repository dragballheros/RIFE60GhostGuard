import Foundation

struct GenerativeModelStore {
    static let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("AnimeGenerative", isDirectory: true)
    static var diffusion: URL { root.appendingPathComponent("diffusion.safetensors") }
    static let revision = "vace-1.3b-fp16-safetensors"
    static var revisionFile: URL { root.appendingPathComponent("revision.txt") }
    static var textEncoder: URL { root.appendingPathComponent("text-encoder.gguf") }
    static var vae: URL { root.appendingPathComponent("vae.safetensors") }
    static var installed: Bool { [diffusion, textEncoder, vae].allSatisfy(valid) }
    static var status: String { installed ? "Wan 2.1 VACE 1.3B Q4 ready" : "Downloading Wan edit model…" }

    static let files: [(url: URL, destination: URL, label: String)] = [
        (URL(string: "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors?download=true")!, diffusion, "VACE 1.3B"),
        (URL(string: "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q3_K_S.gguf?download=true")!, textEncoder, "UMT5 encoder"),
        (URL(string: "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true")!, vae, "Wan VAE")
    ]

    static func ensureInstalled(progress: @escaping (Double, String) -> Void) async throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let current = (try? String(contentsOf: revisionFile, encoding: .utf8)) ?? ""
        if current != revision {
            for file in files { try? FileManager.default.removeItem(at: file.destination) }
        }
        let pending = files.filter { !valid($0.destination) }
        if pending.isEmpty { progress(1, "Wan 2.1 VACE 1.3B Q4 ready"); return }
        for (index, file) in pending.enumerated() {
            let partial = file.destination.appendingPathExtension("partial")
            let base = Double(index) / Double(pending.count)
            let downloaded = try await download(file.url) { fraction in
                progress(base + fraction / Double(pending.count), "Downloading \(file.label) \(Int(fraction * 100))%")
            }
            guard valid(downloaded) else { throw NSError(domain: "RIFE60Generative", code: 3, userInfo: [NSLocalizedDescriptionKey: "\(file.label) download was empty."]) }
            try? FileManager.default.removeItem(at: partial)
            try? FileManager.default.removeItem(at: file.destination)
            try FileManager.default.moveItem(at: downloaded, to: file.destination)
        }
        guard installed else { throw NSError(domain: "RIFE60Generative", code: 2, userInfo: [NSLocalizedDescriptionKey: "Wan edit model download did not finish."]) }
        try revision.write(to: revisionFile, atomically: true, encoding: .utf8)
        progress(1, "Wan 2.1 VACE 1.3B Q4 ready")
    }

    private static func valid(_ url: URL) -> Bool {
        ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 1_000_000
    }

    private static func download(_ remote: URL, fraction: @escaping (Double) -> Void) async throws -> URL {
        let downloader = FileDownloader(fraction: fraction)
        return try await downloader.download(remote)
    }
}

private final class FileDownloader: NSObject, URLSessionDownloadDelegate {
    let fraction: (Double) -> Void
    private var continuation: CheckedContinuation<URL, Error>?
    private var session: URLSession?
    init(fraction: @escaping (Double) -> Void) { self.fraction = fraction }

    func download(_ remote: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation
            let config = URLSessionConfiguration.default
            config.waitsForConnectivity = true
            config.timeoutIntervalForRequest = 120
            let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
            self.session = session
            var request = URLRequest(url: remote)
            request.setValue("RIFE60GhostGuard/1.0", forHTTPHeaderField: "User-Agent")
            session.downloadTask(with: request).resume()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        fraction(min(1, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)))
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let dest = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: dest)
            continuation?.resume(returning: dest)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error, continuation != nil {
            continuation?.resume(throwing: error)
            continuation = nil
        }
        session.finishTasksAndInvalidate()
    }
}
