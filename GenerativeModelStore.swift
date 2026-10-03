import Foundation

struct GenerativeModelStore {
    static let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("AnimeGenerative", isDirectory: true)
    static var diffusion: URL { root.appendingPathComponent("diffusion.gguf") }
    static var textEncoder: URL { root.appendingPathComponent("text-encoder.gguf") }
    static var vae: URL { root.appendingPathComponent("vae.safetensors") }
    static var installed: Bool { [diffusion, textEncoder, vae].allSatisfy { FileManager.default.fileExists(atPath: $0.path) } }
    static var status: String {
        let n = [diffusion, textEncoder, vae].filter { FileManager.default.fileExists(atPath: $0.path) }.count
        return n == 3 ? "Generative model bundle installed" : "Generative models installed: \(n)/3"
    }
    static func install(_ urls: [URL]) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for source in urls {
            let n = source.lastPathComponent.lowercased()
            let destination: URL?
            if n.contains("vace") && n.hasSuffix(".gguf") { destination = diffusion }
            else if n.contains("umt5") && n.hasSuffix(".gguf") { destination = textEncoder }
            else if n.contains("vae") && n.hasSuffix(".safetensors") { destination = vae }
            else { destination = nil }
            guard let destination else { continue }
            let access = source.startAccessingSecurityScopedResource()
            defer { if access { source.stopAccessingSecurityScopedResource() } }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: source, to: destination)
        }
        guard installed else { throw NSError(domain: "RIFE60Generative", code: 1, userInfo: [NSLocalizedDescriptionKey: "Select the VACE diffusion GGUF, UMT5-XXL GGUF, and Wan VAE files."]) }
    }
}
