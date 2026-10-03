import Foundation

struct GenerativeModelStore {
    static let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("AnimeGenerative", isDirectory: true)
    static var diffusion: URL { root.appendingPathComponent("diffusion.gguf") }
    static var textEncoder: URL { root.appendingPathComponent("text-encoder.gguf") }
    static var vae: URL { root.appendingPathComponent("vae.safetensors") }
    static var installed: Bool { [diffusion, textEncoder, vae].allSatisfy { FileManager.default.fileExists(atPath: $0.path) } }
    static var status: String {
        let count = [diffusion, textEncoder, vae].filter { FileManager.default.fileExists(atPath: $0.path) }.count
        return count == 3 ? "Generative model bundle installed" : "Generative models installed: \(count)/3"
    }
    static func install(_ urls: [URL]) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for source in urls {
            let name = source.lastPathComponent.lowercased()
            let destination: URL?
            if name.contains("vace") && name.hasSuffix(".gguf") { destination = diffusion }
            else if name.contains("umt5") && name.hasSuffix(".gguf") { destination = textEncoder }
            else if name.contains("vae") && (name.hasSuffix(".safetensors") || name.hasSuffix(".gguf")) { destination = vae }
            else { destination = nil }
            guard let destination else { continue }
            let secured = source.startAccessingSecurityScopedResource()
            defer { if secured { source.stopAccessingSecurityScopedResource() } }
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: source, to: destination)
        }
        guard installed else {
            throw NSError(domain: "RIFE60Generative", code: 1, userInfo: [NSLocalizedDescriptionKey: "Select the diffusion model, UMT5-XXL encoder, and video VAE."])
        }
    }
}
