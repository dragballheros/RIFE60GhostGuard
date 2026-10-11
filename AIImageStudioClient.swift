import Foundation
import CryptoKit

struct AIImageStudioSettings: Codable {
    var localModelID = "animagine_xl_v3.1_q6p_q8p.ckpt"
    var localLoraFile = ""

    static let storageKey = "RIFE60.AIImageStudio.Settings.Local.v2"

    static func load() -> AIImageStudioSettings {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let value = try? JSONDecoder().decode(AIImageStudioSettings.self, from: data) {
            return value
        }
        // Deliberately do not restore old endpoint URLs or credentials.
        return AIImageStudioSettings()
    }

    func save() throws {
        let data = try JSONEncoder().encode(self)
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

struct AIImageStudioError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

struct LocalTrainingImage {
    let filename: String
    let sourceURL: URL
    let caption: String
}



/// A public character LoRA explicitly trained for Animagine XL 3.1 / SDXL.
/// Downloaded on demand into the app's private Models directory; never bundled in the IPA.
enum RecommendedAnimeLoRAInstaller {
    static let fileName = "XL-Mayuzumi-Fuyuko-Animagine-XL-3.1.safetensors"
    static let displayName = "Mayuzumi Fuyuko (Animagine XL 3.1)"
    static let triggerWord = "mayuzumi fuyuko"
    static let downloadURL = URL(string: "https://civitai.com/api/download/models/454121")!
    static let expectedSHA256 = "a150b41294268bad550af11c2f906a853e15fb0f3881711810ed9621414d41cf"
    static let expectedBytes: Int64 = 456_483_000

    static func install(into directory: URL) async throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(fileName)
        if FileManager.default.fileExists(atPath: destination.path) {
            if try sha256(of: destination) == expectedSHA256 { return destination }
            try FileManager.default.removeItem(at: destination)
        }

        let values = try directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        if let available = values.volumeAvailableCapacityForImportantUsage, available < expectedBytes + 100_000_000 {
            throw AIImageStudioError("The recommended LoRA needs about 435 MB plus temporary download space. Free at least 550 MB, then retry.")
        }

        var lastNetworkError = "Unknown network error."
        for attempt in 1...3 {
            do {
                var request = URLRequest(url: downloadURL)
                request.timeoutInterval = 300
                request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
                request.setValue("RIFE60GhostGuard/1.0 (iOS model downloader)", forHTTPHeaderField: "User-Agent")
                request.setValue("https://civitai.com/", forHTTPHeaderField: "Referer")
                let (temporaryURL, response) = try await URLSession.shared.download(for: request)
                guard let http = response as? HTTPURLResponse else {
                    try? FileManager.default.removeItem(at: temporaryURL)
                    throw AIImageStudioError("The LoRA host returned an invalid HTTP response.")
                }
                guard (200...299).contains(http.statusCode) else {
                    try? FileManager.default.removeItem(at: temporaryURL)
                    let retryable = [408, 425, 429, 500, 502, 503, 504].contains(http.statusCode)
                    if retryable && attempt < 3 {
                        try await Task.sleep(nanoseconds: UInt64(attempt) * 1_000_000_000)
                        continue
                    }
                    throw AIImageStudioError("CivitAI returned HTTP \(http.statusCode) while downloading the verified Animagine XL 3.1 LoRA. Check the network or retry later.")
                }
                let downloadedSize = try FileManager.default.attributesOfItem(atPath: temporaryURL.path)[.size] as? NSNumber
                guard let downloadedSize, downloadedSize.int64Value > 100_000_000 else {
                    try? FileManager.default.removeItem(at: temporaryURL)
                    throw AIImageStudioError("The LoRA host returned an incomplete file, not the 435 MB model.")
                }
                let digest = try sha256(of: temporaryURL)
                guard digest == expectedSHA256 else {
                    try? FileManager.default.removeItem(at: temporaryURL)
                    throw AIImageStudioError("The LoRA checksum did not match the published Animagine XL 3.1 file. The incomplete download was rejected.")
                }
                try FileManager.default.moveItem(at: temporaryURL, to: destination)
                return destination
            } catch {
                lastNetworkError = error.localizedDescription
                if error is AIImageStudioError || attempt == 3 {
                    throw AIImageStudioError("Could not install the verified Mayuzumi Fuyuko LoRA after \(attempt) attempt(s): \(lastNetworkError)")
                }
                try? await Task.sleep(nanoseconds: UInt64(attempt) * 1_000_000_000)
            }
        }
        throw AIImageStudioError("Could not install the verified Mayuzumi Fuyuko LoRA: \(lastNetworkError)")
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            guard let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
