import Foundation
import CryptoKit

struct StoredLocalLoRA: Codable, Equatable, Sendable {
    var file: String
    var weight: Double
    var version: String
    var trigger: String
}

struct AIImageStudioSettings: Codable {
    var localModelID = OnDeviceImageGenerator.defaultModelID
    var localLoraFile = ""
    var localLoras: [StoredLocalLoRA] = []

    static let storageKey = "RIFE60.AIImageStudio.Settings.Local.v2"

    static func load() -> AIImageStudioSettings {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let value = try? JSONDecoder().decode(AIImageStudioSettings.self, from: data) {
            return value
        }
        // Deliberately do not restore old endpoint URLs or credentials.
        return AIImageStudioSettings()
    }

    enum CodingKeys: String, CodingKey {
        case localModelID, localLoraFile, localLoras
    }

    init() {}

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        localModelID = try container.decodeIfPresent(String.self, forKey: .localModelID) ?? OnDeviceImageGenerator.defaultModelID
        localLoraFile = try container.decodeIfPresent(String.self, forKey: .localLoraFile) ?? ""
        localLoras = try container.decodeIfPresent([StoredLocalLoRA].self, forKey: .localLoras) ?? []
        if localModelID == "anima_aesthetic_1.1_q8p.ckpt" {
            localModelID = OnDeviceImageGenerator.animaModelID
        }
        if localModelID != OnDeviceImageGenerator.defaultModelID
            && localModelID != OnDeviceImageGenerator.animaModelID {
            localModelID = OnDeviceImageGenerator.defaultModelID
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(localModelID, forKey: .localModelID)
        try container.encode(localLoraFile, forKey: .localLoraFile)
        try container.encode(localLoras, forKey: .localLoras)
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
