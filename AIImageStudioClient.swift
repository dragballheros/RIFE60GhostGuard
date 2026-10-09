import Foundation

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
    let data: Data
    let caption: String
}
