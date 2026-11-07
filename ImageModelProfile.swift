import Foundation

/// One source of truth for a catalog model and the sampling defaults that belong to it.
/// Suggested tags are never inserted unless the user taps the insert control.
struct ImageModelProfile: Equatable, Sendable {
    enum Family: String, Sendable {
        case sdxl
        case flowMatching
    }

    let modelID: String
    let displayName: String
    let family: Family
    let width: Double
    let height: Double
    let steps: Double
    let cfg: Double
    let shift: Double
    let clipSkip: Double
    let samplerName: String
    /// Shown as an optional insert. Not applied automatically.
    let suggestedTags: String

    var usesShift: Bool { family == .flowMatching }

    static let animagineXL = ImageModelProfile(
        modelID: OnDeviceImageGenerator.defaultModelID,
        displayName: "Animagine XL v3.1 (Local)",
        family: .sdxl,
        width: 832,
        height: 1216,
        steps: 28,
        cfg: 5,
        shift: 1,
        clipSkip: 2,
        samplerName: "Euler a",
        suggestedTags: "masterpiece, best quality"
    )

    static let animaBase = ImageModelProfile(
        modelID: OnDeviceImageGenerator.animaModelID,
        displayName: "Anima Base v1.0 (Local)",
        family: .flowMatching,
        width: 832,
        height: 1216,
        steps: 28,
        cfg: 4,
        shift: 3,
        clipSkip: 1,
        samplerName: "Euler a Trailing",
        suggestedTags: ""
    )

    static let all: [ImageModelProfile] = [.animagineXL, .animaBase]

    static func profile(for modelID: String) -> ImageModelProfile {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == animaBase.modelID || trimmed == "anima_aesthetic_1.1_q8p.ckpt" {
            return .animaBase
        }
        return .animagineXL
    }

    /// Draw Things flow-matching models reject Karras samplers. Keep the user's
    /// choice when it is a trailing/AYS sampler; otherwise use the profile default.
    var resolvedSamplerName: String {
        if family == .flowMatching && !Self.flowCompatibleSamplers.contains(samplerName) {
            return Self.animaBase.samplerName
        }
        return samplerName
    }

    static let flowCompatibleSamplers: Set<String> = [
        "Euler a Trailing",
        "DPM++ SDE Trailing",
        "DPM++ 2M Trailing"
    ]

    func appliedShift(_ requested: Double) -> Double {
        guard usesShift else { return 1 }
        return min(12, max(0.1, requested))
    }
}
