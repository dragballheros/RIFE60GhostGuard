import CoreVideo
import Foundation

/// Model-agnostic Anime Restore policy. No weights live here.
/// A future model registers `AnimeRestoreRegistry.factory` and receives full frames.
/// This pass blends those frames by strength. Strength 1 keeps the model frame.
struct AnimeRestoreDecision: Equatable {
    let enabled: Bool
    let automatic: Bool
    let reason: String
}

enum AnimeRestorePolicy {
    static let shortSideLimit = 720

    static func shortSideQualifies(width: Int, height: Int) -> Bool {
        width > 0 && height > 0 && min(width, height) <= shortSideLimit
    }

    /// Persisted ON stays on for every source. Auto applies only for a short side of 720 px
    /// or less, and only until the user turns the toggle off for that source.
    /// Auto never writes the persisted toggle to ON.
    static func decide(
        width: Int,
        height: Int,
        persistedEnabled: Bool,
        suppressAutoForThisSource: Bool,
        modelRegistered: Bool
    ) -> AnimeRestoreDecision {
        if persistedEnabled {
            let reason = modelRegistered
                ? "user toggle on"
                : "user toggle on • no model registered • frames unchanged"
            return AnimeRestoreDecision(enabled: modelRegistered, automatic: false, reason: reason)
        }
        let qualifies = shortSideQualifies(width: width, height: height)
        if qualifies && !suppressAutoForThisSource {
            let reason = modelRegistered
                ? "auto short side \(min(width, height))px ≤ \(shortSideLimit)"
                : "auto short side \(min(width, height))px ≤ \(shortSideLimit) • no model registered • frames unchanged"
            return AnimeRestoreDecision(enabled: modelRegistered, automatic: true, reason: reason)
        }
        if qualifies && suppressAutoForThisSource {
            return AnimeRestoreDecision(enabled: false, automatic: false, reason: "user turned auto off for this source")
        }
        return AnimeRestoreDecision(enabled: false, automatic: false, reason: "short side above \(shortSideLimit) and toggle off")
    }
}

protocol AnimeRestoreModel: AnyObject {
    /// Full restored frame at the same pixel size as `frame`. The pass applies strength.
    func restore(_ frame: CVPixelBuffer) throws -> CVPixelBuffer
    func release()
}

enum AnimeRestoreRegistry {
    /// Nil until a restoration model is chosen. Nothing in the app registers a factory.
    static var factory: (() throws -> any AnimeRestoreModel)?
    static var displayName = "unregistered"

    static var isRegistered: Bool { factory != nil }
}
