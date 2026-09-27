import Foundation

struct PerformanceTelemetry: Sendable {
    var sourceFrames: Int = 0
    var generatedFrames: Int = 0
    var rejectedFrames: Int = 0

    var compressionMsPerFrame: Double = 0
    var outlineMsPerFrame: Double = 0
    var rifeMsPerGeneratedFrame: Double = 0
    var ghostMsPerGeneratedFrame: Double = 0
    var encodeMsPerOutputFrame: Double = 0
    var generatedFPS: Double = 0

    var thermalState: String = "Nominal"

    var summary: String {
        String(
            format: "Thermal: %@\nCompression: %.1f ms/src\nOutline: %.1f ms/src\nRIFE HQ: %.1f ms/gen\nGhostGuard: %.1f ms/gen\nEncode: %.1f ms/out\nRIFE speed: %.2f gen fps",
            thermalState,
            compressionMsPerFrame,
            outlineMsPerFrame,
            rifeMsPerGeneratedFrame,
            ghostMsPerGeneratedFrame,
            encodeMsPerOutputFrame,
            generatedFPS
        )
    }
}

func currentThermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "Nominal"
    case .fair: return "Fair"
    case .serious: return "Serious"
    case .critical: return "Critical"
    @unknown default: return "Unknown"
    }
}
