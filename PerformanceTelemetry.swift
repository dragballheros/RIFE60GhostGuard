import Foundation

struct PerformanceTelemetry: Sendable {
    var sourceFrames: Int = 0
    var generatedFrames: Int = 0
    var rejectedFrames: Int = 0
    var upscaledFrames: Int = 0

    var compressionMsPerFrame: Double = 0
    var outlineMsPerFrame: Double = 0
    var rifeMsPerGeneratedFrame: Double = 0
    var ghostMsPerGeneratedFrame: Double = 0
    var encodeMsPerOutputFrame: Double = 0
    var cuganMsPerFrame: Double = 0
    var generatedFPS: Double = 0
    var upscaleFPS: Double = 0

    var thermalState: String = currentThermalStateName()

    var summary: String {
        String(
            format: "Thermal: %@\nCompression: %.1f ms/src\nOutline: %.1f ms/src\nRIFE HQ: %.1f ms/gen\nReal-CUGAN: %.1f ms/frame\nGhostGuard: %.1f ms/gen\nEncode: %.1f ms/out\nRIFE speed: %.2f gen fps\nUpscale speed: %.2f fps",
            thermalState,
            compressionMsPerFrame,
            outlineMsPerFrame,
            rifeMsPerGeneratedFrame,
            cuganMsPerFrame,
            ghostMsPerGeneratedFrame,
            encodeMsPerOutputFrame,
            generatedFPS,
            upscaleFPS
        )
    }
}

/// Automatic render policy. There is deliberately no user toggle: the app runs
/// flat-out while iOS reports thermal headroom and falls back before the next
/// frame when the system reaches Serious/Critical. No in-flight frame is changed.
@inline(__always)
func automaticPerformanceModeEnabled() -> Bool {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal, .fair:
        return true
    case .serious, .critical:
        return false
    @unknown default:
        return false
    }
}

@inline(__always)
func currentThermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "Nominal • Performance Mode"
    case .fair: return "Fair • Performance Mode"
    case .serious: return "Serious • Thermal Safe Mode"
    case .critical: return "Critical • Thermal Safe Mode"
    @unknown default: return "Unknown • Thermal Safe Mode"
    }
}

/// Called only between completed frames. Performance Mode does not sleep/yield;
/// Thermal Safe Mode gives iOS a small scheduling window without changing model
/// weights, RIFE quality, CUGAN strength, Sharpie, GhostGuard, FPS, or export.
func thermalFrameBoundaryPacing() async throws {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal, .fair:
        return
    case .serious:
        try Task.checkCancellation()
        try await Task.sleep(nanoseconds: 3_000_000)
    case .critical:
        try Task.checkCancellation()
        try await Task.sleep(nanoseconds: 10_000_000)
    @unknown default:
        try Task.checkCancellation()
        await Task.yield()
    }
}
