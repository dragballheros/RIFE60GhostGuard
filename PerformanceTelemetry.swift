import Foundation
import UIKit
import os

enum RenderPerformanceTier: String, Sendable {
    case performance = "Performance Mode"
    case balanced = "Memory Balanced"
    case safe = "Memory Safe Mode"
}

struct RenderPerformanceSnapshot: Sendable {
    let tier: RenderPerformanceTier
    let availableMemoryMB: Double
    let physicalMemoryMB: Double
    let thermalState: ProcessInfo.ThermalState

    var allowsWideCUGANTiles: Bool {
        tier == .performance && availableMemoryMB >= 1_700
    }

    var modeLabel: String {
        switch thermalState {
        case .serious, .critical:
            return "Thermal Safe Mode"
        default:
            return tier.rawValue
        }
    }

    var thermalAndMode: String {
        let thermal: String
        switch thermalState {
        case .nominal: thermal = "Nominal"
        case .fair: thermal = "Fair"
        case .serious: thermal = "Serious"
        case .critical: thermal = "Critical"
        @unknown default: thermal = "Unknown"
        }
        return "\(thermal) • \(modeLabel)"
    }
}

final class RenderPerformanceGovernor: @unchecked Sendable {
    static let shared = RenderPerformanceGovernor()

    private let lock = NSLock()
    private var forcedSafeUntil: TimeInterval = 0
    private var previousTier: RenderPerformanceTier = .performance
    private var warningObserver: NSObjectProtocol?

    private init() {
        warningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            self?.recordMemoryWarning()
        }
    }

    deinit {
        if let warningObserver {
            NotificationCenter.default.removeObserver(warningObserver)
        }
    }

    func recordMemoryWarning() {
        lock.lock()
        forcedSafeUntil = max(forcedSafeUntil, Date().timeIntervalSince1970 + 120)
        previousTier = .safe
        lock.unlock()
        DiagnosticsLogger.shared.log("Memory governor • iOS memory warning received • forcing Memory Safe Mode for 120s")
    }

    func snapshot() -> RenderPerformanceSnapshot {
        let availableMB = Double(os_proc_available_memory()) / 1_048_576.0
        let physicalMB = Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576.0
        let thermal = ProcessInfo.processInfo.thermalState
        let now = Date().timeIntervalSince1970

        lock.lock()
        let tier: RenderPerformanceTier
        if now < forcedSafeUntil || thermal == .serious || thermal == .critical || availableMB < 700 {
            tier = .safe
        } else if availableMB < 1_400 {
            tier = .balanced
        } else {
            switch previousTier {
            case .safe where availableMB < 1_550:
                tier = .balanced
            case .balanced where availableMB < 1_650:
                tier = .balanced
            default:
                tier = .performance
            }
        }
        previousTier = tier
        lock.unlock()

        return RenderPerformanceSnapshot(
            tier: tier,
            availableMemoryMB: availableMB,
            physicalMemoryMB: physicalMB,
            thermalState: thermal
        )
    }
}

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

    var thermalState: String
    var performanceMode: String
    var availableMemoryMB: Double
    var physicalMemoryMB: Double

    init() {
        let memory = RenderPerformanceGovernor.shared.snapshot()
        thermalState = memory.thermalAndMode
        performanceMode = memory.modeLabel
        availableMemoryMB = memory.availableMemoryMB
        physicalMemoryMB = memory.physicalMemoryMB
    }

    var summary: String {
        String(
            format: "Thermal: %@\nMode: %@\nMemory headroom: %.0f MB / %.0f MB physical\nCompression: %.1f ms/src\nOutline: %.1f ms/src\nRIFE HQ: %.1f ms/gen\nReal-CUGAN: %.1f ms/frame\nGhostGuard: %.1f ms/gen\nEncode: %.1f ms/out\nRIFE speed: %.2f gen fps\nUpscale speed: %.2f fps",
            thermalState,
            performanceMode,
            availableMemoryMB,
            physicalMemoryMB,
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

@inline(__always)
func currentRenderPerformanceSnapshot() -> RenderPerformanceSnapshot {
    RenderPerformanceGovernor.shared.snapshot()
}

@inline(__always)
func automaticPerformanceModeEnabled() -> Bool {
    currentRenderPerformanceSnapshot().tier == .performance
}

@inline(__always)
func currentThermalStateName() -> String {
    currentRenderPerformanceSnapshot().thermalAndMode
}

func adaptiveFrameBoundaryPacing(_ sampled: RenderPerformanceSnapshot? = nil) async throws {
    let state = sampled ?? currentRenderPerformanceSnapshot()
    switch state.tier {
    case .performance:
        return
    case .balanced:
        try Task.checkCancellation()
        try await Task.sleep(nanoseconds: 3_000_000)
    case .safe:
        try Task.checkCancellation()
        try await Task.sleep(nanoseconds: 15_000_000)
    }
}

func thermalFrameBoundaryPacing() async throws {
    try await adaptiveFrameBoundaryPacing()
}
