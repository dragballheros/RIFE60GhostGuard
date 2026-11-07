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
    let reclaimableMemoryMB: Double
    let effectiveMemoryMB: Double
    let physicalMemoryMB: Double
    let thermalState: ProcessInfo.ThermalState

    var allowsWideCUGANTiles: Bool {
        tier == .performance && effectiveMemoryMB >= 1_700
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
    private var didFlushForSafe = false
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
        ReclaimableMemoryRegistry.shared.flushAll()
        DiagnosticsLogger.shared.log("Memory governor • iOS memory warning received • pools flushed • forcing Memory Safe Mode for 120s")
        CrashCapture.step("Memory governor • iOS memory warning • Memory Safe Mode for 120s")
    }

    func snapshot() -> RenderPerformanceSnapshot {
        var availableMB = Double(os_proc_available_memory()) / 1_048_576.0
        let physicalMB = Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576.0
        let thermal = ProcessInfo.processInfo.thermalState
        let now = Date().timeIntervalSince1970
        let reclaimableMB = Double(ReclaimableMemoryRegistry.shared.totalBytes) / 1_048_576.0

        lock.lock()
        let forced = now < forcedSafeUntil
        let thermalSafe = thermal == .serious || thermal == .critical
        if forced || thermalSafe || availableMB < 700 {
            if !didFlushForSafe {
                didFlushForSafe = true
                lock.unlock()
                ReclaimableMemoryRegistry.shared.flushAll()
                availableMB = Double(os_proc_available_memory()) / 1_048_576.0
                lock.lock()
            }
        } else {
            didFlushForSafe = false
        }
        let decision = Self.decide(
            rawMB: availableMB,
            reclaimableMB: reclaimableMB,
            thermalSerious: thermalSafe,
            forcedSafe: forced,
            previous: previousTier
        )
        let changed = decision.tier != previousTier
        previousTier = decision.tier
        lock.unlock()

        if changed {
            let line = String(
                format: "Memory governor • %@ • reason %@ • raw %.0f MB • reclaimable %.0f MB • effective %.0f MB • thermal %@",
                decision.tier.rawValue,
                decision.reason,
                availableMB,
                reclaimableMB,
                availableMB + max(0, reclaimableMB),
                thermalName(thermal)
            )
            DiagnosticsLogger.shared.log(line)
            CrashCapture.step(line)
        }

        return RenderPerformanceSnapshot(
            tier: decision.tier,
            availableMemoryMB: availableMB,
            reclaimableMemoryMB: reclaimableMB,
            effectiveMemoryMB: availableMB + max(0, reclaimableMB),
            physicalMemoryMB: physicalMB,
            thermalState: thermal
        )
    }

    /// Performance versus Balanced uses raw headroom plus reclaimable pool bytes.
    /// Safe still follows the raw 700 MB floor, thermal state, and memory warnings.
    static func decide(
        rawMB: Double,
        reclaimableMB: Double,
        thermalSerious: Bool,
        forcedSafe: Bool,
        previous: RenderPerformanceTier
    ) -> (tier: RenderPerformanceTier, reason: String) {
        if forcedSafe { return (.safe, "memory warning") }
        if thermalSerious { return (.safe, "thermal") }
        if rawMB < 700 { return (.safe, "raw floor") }
        let effective = rawMB + max(0, reclaimableMB)
        if effective < 1_400 { return (.balanced, "effective below 1400") }
        switch previous {
        case .safe where effective < 1_550:
            return (.balanced, "hysteresis from safe")
        case .balanced where effective < 1_650:
            return (.balanced, "hysteresis")
        default:
            return (.performance, "effective headroom")
        }
    }

    private func thermalName(_ thermal: ProcessInfo.ThermalState) -> String {
        switch thermal {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
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
    /// Latest Real-CUGAN frame. Upscale fps stays wall-clock; these split that wall.
    var cuganWaitInMs: Double = 0
    var cuganWaitComputeMs: Double = 0
    var cuganPredictMs: Double = 0
    var cuganTiles: Int = 0
    var cuganConcurrency: Int = 0
    var cuganStitchMs: Double = 0
    var cuganWallMs: Double = 0

    var thermalState: String
    var performanceMode: String
    var availableMemoryMB: Double
    var reclaimableMemoryMB: Double
    var effectiveMemoryMB: Double
    var physicalMemoryMB: Double

    init() {
        let memory = RenderPerformanceGovernor.shared.snapshot()
        thermalState = memory.thermalAndMode
        performanceMode = memory.modeLabel
        availableMemoryMB = memory.availableMemoryMB
        reclaimableMemoryMB = memory.reclaimableMemoryMB
        effectiveMemoryMB = memory.effectiveMemoryMB
        physicalMemoryMB = memory.physicalMemoryMB
    }

    mutating func applyCUGANBreakdown(from sample: PerformanceTelemetry) {
        cuganWaitInMs = sample.cuganWaitInMs
        cuganWaitComputeMs = sample.cuganWaitComputeMs
        cuganPredictMs = sample.cuganPredictMs
        cuganTiles = sample.cuganTiles
        cuganConcurrency = sample.cuganConcurrency
        cuganStitchMs = sample.cuganStitchMs
        cuganWallMs = sample.cuganWallMs
    }

    var summary: String {
        String(
            format: "Thermal: %@\nMode: %@\nMemory headroom: %.0f MB effective (%.0f MB raw + %.0f MB reclaimable) / %.0f MB physical\nCompression: %.1f ms/src\nOutline: %.1f ms/src\nRIFE HQ: %.1f ms/gen\nReal-CUGAN: %.1f ms/frame\nGhostGuard: %.1f ms/gen\nEncode: %.1f ms/out\nRIFE speed: %.2f gen fps\nUpscale speed (wall-clock): %.2f fps\nCUGAN breakdown: wait_in %.1f ms, wait_compute %.1f ms, predict %.1f ms (%d tiles, %d concurrent), stitch %.1f ms, wall %.1f ms",
            thermalState,
            performanceMode,
            effectiveMemoryMB,
            availableMemoryMB,
            reclaimableMemoryMB,
            physicalMemoryMB,
            compressionMsPerFrame,
            outlineMsPerFrame,
            rifeMsPerGeneratedFrame,
            cuganMsPerFrame,
            ghostMsPerGeneratedFrame,
            encodeMsPerOutputFrame,
            generatedFPS,
            upscaleFPS,
            cuganWaitInMs,
            cuganWaitComputeMs,
            cuganPredictMs,
            cuganTiles,
            cuganConcurrency,
            cuganStitchMs,
            cuganWallMs
        )
    }
}

func runtimePassDiagnosticLine(
    pass: String,
    computeUnits: String,
    decodeMs: Double,
    predictMs: Double,
    postMs: Double,
    encodeMs: Double,
    ioMs: Double
) -> String {
    let process = ProcessInfo.processInfo
    let thermal = currentThermalStateName()
    let power = process.isLowPowerModeEnabled ? "ON" : "OFF"
    let appState: String
    switch UIApplication.shared.applicationState {
    case .active: appState = "foreground"
    case .inactive: appState = "inactive"
    case .background: appState = "background"
    @unknown default: appState = "unknown"
    }
    return String(
        format: "PASS %@ • units=%@ • thermal=%@ • lowPower=%@ • app=%@ • decode=%.1fms • predict=%.1fms • post=%.1fms • encode=%.1fms • io=%.1fms",
        pass, computeUnits, thermal, power, appState, decodeMs, predictMs, postMs, encodeMs, ioMs
    )
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
    case .safe:
        try Task.checkCancellation()
        try await Task.sleep(nanoseconds: 15_000_000)
    }
}

func thermalFrameBoundaryPacing() async throws {
    try await adaptiveFrameBoundaryPacing()
}

/// Validate geometry only; adaptive RIFE uses the existing governor at runtime.
func requireUpscaleFirstMemory(width: Int, height: Int) throws {
    guard UpscaleFirstPlan.admits(width: width, height: height) else {
        throw ProcessorError.conversionFailed("Upscale-first output must be positive and no larger than 3840×2160 pixels. Completed checkpoints are retained.")
    }
    DiagnosticsLogger.shared.log("Upscale-first adaptive RIFE • headroom=\(Int(currentRenderPerformanceSnapshot().availableMemoryMB)) MB • single shared stream • no fixed 3119 MB cutoff")
}
