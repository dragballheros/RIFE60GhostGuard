import Foundation
import SwiftUI
import UIKit
import Photos
import PhotosUI
import CoreTransferable
import UniformTypeIdentifiers
import RifeMetal

struct PickedVideo: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in SentTransferredFile(video.url) } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent("photos-\(UUID().uuidString).\(ext)")
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: received.file, to: destination)
            return PickedVideo(url: destination)
        }
    }
}

@MainActor
final class VideoProcessorViewModel: ObservableObject {
    @Published var inputURL: URL?
    @Published var outputURL: URL?
    @Published var progress: Double = 0
    @Published var restorationProgress: Double = 0
    @Published var upscaleProgress: Double = 0
    @Published var statusText = ""
    @Published var errorText: String?
    @Published var saveStatusText = ""
    @Published var isProcessing = false
    @Published var isImporting = false
    @Published var importProgress: Double? = nil
    @Published var ghostProtection = true
    @Published var sceneCutProtection = true
    @Published var compressionProtection = true
    @Published var outlineProtection = true
    @Published var upscaleTo4K = true
    @Published var autoSaveToPhotos = true
    @Published var ghostSensitivity = 1.0
    @Published var preserveAudio = true
    @Published var renderPowerMode = true
    @Published var telemetry = PerformanceTelemetry()
    @Published var processingScreenAwake = false
    @Published var elapsedSeconds: Double = 0
    @Published var etaSeconds: Double? = nil
    @Published var recoveryAvailable = false
    @Published var recoveryStatusText = ""
    @Published var diagnosticsCopyStatus = ""

    private var currentTask: Task<Void, Never>?
    private var blackScreenTask: Task<Void, Never>?
    private var savedBrightness: CGFloat?
    private var savedIdleTimerDisabled: Bool?
    private var renderStartedAt: Date?

    init() {
        if let job = RecoveryStore.existingJob() {
            inputURL = job.sourceURL
            progress = job.manifest.progress
            recoveryAvailable = true
            recoveryStatusText = "Interrupted render recovered at \(String(format: "%.1f", job.manifest.progress * 100))% • \(job.manifest.lastMessage)"
            statusText = "Recovery ready"
            DiagnosticsLogger.shared.log("App relaunched with unfinished recovery job at \(Int(job.manifest.progress * 100))%. Previous process may have been terminated or crashed.")
        } else {
            DiagnosticsLogger.shared.log("App launched.")
        }
    }

    func handleImport(_ result: Result<[URL], Error>) {
        isImporting = true
        importProgress = 0
        defer { isImporting = false; importProgress = nil }
        do {
            guard let url = try result.get().first else { return }
            importProgress = 0.5
            RecoveryStore.discardCurrent()
            recoveryAvailable = false
            recoveryStatusText = ""
            inputURL = url
            outputURL = nil
            errorText = nil
            saveStatusText = ""
            importProgress = 1.0
            DiagnosticsLogger.shared.log("Selected input from Files: \(url.lastPathComponent)")
        } catch {
            errorText = error.localizedDescription
            DiagnosticsLogger.shared.log("Files import failed: \(error.localizedDescription)")
        }
    }

    func handlePhotoSelection(_ item: PhotosPickerItem) async {
        isImporting = true
        importProgress = nil
        do {
            statusText = "Importing from Photos…"
            guard let picked = try await item.loadTransferable(type: PickedVideo.self) else {
                throw NSError(domain: "RIFE60GhostGuard", code: 1, userInfo: [NSLocalizedDescriptionKey: "The selected Photos video could not be loaded."])
            }
            RecoveryStore.discardCurrent()
            recoveryAvailable = false
            recoveryStatusText = ""
            inputURL = picked.url
            outputURL = nil
            errorText = nil
            saveStatusText = ""
            statusText = "Import complete"
            DiagnosticsLogger.shared.log("Selected input from Photos: \(picked.url.lastPathComponent)")
        } catch {
            errorText = error.localizedDescription
            statusText = ""
            DiagnosticsLogger.shared.log("Photos import failed: \(error.localizedDescription)")
        }
        isImporting = false
        importProgress = nil
    }

    func cancel() { currentTask?.cancel() }

    func copyErrorLogs() {
        var report = DiagnosticsLogger.shared.text()
        report += """

        Current UI state
        Processing: \(isProcessing)
        Progress: \(String(format: "%.2f", progress * 100))%
        Status: \(statusText)
        Last error: \(errorText ?? "none")
        Recovery available: \(recoveryAvailable)
        Recovery status: \(recoveryStatusText)
        Thermal: \(telemetry.thermalState)
        Source frames: \(telemetry.sourceFrames)
        Generated frames: \(telemetry.generatedFrames)
        Upscaled frames: \(telemetry.upscaledFrames)
        RIFE gen fps: \(String(format: "%.3f", telemetry.generatedFPS))
        Real-CUGAN fps: \(String(format: "%.3f", telemetry.upscaleFPS))
        """
        UIPasteboard.general.string = report
        diagnosticsCopyStatus = "Error logs copied to clipboard"
    }

    func wakeProcessingScreen() {
        guard isProcessing else { return }
        processingScreenAwake = true
        if renderPowerMode { UIScreen.main.brightness = 0.05 }
        blackScreenTask?.cancel()
        blackScreenTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 20_000_000_000)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.isProcessing else { return }
                self.processingScreenAwake = false
                if self.renderPowerMode { UIScreen.main.brightness = 0.01 }
            }
        }
    }

    func handleScenePhase(_ phase: ScenePhase) {
        switch phase {
        case .active:
            if isProcessing && renderPowerMode { applyRenderPowerMode() }
        case .inactive, .background:
            restoreDisplayState()
        @unknown default: break
        }
    }

    func start() async {
        guard let source = inputURL, !isProcessing else { return }
        isProcessing = true
        processingScreenAwake = false
        if !recoveryAvailable {
            progress = 0
            restorationProgress = 0
            upscaleProgress = 0
        }
        telemetry = PerformanceTelemetry()
        outputURL = nil
        errorText = nil
        saveStatusText = ""
        diagnosticsCopyStatus = ""
        elapsedSeconds = 0
        etaSeconds = nil
        renderStartedAt = Date()
        statusText = recoveryAvailable ? "Validating crash-recovery checkpoints…" : "Creating crash-safe source checkpoint…"
        if renderPowerMode { applyRenderPowerMode() }

        let guardEnabled = ghostProtection
        let cuts = sceneCutProtection
        let compressionEnabled = compressionProtection
        let outlineEnabled = outlineProtection
        let sensitivity = ghostSensitivity
        let audio = preserveAudio
        let upscale = upscaleTo4K
        let autoPhotos = autoSaveToPhotos
        let configKey = [
            "pipeline-v2",
            "hq",
            "ghost=\(guardEnabled)",
            "cuts=\(cuts)",
            "compression=\(compressionEnabled)",
            "outline=\(outlineEnabled)",
            String(format: "sensitivity=%.2f", sensitivity),
            "audio=\(audio)",
            "upscale=\(upscale)",
            "fps=60"
        ].joined(separator: "|")

        DiagnosticsLogger.shared.log("Render requested • \(configKey)")

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }

                let job = try RecoveryStore.prepare(source: source, configurationKey: configKey)
                RecoveryStore.update(progress: job.manifest.progress, message: "Recovery source secured", force: true)

                let config = ProcessorConfiguration(
                    quality: .hq,
                    ghostProtection: guardEnabled,
                    sceneCutProtection: cuts,
                    compressionProtection: compressionEnabled,
                    outlineProtection: outlineEnabled,
                    ghostSensitivity: sensitivity,
                    preserveAudio: audio,
                    targetFPS: 60
                )
                let processor = TwoPassVideoProcessor(configuration: config, upscaleTo4K: upscale)
                let result = try await processor.process(
                    sourceURL: job.sourceURL,
                    recoveryDirectory: job.directory,
                    progress: { p, message in
                        RecoveryStore.update(progress: p, message: message)
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.progress = p
                            self.recoveryAvailable = true
                            self.recoveryStatusText = "Crash-safe checkpoints active • \(String(format: "%.1f", p * 100))%"
                            if message.contains("Pass 1/3") {
                                self.restorationProgress = min(max(p / 0.18, 0), 1)
                            } else if p >= 0.19 { self.restorationProgress = 1 }
                            if p >= 0.56 {
                                self.upscaleProgress = min(max((p - 0.56) / 0.41, 0), 1)
                            }
                            self.statusText = message
                            self.updateClock(progress: p)
                        }
                    },
                    telemetry: { sample in
                        Task { @MainActor [weak self] in self?.telemetry = sample }
                    }
                )
                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    self?.statusText = "Saving result…"
                    self?.progress = 0.995
                }
                RecoveryStore.update(progress: 0.995, message: "Saving finished result", force: true)
                let saved = try await self?.saveFinishedVideo(result, preferPhotos: autoPhotos)
                RecoveryStore.finishAndClean()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.outputURL = saved?.url ?? result
                    self.saveStatusText = saved?.message ?? "Finished"
                    self.progress = 1
                    self.restorationProgress = 1
                    self.upscaleProgress = upscale ? 1 : 0
                    self.updateClock(progress: 1)
                    self.etaSeconds = 0
                    self.statusText = "Finished"
                    self.recoveryAvailable = false
                    self.recoveryStatusText = ""
                    self.isProcessing = false
                    self.currentTask = nil
                    self.blackScreenTask?.cancel()
                    self.processingScreenAwake = true
                    self.restoreDisplayState()
                }
            } catch is CancellationError {
                RecoveryStore.markCancelled()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.statusText = "Cancelled • checkpoints kept"
                    self.recoveryAvailable = true
                    self.recoveryStatusText = "Resume available from the last completed checkpoint"
                    self.isProcessing = false
                    self.currentTask = nil
                    self.blackScreenTask?.cancel()
                    self.processingScreenAwake = true
                    self.restoreDisplayState()
                }
            } catch {
                RecoveryStore.markFailed(error)
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.errorText = error.localizedDescription
                    self.statusText = "Failed • recovery data kept"
                    self.recoveryAvailable = RecoveryStore.existingJob() != nil
                    self.recoveryStatusText = self.recoveryAvailable ? "Resume available from the last completed checkpoint" : ""
                    self.isProcessing = false
                    self.currentTask = nil
                    self.blackScreenTask?.cancel()
                    self.processingScreenAwake = true
                    self.restoreDisplayState()
                }
            }
        }
    }

    private func updateClock(progress: Double) {
        guard let renderStartedAt else { return }
        let elapsed = Date().timeIntervalSince(renderStartedAt)
        elapsedSeconds = elapsed
        if progress > 0.025 && progress < 0.995 {
            let raw = elapsed * (1.0 - progress) / progress
            etaSeconds = etaSeconds.map { $0 > 0 ? $0 * 0.72 + raw * 0.28 : raw } ?? raw
        }
    }

    private func applyRenderPowerMode() {
        if savedBrightness == nil { savedBrightness = UIScreen.main.brightness }
        if savedIdleTimerDisabled == nil { savedIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled }
        UIScreen.main.brightness = processingScreenAwake ? 0.05 : 0.01
        UIApplication.shared.isIdleTimerDisabled = true
    }

    private func restoreDisplayState() {
        if let savedBrightness { UIScreen.main.brightness = savedBrightness; self.savedBrightness = nil }
        if let savedIdleTimerDisabled { UIApplication.shared.isIdleTimerDisabled = savedIdleTimerDisabled; self.savedIdleTimerDisabled = nil }
    }

    private struct SavedResult: Sendable { let url: URL; let message: String }

    private func saveFinishedVideo(_ source: URL, preferPhotos: Bool) async throws -> SavedResult {
        if preferPhotos {
            do {
                try await saveToPhotos(source)
                return SavedResult(url: source, message: "Automatically saved to Photos")
            } catch {
                DiagnosticsLogger.shared.log("Photos save failed: \(error.localizedDescription). Falling back to Files.")
                let fallback = try saveToFiles(source)
                return SavedResult(url: fallback, message: "Photos save failed, so the video was saved to Files: On My iPhone > RIFE 60 Ghost Guard > Exports > \(fallback.lastPathComponent)")
            }
        }
        let fallback = try saveToFiles(source)
        return SavedResult(url: fallback, message: "Saved to Files: On My iPhone > RIFE 60 Ghost Guard > Exports > \(fallback.lastPathComponent)")
    }

    private func saveToPhotos(_ url: URL) async throws {
        var status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        if status == .notDetermined { status = await PHPhotoLibrary.requestAuthorization(for: .addOnly) }
        guard status == .authorized || status == .limited else {
            throw NSError(domain: "RIFE60GhostGuard", code: 20, userInfo: [NSLocalizedDescriptionKey: "Photos add permission was not granted."])
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            PHPhotoLibrary.shared().performChanges({ PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url) }) { success, error in
                if success { continuation.resume() }
                else { continuation.resume(throwing: error ?? NSError(domain: "RIFE60GhostGuard", code: 21, userInfo: [NSLocalizedDescriptionKey: "Photos could not save the finished video."])) }
            }
        }
    }

    private func saveToFiles(_ source: URL) throws -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let destination = exports.appendingPathComponent("RIFE60-4K60-\(formatter.string(from: Date())).mov")
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.copyItem(at: source, to: destination)
        return destination
    }
}
