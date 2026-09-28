import Foundation
import SwiftUI
import UIKit
import Photos
import PhotosUI
import AVFoundation
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
    @Published var exportFolderName = "On My iPhone > RIFE 60 Ghost Guard > Exports"

    private var currentTask: Task<Void, Never>?
    private var blackScreenTask: Task<Void, Never>?
    private var savedBrightness: CGFloat?
    private var savedIdleTimerDisabled: Bool?
    private var renderStartedAt: Date?
    private let exportFolderBookmarkKey = "RIFE60.ExportFolderBookmark.v1"

    init() {
        refreshExportFolderName()
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

    func handleExportFolderSelection(_ result: Result<[URL], Error>) {
        do {
            guard let folder = try result.get().first else { return }
            let secured = folder.startAccessingSecurityScopedResource()
            defer { if secured { folder.stopAccessingSecurityScopedResource() } }
            let bookmark = try folder.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
            UserDefaults.standard.set(bookmark, forKey: exportFolderBookmarkKey)
            exportFolderName = folder.lastPathComponent
            saveStatusText = "Future videos will auto-save to Files > \(folder.lastPathComponent)"
            DiagnosticsLogger.shared.log("User selected Files export folder: \(folder.path)")
        } catch {
            saveStatusText = "Could not use that Files folder: \(error.localizedDescription)"
            DiagnosticsLogger.shared.log("Export folder selection failed: \(error.localizedDescription)")
        }
    }

    func useDefaultExportFolder() {
        UserDefaults.standard.removeObject(forKey: exportFolderBookmarkKey)
        exportFolderName = "On My iPhone > RIFE 60 Ghost Guard > Exports"
        saveStatusText = "Using the app's default Files export folder"
    }

    func cancel() { currentTask?.cancel() }

    func clearRecoveryData() {
        guard !isProcessing else { return }
        DiagnosticsLogger.shared.log("User chose Clear Recovery Data. Discarding recovery source and all checkpoints.")
        RecoveryStore.discardCurrent()
        recoveryAvailable = false
        recoveryStatusText = ""
        inputURL = nil
        outputURL = nil
        progress = 0
        restorationProgress = 0
        upscaleProgress = 0
        elapsedSeconds = 0
        etaSeconds = nil
        statusText = "Recovery data cleared • select a new video"
        errorText = nil
        saveStatusText = ""
        telemetry = PerformanceTelemetry()
    }

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
        Export folder: \(exportFolderName)
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
        let configKey = ["pipeline-v2", "hq", "ghost=\(guardEnabled)", "cuts=\(cuts)", "compression=\(compressionEnabled)", "outline=\(outlineEnabled)", String(format: "sensitivity=%.2f", sensitivity), "audio=\(audio)", "upscale=\(upscale)", "fps=60"].joined(separator: "|")
        DiagnosticsLogger.shared.log("Render requested • \(configKey)")

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }
                let job = try RecoveryStore.prepare(source: source, configurationKey: configKey)
                RecoveryStore.update(progress: job.manifest.progress, message: "Recovery source secured", force: true)
                let config = ProcessorConfiguration(quality: .hq, ghostProtection: guardEnabled, sceneCutProtection: cuts, compressionProtection: compressionEnabled, outlineProtection: outlineEnabled, ghostSensitivity: sensitivity, preserveAudio: audio, targetFPS: 60)
                let processor = TwoPassVideoProcessor(configuration: config, upscaleTo4K: upscale)
                let result = try await processor.process(sourceURL: job.sourceURL, recoveryDirectory: job.directory, progress: { p, message in
                    RecoveryStore.update(progress: p, message: message)
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.progress = p
                        self.recoveryAvailable = true
                        self.recoveryStatusText = "Crash-safe checkpoints active • \(String(format: "%.1f", p * 100))%"
                        if message.contains("Pass 1/3") { self.restorationProgress = min(max(p / 0.18, 0), 1) } else if p >= 0.19 { self.restorationProgress = 1 }
                        if p >= 0.56 { self.upscaleProgress = min(max((p - 0.56) / 0.41, 0), 1) }
                        self.statusText = message
                        self.updateClock(progress: p)
                    }
                }, telemetry: { sample in Task { @MainActor [weak self] in self?.telemetry = sample } })
                try Task.checkCancellation()
                await MainActor.run { [weak self] in self?.statusText = "Saving to Files…"; self?.progress = 0.995 }
                RecoveryStore.update(progress: 0.995, message: "Saving finished result to Files", force: true)
                guard let self else { return }
                let saved = try await self.saveFinishedVideo(result)
                RecoveryStore.finishAndClean()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.outputURL = saved.url
                    self.saveStatusText = saved.message
                    self.progress = 1; self.restorationProgress = 1; self.upscaleProgress = upscale ? 1 : 0
                    self.updateClock(progress: 1); self.etaSeconds = 0; self.statusText = "Finished"
                    self.recoveryAvailable = false; self.recoveryStatusText = ""; self.isProcessing = false; self.currentTask = nil
                    self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState()
                }
            } catch is CancellationError {
                RecoveryStore.markCancelled()
                await MainActor.run { [weak self] in guard let self else { return }; self.statusText = "Cancelled • checkpoints kept"; self.recoveryAvailable = true; self.recoveryStatusText = "Resume available from the last completed checkpoint"; self.isProcessing = false; self.currentTask = nil; self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState() }
            } catch {
                RecoveryStore.markFailed(error)
                await MainActor.run { [weak self] in guard let self else { return }; self.errorText = error.localizedDescription; self.statusText = "Failed • recovery data kept"; self.recoveryAvailable = RecoveryStore.existingJob() != nil; self.recoveryStatusText = self.recoveryAvailable ? "Resume available from the last completed checkpoint" : ""; self.isProcessing = false; self.currentTask = nil; self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState() }
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

    private func saveFinishedVideo(_ source: URL) async throws -> SavedResult {
        let saved = try saveToFiles(source)
        try await validateFinishedVideo(saved.url)
        let attrs = try FileManager.default.attributesOfItem(atPath: saved.url.path)
        let bytes = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        DiagnosticsLogger.shared.log("Files export verified • \(saved.url.lastPathComponent) • \(bytes) bytes • \(saved.url.path)")
        return SavedResult(url: saved.url, message: saved.message)
    }

    private func validateFinishedVideo(_ url: URL) async throws {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw NSError(domain: "RIFE60GhostGuard", code: 30, userInfo: [NSLocalizedDescriptionKey: "Finished export has no readable video track."]) }
        let duration = try await asset.load(.duration)
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds > 0 else { throw NSError(domain: "RIFE60GhostGuard", code: 31, userInfo: [NSLocalizedDescriptionKey: "Finished export has an invalid duration."]) }
        let size = try await track.load(.naturalSize)
        DiagnosticsLogger.shared.log("Finished export validation passed • \(Int(abs(size.width)))x\(Int(abs(size.height))) • \(String(format: "%.3f", seconds))s")
    }

    private func saveToFiles(_ source: URL) throws -> (url: URL, message: String) {
        let formatter = DateFormatter(); formatter.dateFormat = "yyyyMMdd-HHmmss"
        let ext = source.pathExtension.isEmpty ? "mov" : source.pathExtension.lowercased()
        let filename = "RIFE60-2X60-\(formatter.string(from: Date())).\(ext)"

        if let folder = resolveSelectedExportFolder() {
            let secured = folder.startAccessingSecurityScopedResource()
            defer { if secured { folder.stopAccessingSecurityScopedResource() } }
            let destination = uniqueDestination(in: folder, filename: filename)
            do {
                try FileManager.default.copyItem(at: source, to: destination)
                try verifyPersistedFile(destination)
                return (destination, "Auto-saved to Files > \(folder.lastPathComponent) > \(destination.lastPathComponent)")
            } catch {
                DiagnosticsLogger.shared.log("Selected Files folder write failed: \(error.localizedDescription). Falling back to app Exports folder.")
            }
        }

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
        let destination = uniqueDestination(in: exports, filename: filename)
        try FileManager.default.copyItem(at: source, to: destination)
        try verifyPersistedFile(destination)
        return (destination, "Auto-saved to Files: On My iPhone > RIFE 60 Ghost Guard > Exports > \(destination.lastPathComponent)")
    }

    private func verifyPersistedFile(_ url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path), let attrs = try? FileManager.default.attributesOfItem(atPath: url.path), let size = attrs[.size] as? NSNumber, size.int64Value > 0 else {
            throw NSError(domain: "RIFE60GhostGuard", code: 32, userInfo: [NSLocalizedDescriptionKey: "The finished video could not be persisted to Files."])
        }
    }

    private func uniqueDestination(in folder: URL, filename: String) -> URL {
        let fm = FileManager.default
        var candidate = folder.appendingPathComponent(filename)
        guard fm.fileExists(atPath: candidate.path) else { return candidate }
        let base = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        var index = 2
        while fm.fileExists(atPath: candidate.path) {
            candidate = folder.appendingPathComponent("\(base)-\(index).\(ext)")
            index += 1
        }
        return candidate
    }

    private func resolveSelectedExportFolder() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: exportFolderBookmarkKey) else { return nil }
        var stale = false
        do {
            let url = try URL(resolvingBookmarkData: data, options: [.withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale)
            if stale {
                let secured = url.startAccessingSecurityScopedResource()
                defer { if secured { url.stopAccessingSecurityScopedResource() } }
                let renewed = try url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
                UserDefaults.standard.set(renewed, forKey: exportFolderBookmarkKey)
            }
            return url
        } catch {
            DiagnosticsLogger.shared.log("Stored Files export bookmark could not be resolved: \(error.localizedDescription)")
            return nil
        }
    }

    private func refreshExportFolderName() {
        if let url = resolveSelectedExportFolder() { exportFolderName = url.lastPathComponent }
    }
}
