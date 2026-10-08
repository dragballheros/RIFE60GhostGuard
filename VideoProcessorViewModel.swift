import Foundation
import SwiftUI
import UIKit
import Photos
import PhotosUI
import AVFoundation
import CoreTransferable
import ImageIO
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

struct PickedImage: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .image) { received in
            let ext = received.file.pathExtension.isEmpty ? "heic" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent("photos-image-\(UUID().uuidString).\(ext)")
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: received.file, to: destination)
            return PickedImage(url: destination)
        }
    }
}

@MainActor
final class VideoProcessorViewModel: ObservableObject {
    @Published var inputURL: URL?
    @Published var inputKind: InputMediaKind = .video
    @Published var outputURL: URL?
    @Published var progress: Double = 0
    @Published var restorationProgress: Double = 0
    @Published var upscaleProgress: Double = 0
    @Published var statusText = ""
    @Published var errorText: String?
    @Published var saveStatusText = ""
    @Published var isProcessing = false
    @Published private(set) var isPaused = false
    @Published var isImporting = false
    @Published var importProgress: Double? = nil
    @Published var ghostProtection = true
    @Published var sceneCutProtection = true
    @Published var compressionProtection = true
    @Published var outlineProtection = true
    @Published var watermarkRemovalEnabled = false
    @Published var watermarkRegions: [WatermarkRegion] = [] { didSet { saveSelectedMask() } }
    @Published var watermarkPaddingPixels = 1.0
    @Published var colorPopEnabled = true
    @Published var colorPopStrength = 1.0
    @Published var upscaleTo4K = true
    @Published var upscaleFirst = false
    @Published var computePreference = ModelComputePreference.current { didSet { computePreference.persist() } }
    @Published var isBenchmarking = false
    @Published var benchmarkProgress: Double = 0
    @Published var benchmarkStatus = ""
    @Published var benchmarkReport = ""
    private var benchmarkTask: Task<Void, Never>?
    @Published var ghostSensitivity = 1.0
    @Published var preserveAudio = true
    @Published var redditMode = false
    @Published var videoToGIFEnabled = false
    @Published var renderPowerMode = true
    @Published var telemetry = PerformanceTelemetry()
    @Published var processingScreenAwake = false
    @Published var elapsedSeconds: Double = 0
    @Published var etaSeconds: Double? = nil
    @Published var recoveryAvailable = false
    @Published var recoveryStatusText = ""
    @Published var diagnosticsCopyStatus = ""
    @Published var exportFolderName = "On My iPhone > RIFE 60 Ghost Guard > Exports"

    @Published var mediaQueue: BatchMediaManifest?
    @Published var selectedMediaID: UUID?
    @Published var isBatchProcessing = false
    @Published var batchStatusText = ""
    private var batchCancelRequested = false
    private var loadingQueueSelection = false
    private let batchStore = BatchMediaQueueStore()
    var isBusy: Bool { isProcessing || isBatchProcessing }
    var queueLocked: Bool {
        isBusy || recoveryAvailable || (mediaQueue?.settingsKey != nil && mediaQueue?.nextItemID != nil)
    }
    var selectedQueueItem: BatchMediaItem? { mediaQueue?.items.first { $0.id == selectedMediaID } }
    var queueHasRemaining: Bool { mediaQueue?.nextItemID != nil }
    var batchProgress: Double {
        guard let queue = mediaQueue, !queue.items.isEmpty else { return progress }
        let complete = queue.items.filter { $0.state == .completed }.count
        let active = isBatchProcessing && selectedQueueItem?.state == .processing ? min(max(progress, 0), 1) : 0
        return min(1, (Double(complete) + active) / Double(queue.items.count))
    }
    private var effectiveWatermarkEnabled: Bool { watermarkRemovalEnabled && (selectedQueueItem?.removeWatermark ?? true) }

    private var renderGeneration = UUID()
    private var currentTask: Task<Void, Never>?
    private var currentJobForcesGIF = false
    private var blackScreenTask: Task<Void, Never>?
    private var savedBrightness: CGFloat?
    private var savedIdleTimerDisabled: Bool?
    private var renderStartedAt: Date?
    private var resumeBaseProgress: Double?
    private let exportFolderBookmarkKey = "RIFE60.ExportFolderBookmark.v1"

    init() {
        refreshExportFolderName()
        if let job = RecoveryStore.existingJob() {
            // The toggles are greyed out while recovery data exists, so they MUST show the
            // settings the interrupted job was started with. Previously they reset to defaults
            // on relaunch, the configuration key no longer matched, and RecoveryStore wiped
            // every checkpoint while the UI still claimed a resume was available.
            applyRecoveredSettings(from: job.manifest.configurationKey)
            inputKind = .video
            inputURL = job.sourceURL
            progress = job.manifest.progress
            recoveryAvailable = true
            recoveryStatusText = "Interrupted render recovered at \(String(format: "%.1f", job.manifest.progress * 100))% • \(job.manifest.lastMessage)"
            statusText = "Recovery ready"
            DiagnosticsLogger.shared.log("App relaunched with unfinished recovery job at \(Int(job.manifest.progress * 100))%. Previous process may have been terminated or crashed.")
        } else {
            DiagnosticsLogger.shared.log("App launched.")
        }
        restoreMediaQueue()
    }

    func handleImport(_ result: Result<[URL], Error>) async {
        guard !isBusy, !isImporting, !isBenchmarking else { return }
        isImporting = true; importProgress = 0
        defer { isImporting = false; importProgress = nil }
        do {
            let urls = try result.get()
            guard !urls.isEmpty else { return }
            try await replaceMediaQueue(with: urls)
        } catch {
            errorText = error.localizedDescription
            DiagnosticsLogger.shared.log("Files import failed: \(error.localizedDescription)")
        }
    }

    func handlePhotoSelections(_ items: [PhotosPickerItem]) async {
        guard !items.isEmpty, !isBusy, !isImporting, !isBenchmarking else { return }
        isImporting = true; importProgress = 0
        var imported: [URL] = []
        defer {
            for url in imported { try? FileManager.default.removeItem(at: url) }
            isImporting = false; importProgress = nil
        }
        do {
            for (index, item) in items.enumerated() {
                statusText = "Importing Photos item \(index + 1) of \(items.count)…"
                let isVideo = item.supportedContentTypes.contains { $0.conforms(to: .movie) || $0.conforms(to: .video) }
                if !isVideo && item.supportedContentTypes.contains(where: { $0.conforms(to: .image) }) {
                    guard let picked = try await item.loadTransferable(type: PickedImage.self) else {
                        throw ProcessorError.conversionFailed("Photos image \(index + 1) could not be loaded")
                    }
                    imported.append(picked.url)
                } else {
                    guard let picked = try await item.loadTransferable(type: PickedVideo.self) else {
                        throw ProcessorError.conversionFailed("Photos video \(index + 1) could not be loaded")
                    }
                    imported.append(picked.url)
                }
                importProgress = Double(index + 1) / Double(items.count) * 0.5
            }
            try await replaceMediaQueue(with: imported)
        } catch {
            errorText = error.localizedDescription
            DiagnosticsLogger.shared.log("Photos import failed: \(error.localizedDescription)")
        }
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

    func benchmarkModels() {
        guard !isBusy, !isBenchmarking, !isImporting else { return }
        isBenchmarking = true
        benchmarkProgress = 0
        benchmarkStatus = "Preparing model benchmark… Keep the app open."
        benchmarkReport = ""
        benchmarkTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let report = try ModelBenchmark.run(progress: { fraction, message in
                    Task { @MainActor [weak self] in
                        self?.benchmarkProgress = fraction
                        self?.benchmarkStatus = message
                    }
                })
                await MainActor.run { [weak self] in
                    self?.benchmarkReport = report
                    self?.benchmarkStatus = "Benchmark complete • Copy Error Logs to share the table"
                    self?.isBenchmarking = false
                    self?.benchmarkTask = nil
                }
            } catch {
                DiagnosticsLogger.shared.log("Benchmark stopped • \(error.localizedDescription)")
                await MainActor.run { [weak self] in
                    self?.benchmarkStatus = error is CancellationError ? "Benchmark cancelled" : error.localizedDescription
                    self?.isBenchmarking = false
                    self?.benchmarkTask = nil
                }
            }
        }
    }

    func cancelBenchmark() { benchmarkTask?.cancel() }

    /// Pause is the only action that requests an incremental checkpoint.
    /// The worker is cancelled immediately, but the active stage writer intercepts
    /// that cancellation and finalizes its current encoded progress before returning.
    func pause() {
        guard isBusy else { return }
        batchCancelRequested = true
        isPaused = true
        PauseCheckpointCoordinator.shared.request()
        DiagnosticsLogger.shared.log("User paused render • requesting pause-only incremental stage checkpoint.")
        currentTask?.cancel()
    }

    func cancel() {
        batchCancelRequested = true
        isPaused = false
        PauseCheckpointCoordinator.shared.clear()
        DiagnosticsLogger.shared.log("User cancelled render • no incremental checkpoint requested.")
        currentTask?.cancel()
    }
    func clearRecoveryData() {
        guard !isBusy else { return }
        do { try batchStore.discard(); mediaQueue = nil; selectedMediaID = nil; batchStatusText = "" }
        catch { errorText = error.localizedDescription; return }
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
        Performance tier: \(telemetry.performanceMode)
        Memory headroom: \(String(format: "%.0f", telemetry.availableMemoryMB)) MB
        Physical memory: \(String(format: "%.0f", telemetry.physicalMemoryMB)) MB
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

    private func startSingle() async -> Task<Void, Never>? {
        guard let selectedSource = inputURL, !isProcessing, !isBenchmarking else { return nil }
        let watermark = WatermarkConfiguration(enabled: effectiveWatermarkEnabled, regions: watermarkRegions, paddingPixels: Int(watermarkPaddingPixels))
        guard watermark.isValid else {
            errorText = "Mark at least one watermark region before starting removal."
            return nil
        }
        if inputKind == .image {
            return await startImage(source: selectedSource)
        }

        let inputWasGIF = inputKind == .gif
        currentJobForcesGIF = inputWasGIF || videoToGIFEnabled
        if inputWasGIF {
            // GIFs now enter the exact same video pipeline as normal video. The only
            // format-specific step is the temporary GIF -> MP4 bridge before rendering.
            videoToGIFEnabled = true
        }

        let resumingExistingJob = recoveryAvailable
        let generation = UUID(); renderGeneration = generation
        PauseCheckpointCoordinator.shared.clear()
        resumeBaseProgress = nil
        isProcessing = true
        isPaused = false
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
        let colorPop = colorPopEnabled
        let gradeStrength = colorPopStrength
        let audio = preserveAudio
        let upscale = upscaleTo4K
        let upscaleBeforeRIFE = upscaleFirst && upscale
        let modelCompute = computePreference.rawValue
        let configKey = ["pipeline-v4-gif-video", "hq", "ghost=\(guardEnabled)", "cuts=\(cuts)", "compression=\(compressionEnabled)", "outline=\(outlineEnabled)", "colorpop=\(colorPop)", "colorpopStrength=\(gradeStrength)", "watermark=\(watermark.enabled)", "watermarkMasks=\(watermark.serializedRegions)", "watermarkPadding=\(watermark.paddingPixels)", "watermarkModel=\(WatermarkConfiguration.modelID)", String(format: "sensitivity=%.2f", sensitivity), "audio=\(audio)", "reddit=\(redditMode)", "videoToGIF=\(videoToGIFEnabled)", "upscale=\(upscale)", "upscaleFirst=\(upscaleBeforeRIFE)", "computeUnits=\(modelCompute)", "fps=60"].joined(separator: "|")
        DiagnosticsLogger.shared.log("Render requested • \(configKey)")

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var processingSource = selectedSource
                let secured = selectedSource.startAccessingSecurityScopedResource()
                defer { if secured { selectedSource.stopAccessingSecurityScopedResource() } }

                if inputWasGIF {
                    await MainActor.run { [weak self] in
                        guard let self, self.renderGeneration == generation, self.isProcessing else { return }
                        self.statusText = "Converting GIF to temporary video…"
                        self.progress = 0.01
                        self.updateClock(progress: self.progress)
                    }
                    processingSource = try await GIFVideoBridge.makeVideo(from: selectedSource) { p, message in
                        Task { @MainActor [weak self] in
                            guard let self, self.renderGeneration == generation, self.isProcessing else { return }
                            self.progress = min(0.08, max(0.01, p * 0.08))
                            self.statusText = message
                            self.updateClock(progress: self.progress)
                        }
                    }
                    try Task.checkCancellation()
                    DiagnosticsLogger.shared.log("GIF input bridged to video • source=\(selectedSource.lastPathComponent) • bridge=\(processingSource.lastPathComponent)")
                    await MainActor.run { [weak self] in
                        guard let self, self.renderGeneration == generation, self.isProcessing else { return }
                        self.statusText = "GIF converted • starting full RIFE/CUGAN video pipeline…"
                        self.progress = 0.08
                        self.updateClock(progress: self.progress)
                    }
                }

                let job = try RecoveryStore.prepare(source: processingSource, configurationKey: configKey)
                RecoveryStore.update(progress: job.manifest.progress, message: "Recovery source secured", force: true)
                guard let self else { return }
                if resumingExistingJob && !job.resumed {
                    // Be honest: the saved checkpoints could not be matched to this run, so nothing is reused.
                    DiagnosticsLogger.shared.log("Recovery data did not match this pipeline/configuration; starting from the beginning.")
                    await MainActor.run {
                        self.progress = 0
                        self.restorationProgress = 0
                        self.upscaleProgress = 0
                        self.statusText = "Saved checkpoints came from a different configuration • starting fresh"
                    }
                }

                let result: URL
                if let deliveryReady = self.existingDeliveryCheckpoint(in: job.directory) {
                    DiagnosticsLogger.shared.log("Recovery: reused delivery-ready checkpoint; skipping all render and mux stages.")
                    RecoveryStore.update(progress: 0.995, message: "Recovered delivery-ready checkpoint • retrying Files export only", force: true)
                    await MainActor.run {
                        self.progress = 0.995
                        self.statusText = "Recovered finished video • retrying Files export only…"
                        self.etaSeconds = nil
                    }
                    result = deliveryReady
                } else {
                    let config = ProcessorConfiguration(quality: .hq, ghostProtection: guardEnabled, sceneCutProtection: cuts, compressionProtection: compressionEnabled, outlineProtection: outlineEnabled, ghostSensitivity: sensitivity, preserveAudio: audio, targetFPS: 60)
                    let processor = TwoPassVideoProcessor(configuration: config, upscaleTo4K: upscale, colorPopEnabled: colorPop, colorPopStrength: gradeStrength, watermark: watermark, upscaleFirst: upscaleBeforeRIFE)
                    let generated = try await processor.process(sourceURL: job.sourceURL, recoveryDirectory: job.directory, progress: { p, message in
                        RecoveryStore.update(progress: p, message: message)
                        Task { @MainActor [weak self] in
                            guard let self, self.renderGeneration == generation, self.isProcessing else { return }
                            self.progress = p
                            self.recoveryAvailable = true
                            self.recoveryStatusText = "Crash-safe checkpoints active • \(String(format: "%.1f", p * 100))%"
                            if message.contains("Pass 1/3") { self.restorationProgress = min(max(p / 0.18, 0), 1) } else if p >= 0.19 { self.restorationProgress = 1 }
                            if p >= 0.56 { self.upscaleProgress = min(max((p - 0.56) / 0.41, 0), 1) }
                            self.statusText = message
                            self.updateClock(progress: p)
                        }
                    }, telemetry: { sample in Task { @MainActor [weak self] in
                        guard let self, self.renderGeneration == generation, self.isProcessing else { return }
                        self.telemetry = sample
                    } })
                    try Task.checkCancellation()
                    result = try self.persistDeliveryCheckpoint(from: generated, in: job.directory)
                    RecoveryStore.update(progress: 0.97, message: "High-quality master checkpointed • ready for final size/export", force: true)
                }

                await MainActor.run {
                    self.statusText = "Checking final file size…"
                    self.progress = max(self.progress, 0.97)
                    self.etaSeconds = nil
                }
                let saved = try await self.saveFinishedVideo(result)
                if inputWasGIF {
                    // The bridge is only an internal processing source. The final user-facing
                    // asset is the GIF produced from the fully processed video master.
                    try? FileManager.default.removeItem(at: processingSource)
                }
                RecoveryStore.finishAndClean()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.outputURL = saved.url
                    self.saveStatusText = saved.message
                    self.progress = 1; self.restorationProgress = 1; self.upscaleProgress = upscale ? 1 : 0
                    self.updateClock(progress: 1); self.etaSeconds = 0; self.statusText = "Finished"
                    self.recoveryAvailable = false; self.recoveryStatusText = ""; self.isProcessing = false; self.currentTask = nil; self.currentJobForcesGIF = false
                    self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState()
                }
            } catch is CancellationError {
                RecoveryStore.markCancelled()
                await MainActor.run { [weak self] in guard let self else { return }; self.statusText = self.isPaused ? "Paused • stage progress checkpoint saved" : "Cancelled • checkpoints kept"; self.recoveryAvailable = true; self.recoveryStatusText = self.isPaused ? "Resume available from the saved point in the interrupted stage" : "Resume available from the last completed checkpoint"; self.isProcessing = false; self.currentTask = nil; self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState() }
            } catch {
                RecoveryStore.markFailed(error)
                await MainActor.run { [weak self] in guard let self else { return }; self.errorText = error.localizedDescription; self.statusText = "Failed • recovery data kept"; self.recoveryAvailable = RecoveryStore.existingJob() != nil; self.recoveryStatusText = self.recoveryAvailable ? "Resume available from the last completed checkpoint" : ""; self.isProcessing = false; self.currentTask = nil; self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState() }
            }
        }
        return currentTask
    }

    private func applyRecoveredSettings(from key: String) {
        // Build 162 manifests predate Color Pop; missing fields mean the original look.
        upscaleFirst = false
        computePreference = .auto
        colorPopEnabled = false
        colorPopStrength = 0.5
        watermarkRemovalEnabled = false
        watermarkRegions = []
        watermarkPaddingPixels = 3
        for part in key.split(separator: "|") {
            let pair = part.split(separator: "=", maxSplits: 1).map(String.init)
            guard pair.count == 2 else { continue }
            let enabled = pair[1] == "true"
            switch pair[0] {
            case "ghost": ghostProtection = enabled
            case "cuts": sceneCutProtection = enabled
            case "compression": compressionProtection = enabled
            case "outline": outlineProtection = enabled
            case "watermark": watermarkRemovalEnabled = enabled
            case "watermarkMasks": watermarkRegions = WatermarkConfiguration.decodeRegions(pair[1])
            case "watermarkPadding":
                if let value = Double(pair[1]), value.isFinite { watermarkPaddingPixels = min(max(value.rounded(), 0), 16) }
            case "colorpop": colorPopEnabled = enabled
            case "colorpopStrength":
                if let value = Double(pair[1]), value.isFinite { colorPopStrength = min(max(value, 0), 1) }
            case "audio": preserveAudio = enabled
            case "reddit": redditMode = enabled
            case "videoToGIF": videoToGIFEnabled = enabled
            case "upscaleFirst": upscaleFirst = enabled
            case "computeUnits": computePreference = ModelComputePreference(rawValue: pair[1]) ?? .auto
            case "upscale": upscaleTo4K = enabled
            case "sensitivity": if let value = Double(pair[1]) { ghostSensitivity = value }
            default: break
            }
        }
    }

    // MARK: - GIF inputs now use the normal video pipeline

    // MARK: - Still images (RIFE is skipped entirely)

    private func startImage(source: URL) async -> Task<Void, Never>? {
        let generation = UUID(); renderGeneration = generation
        isProcessing = true
        isPaused = false
        processingScreenAwake = false
        progress = 0
        restorationProgress = 0
        upscaleProgress = 0
        telemetry = PerformanceTelemetry()
        outputURL = nil
        errorText = nil
        saveStatusText = ""
        diagnosticsCopyStatus = ""
        elapsedSeconds = 0
        etaSeconds = nil
        resumeBaseProgress = nil
        renderStartedAt = Date()
        statusText = "Preparing image…"
        if renderPowerMode { applyRenderPowerMode() }

        let compressionEnabled = compressionProtection
        let outlineEnabled = outlineProtection
        let upscale = upscaleTo4K
        let colorPop = colorPopEnabled
        let gradeStrength = colorPopStrength
        let watermark = WatermarkConfiguration(enabled: effectiveWatermarkEnabled, regions: watermarkRegions, paddingPixels: Int(watermarkPaddingPixels))
        DiagnosticsLogger.shared.log("Image render requested • RIFE skipped • compression=\(compressionEnabled) • cugan2x=\(upscale) • outline=\(outlineEnabled)")

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }
                let processor = ImageStillProcessor(compressionProtection: compressionEnabled, outlineProtection: outlineEnabled, upscale2x: upscale, colorPopEnabled: colorPop, colorPopStrength: gradeStrength, watermark: watermark)
                let generated = try await processor.process(sourceURL: source, progress: { p, message in
                    Task { @MainActor [weak self] in
                        guard let self, self.isProcessing, self.renderGeneration == generation else { return }
                        self.progress = p
                        self.statusText = message
                        self.updateClock(progress: p)
                    }
                })
                try Task.checkCancellation()
                guard let self else { return }
                let saved = try await self.saveFinishedImage(generated.url, notes: generated.notes)
                // Only delete the imported copy if WE created it (Photos import); never touch a Files original.
                if source.lastPathComponent.hasPrefix("photos-image-") { try? FileManager.default.removeItem(at: source) }
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.outputURL = saved.url
                    self.saveStatusText = saved.message
                    self.progress = 1; self.restorationProgress = 1; self.upscaleProgress = upscale ? 1 : 0
                    self.updateClock(progress: 1); self.etaSeconds = 0; self.statusText = "Finished"
                    self.isProcessing = false; self.currentTask = nil
                    self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState()
                }
            } catch is CancellationError {
                DiagnosticsLogger.shared.log("Image render cancelled.")
                await MainActor.run { [weak self] in guard let self else { return }; self.statusText = "Cancelled"; self.isProcessing = false; self.currentTask = nil; self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState() }
            } catch {
                DiagnosticsLogger.shared.log("Image render failed: \(error.localizedDescription)")
                await MainActor.run { [weak self] in guard let self else { return }; self.errorText = error.localizedDescription; self.statusText = "Failed"; self.isProcessing = false; self.currentTask = nil; self.blackScreenTask?.cancel(); self.processingScreenAwake = true; self.restoreDisplayState() }
            }
        }
        return currentTask
    }

    private func saveFinishedImage(_ source: URL, notes: [String]) async throws -> SavedResult {
        if redditMode {
            statusText = "Preparing highest-quality Reddit image under 20 MB…"
            progress = max(progress, 0.97)
            let optimizer = RedditMediaOptimizer()
            let result = try await optimizer.optimizeImage(sourceURL: source) { [weak self] p, message in
                Task { @MainActor in
                    guard let self else { return }
                    self.progress = 0.97 + min(max(p, 0), 1) * 0.02
                    self.statusText = message
                }
            }
            let saved = try await saveRedditAsset(result, filenamePrefix: "RIFE60-Reddit-Image")
            if result.url != source { try? FileManager.default.removeItem(at: result.url) }
            if source != saved.url { try? FileManager.default.removeItem(at: source) }
            DiagnosticsLogger.shared.log("Reddit image export • \(result.processing.rawValue) • source=\(result.originalBytes) bytes • output=\(result.bytes) bytes • \(result.summary)")
            return SavedResult(
                url: saved.url,
                message: "\(result.summary) • saved \(saved.bytes) bytes • \(saved.message)"
            )
        }

        statusText = "Saving image to Files…"
        progress = max(progress, 0.97)
        let formatter = DateFormatter(); formatter.dateFormat = "yyyyMMdd-HHmmss"
        let ext = source.pathExtension.isEmpty ? "png" : source.pathExtension.lowercased()
        let filename = "RIFE60-Image-\(formatter.string(from: Date())).\(ext)"
        let noteText = notes.isEmpty ? "" : " • " + notes.joined(separator: " • ")

        if let folder = resolveSelectedExportFolder() {
            let secured = folder.startAccessingSecurityScopedResource()
            if secured {
                defer { folder.stopAccessingSecurityScopedResource() }
                do {
                    let destination = uniqueDestination(in: folder, filename: filename)
                    let persisted = try coordinatedCopy(source, to: destination, inside: folder)
                    try verifyPersistedFile(persisted)
                    try validateFinishedImage(persisted)
                    try? FileManager.default.removeItem(at: source)
                    return SavedResult(url: persisted, message: "Files > \(folder.lastPathComponent) > \(persisted.lastPathComponent)\(noteText)")
                } catch {
                    DiagnosticsLogger.shared.log("Selected Files folder image write/validation failed: \(error.localizedDescription). Falling back to app Exports folder.")
                }
            } else {
                DiagnosticsLogger.shared.log("Selected Files folder security scope could not be activated for image export. Falling back to app Exports folder.")
            }
        }

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
        let destination = uniqueDestination(in: exports, filename: filename)
        try FileManager.default.copyItem(at: source, to: destination)
        try verifyPersistedFile(destination)
        try validateFinishedImage(destination)
        try? FileManager.default.removeItem(at: source)
        return SavedResult(url: destination, message: "On My iPhone > RIFE 60 Ghost Guard > Exports > \(destination.lastPathComponent)\(noteText)")
    }

    private func validateFinishedImage(_ url: URL) throws {
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(imageSource) > 0,
              let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0 else {
            throw NSError(domain: "RIFE60GhostGuard", code: 34, userInfo: [NSLocalizedDescriptionKey: "The finished image could not be read back after saving."])
        }
        DiagnosticsLogger.shared.log("Finished image validation passed • \(width)x\(height)")
    }

    private func updateClock(progress: Double) {
        guard let renderStartedAt else { return }
        let elapsed = Date().timeIntervalSince(renderStartedAt)
        elapsedSeconds = elapsed
        // After a resume the first reported progress is the recovered stage, not zero, so the
        // ETA must be based on progress made during THIS run or it would read near zero.
        if resumeBaseProgress == nil { resumeBaseProgress = progress }
        let done = progress - (resumeBaseProgress ?? 0)
        if done > 0.025 && progress < 0.97 {
            let raw = elapsed * (1.0 - progress) / done
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
        if currentJobForcesGIF || videoToGIFEnabled {
            let redditOptimizer = RedditMediaOptimizer()
            statusText = "Converting fully processed video to GIF…"
            let result = try await redditOptimizer.optimizeVideoAsGIF(sourceURL: source) { [weak self] p, message in
                Task { @MainActor in
                    guard let self else { return }
                    self.progress = 0.97 + min(max(p, 0), 1) * 0.025
                    self.statusText = message
                }
            }
            try Task.checkCancellation()

            let saved = try await saveRedditAsset(
                result,
                filenamePrefix: "RIFE60-Processed-GIF"
            )
            if result.url != source { try? FileManager.default.removeItem(at: result.url) }
            if source != saved.url { try? FileManager.default.removeItem(at: source) }

            DiagnosticsLogger.shared.log("Processed video-to-GIF export • \(result.processing.rawValue) • source=\(result.originalBytes) bytes • output=\(result.bytes) bytes • \(result.summary)")
            return SavedResult(
                url: saved.url,
                message: "\(result.summary) • saved \(saved.bytes) bytes • \(saved.message) • audio omitted because GIF has no audio track"
            )
        }

        let optimizer = FinalSizeOptimizer()
        let optimized = try await optimizer.optimizeIfNeeded(sourceURL: source) { [weak self] p, message in
            Task { @MainActor in
                guard let self else { return }
                self.progress = 0.97 + min(max(p, 0), 1) * 0.025
                self.statusText = message
            }
        }
        try Task.checkCancellation()

        statusText = optimized.optimized ? "Final compression complete • saving to Files…" : "Master already under 1 GB • saving to Files…"
        progress = 0.995
        RecoveryStore.update(progress: 0.995, message: "Saving finished result to Files", force: true)

        let saved = try await saveToFiles(optimized.url)
        let bytes = saved.bytes
        guard bytes < FinalSizeOptimizer.hardLimitBytes else {
            throw NSError(domain: "RIFE60GhostGuard", code: 33, userInfo: [NSLocalizedDescriptionKey: "The final saved video is still 1 GB or larger."])
        }

        if optimized.optimized {
            DiagnosticsLogger.shared.log("Files export verified • final size optimized from \(optimized.originalBytes) to \(bytes) bytes • \(saved.url.path)")
        } else {
            DiagnosticsLogger.shared.log("Files export verified • size pass skipped • \(bytes) bytes • \(saved.url.path)")
        }

        if optimized.url != saved.url { try? FileManager.default.removeItem(at: optimized.url) }
        if source != saved.url, source != optimized.url { try? FileManager.default.removeItem(at: source) }

        let sizeText = String(format: "%.0f MB", Double(bytes) / 1_000_000.0)
        let message = optimized.optimized
            ? "Auto-saved final HEVC Main10 video • \(sizeText) • under 1 GB • \(saved.message)"
            : "Auto-saved without extra compression • \(sizeText) • \(saved.message)"
        return SavedResult(url: saved.url, message: message)
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

    private func saveRedditAsset(
        _ result: RedditMediaResult,
        filenamePrefix: String
    ) async throws -> (url: URL, message: String, bytes: Int64) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let ext = result.url.pathExtension.isEmpty
            ? (result.kind == .gif ? "gif" : "jpg")
            : result.url.pathExtension.lowercased()
        let filename = "\(filenamePrefix)-\(formatter.string(from: Date())).\(ext)"

        func persist(to folder: URL, messagePrefix: String) async throws -> (url: URL, message: String, bytes: Int64) {
            let destination = uniqueDestination(in: folder, filename: filename)
            let persisted = try coordinatedCopy(result.url, to: destination, inside: folder)
            try verifyPersistedFile(persisted)

            if result.kind == .gif {
                try validateFinishedGIF(persisted)
            } else {
                try validateFinishedImage(persisted)
            }

            let attrs = try FileManager.default.attributesOfItem(atPath: persisted.path)
            let bytes = (attrs[.size] as? NSNumber)?.int64Value ?? 0
            guard bytes > 0, bytes <= RedditMediaOptimizer.hardLimitBytes else {
                throw NSError(
                    domain: "RIFE60GhostGuard",
                    code: 35,
                    userInfo: [NSLocalizedDescriptionKey: "The Reddit-ready export exceeded 20 MB."]
                )
            }

            return (persisted, "\(messagePrefix) > \(persisted.lastPathComponent)", bytes)
        }

        if let folder = resolveSelectedExportFolder() {
            let secured = folder.startAccessingSecurityScopedResource()
            if secured {
                defer { folder.stopAccessingSecurityScopedResource() }
                do {
                    return try await persist(
                        to: folder,
                        messagePrefix: "Files > (folder.lastPathComponent)"
                    )
                } catch {
                    DiagnosticsLogger.shared.log("Selected Files folder Reddit export failed: (error.localizedDescription). Falling back to app Exports folder.")
                }
            } else {
                DiagnosticsLogger.shared.log("Selected Files folder security scope could not be activated for Reddit export. Falling back to app Exports folder.")
            }
        }

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)

        return try await persist(
            to: exports,
            messagePrefix: "On My iPhone > RIFE 60 Ghost Guard > Exports"
        )
    }

    private func validateFinishedGIF(_ url: URL) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(source) > 0,
              let firstFrame = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NSError(
                domain: "RIFE60GhostGuard",
                code: 36,
                userInfo: [NSLocalizedDescriptionKey: "The finished GIF could not be read back after saving."]
            )
        }

        let count = CGImageSourceGetCount(source)
        let width = firstFrame.width
        let height = firstFrame.height
        guard width > 0, height > 0 else {
            throw NSError(
                domain: "RIFE60GhostGuard",
                code: 37,
                userInfo: [NSLocalizedDescriptionKey: "The finished GIF has invalid frame dimensions."]
            )
        }
        DiagnosticsLogger.shared.log(
            "Finished GIF validation passed • \(count) frame\(count == 1 ? "" : "s") • \(width)x\(height) • long edge \(max(width, height))px"
        )
    }

    private func saveToFiles(_ source: URL) async throws -> (url: URL, message: String, bytes: Int64) {
        let formatter = DateFormatter(); formatter.dateFormat = "yyyyMMdd-HHmmss"
        let ext = source.pathExtension.isEmpty ? "mp4" : source.pathExtension.lowercased()
        let filename = "RIFE60-2X60-\(formatter.string(from: Date())).\(ext)"

        if let folder = resolveSelectedExportFolder() {
            let secured = folder.startAccessingSecurityScopedResource()
            if secured {
                defer { folder.stopAccessingSecurityScopedResource() }
                do {
                    let destination = uniqueDestination(in: folder, filename: filename)
                    let persisted = try coordinatedCopy(source, to: destination, inside: folder)
                    try verifyPersistedFile(persisted)
                    try await validateFinishedVideo(persisted)
                    let attrs = try FileManager.default.attributesOfItem(atPath: persisted.path)
                    let bytes = (attrs[.size] as? NSNumber)?.int64Value ?? 0
                    return (persisted, "Files > \(folder.lastPathComponent) > \(persisted.lastPathComponent)", bytes)
                } catch {
                    DiagnosticsLogger.shared.log("Selected Files folder write/validation failed while security scope was active: \(error.localizedDescription). Falling back to app Exports folder.")
                }
            } else {
                DiagnosticsLogger.shared.log("Selected Files folder security scope could not be activated. Falling back to app Exports folder.")
            }
        }

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
        let destination = uniqueDestination(in: exports, filename: filename)
        try FileManager.default.copyItem(at: source, to: destination)
        try verifyPersistedFile(destination)
        try await validateFinishedVideo(destination)
        let attrs = try FileManager.default.attributesOfItem(atPath: destination.path)
        let bytes = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        return (destination, "On My iPhone > RIFE 60 Ghost Guard > Exports > \(destination.lastPathComponent)", bytes)
    }

    private func coordinatedCopy(_ source: URL, to destination: URL, inside folder: URL) throws -> URL {
        let fm = FileManager.default
        let coordinator = NSFileCoordinator(filePresenter: nil)
        var coordinationError: NSError?
        var operationError: Error?
        var actualDestination = destination

        coordinator.coordinate(writingItemAt: folder, options: .forMerging, error: &coordinationError) { coordinatedFolder in
            let coordinatedDestination = coordinatedFolder.appendingPathComponent(destination.lastPathComponent)
            actualDestination = coordinatedDestination
            do {
                try? fm.removeItem(at: coordinatedDestination)
                try fm.copyItem(at: source, to: coordinatedDestination)
            } catch {
                operationError = error
            }
        }

        if let operationError { throw operationError }
        if let coordinationError { throw coordinationError }
        return actualDestination
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

    nonisolated private func persistDeliveryCheckpoint(from source: URL, in recoveryDirectory: URL) throws -> URL {
        let fm = FileManager.default
        for candidate in deliveryCheckpointCandidates(in: recoveryDirectory) { try? fm.removeItem(at: candidate) }
        let ext = source.pathExtension.isEmpty ? "mp4" : source.pathExtension.lowercased()
        let destination = recoveryDirectory.appendingPathComponent("checkpoint-delivery-ready.\(ext)")
        do {
            try fm.moveItem(at: source, to: destination)
            DiagnosticsLogger.shared.log("Delivery-ready checkpoint persisted by move • future export retries skip render and mux")
        } catch {
            DiagnosticsLogger.shared.log("Delivery checkpoint move unavailable • falling back to copy: \(error.localizedDescription)")
            try fm.copyItem(at: source, to: destination)
        }
        return destination
    }

    nonisolated private func existingDeliveryCheckpoint(in recoveryDirectory: URL) -> URL? {
        let fm = FileManager.default
        for candidate in deliveryCheckpointCandidates(in: recoveryDirectory) {
            if fm.fileExists(atPath: candidate.path),
               let attrs = try? fm.attributesOfItem(atPath: candidate.path),
               let size = attrs[.size] as? NSNumber,
               size.int64Value > 1_000_000 {
                return candidate
            }
        }
        return nil
    }

    nonisolated private func deliveryCheckpointCandidates(in recoveryDirectory: URL) -> [URL] {
        ["mp4", "mov", "m4v"].map { recoveryDirectory.appendingPathComponent("checkpoint-delivery-ready.\($0)") }
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
                DiagnosticsLogger.shared.log("Renewed stale Files export bookmark.")
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
    // MARK: - Persistent serial media queue

    private func restoreMediaQueue() {
        do {
            guard let queue = try batchStore.load() else { return }
            mediaQueue = queue
            if let settings = queue.settingsKey { applyRecoveredSettings(from: settings) }
            if let id = queue.nextItemID ?? queue.items.last?.id {
                activateQueueItem(id)
                if queue.nextItemID != nil {
                    batchStatusText = "Queue restored • \(queue.items.filter { $0.state != .completed }.count) remaining"
                } else { batchStatusText = "Batch complete" }
            }
        } catch {
            errorText = "Could not restore media queue: " + error.localizedDescription
            DiagnosticsLogger.shared.log(errorText ?? "Queue restore failed")
        }
    }

    private func replaceMediaQueue(with urls: [URL]) async throws {
        let store = batchStore
        let staged = try await Task.detached(priority: .userInitiated) { [weak self] in
            try store.stage(urls) { fraction in
                Task { @MainActor [weak self] in self?.importProgress = 0.5 + fraction * 0.5 }
            }
        }.value
        do { try store.save(staged) }
        catch { store.removeInputs(staged); throw error }
        if let old = mediaQueue { store.removeInputs(old) }
        RecoveryStore.discardCurrent()
        recoveryAvailable = false; recoveryStatusText = ""
        mediaQueue = staged
        errorText = nil; saveStatusText = ""; outputURL = nil
        if let first = staged.items.first { activateQueueItem(first.id) }
        statusText = "Import complete"
        batchStatusText = "\(staged.items.count) media item(s) ready • sequential processing"
        DiagnosticsLogger.shared.log("Media queue imported • count=\(staged.items.count) • ordered, one render at a time")
    }

    private func activateQueueItem(_ id: UUID) {
        guard let queue = mediaQueue, let item = queue.items.first(where: { $0.id == id }) else { return }
        loadingQueueSelection = true
        selectedMediaID = id
        inputKind = item.kind
        inputURL = batchStore.sourceURL(for: item, in: queue)
        watermarkRegions = item.masks
        loadingQueueSelection = false
        if recoveryAvailable, id == queue.nextItemID, let job = RecoveryStore.existingJob() {
            inputURL = job.sourceURL
            progress = job.manifest.progress
            if item.kind == .gif {
                inputKind = .video
                videoToGIFEnabled = true
            }
        } else { progress = item.state == .completed ? 1 : 0 }
        outputURL = item.outputURL
    }

    func selectQueueItem(_ id: UUID) {
        guard !queueLocked, !isImporting else { return }
        activateQueueItem(id)
    }

    private func saveSelectedMask() {
        guard !loadingQueueSelection, !queueLocked, let id = selectedMediaID,
              var queue = mediaQueue, let index = queue.items.firstIndex(where: { $0.id == id }) else { return }
        queue.items[index].masks = watermarkRegions
        do { try batchStore.save(queue); mediaQueue = queue }
        catch { errorText = "Could not save mask: " + error.localizedDescription }
    }

    func setSelectedWatermarkRemoval(_ enabled: Bool) {
        guard !queueLocked, var queue = mediaQueue, let index = queue.items.firstIndex(where: { $0.id == selectedMediaID }) else { return }
        queue.items[index].removeWatermark = enabled
        do { try batchStore.save(queue); mediaQueue = queue }
        catch { errorText = error.localizedDescription }
    }

    func copyMaskToOtherMedia() {
        guard !queueLocked, !watermarkRegions.isEmpty, var queue = mediaQueue else { return }
        for index in queue.items.indices where queue.items[index].id != selectedMediaID && queue.items[index].state != .completed {
            queue.items[index].masks = watermarkRegions
        }
        do { try batchStore.save(queue); mediaQueue = queue; batchStatusText = "Mask copied • review it on every other media item" }
        catch { errorText = error.localizedDescription }
    }

    func moveQueueItem(_ id: UUID, by offset: Int) {
        guard !queueLocked, var queue = mediaQueue, let index = queue.items.firstIndex(where: { $0.id == id }) else { return }
        let next = index + offset
        guard queue.items.indices.contains(next) else { return }
        queue.items.swapAt(index, next)
        do { try batchStore.save(queue); mediaQueue = queue }
        catch { errorText = error.localizedDescription }
    }

    func clearMediaQueue() { clearRecoveryData() }

    func start() async {
        guard !isBusy, !isBenchmarking, !isImporting else { return }
        guard var queue = mediaQueue else { _ = await startSingle(); return }
        guard queue.nextItemID != nil else { return }
        if watermarkRemovalEnabled {
            if let index = queue.items.firstIndex(where: { $0.state != .completed && $0.removeWatermark &&
                !WatermarkConfiguration(enabled: true, regions: $0.masks).isValid }) {
                activateQueueItem(queue.items[index].id)
                errorText = "Paint a watermark mask for item \(index + 1), or turn off removal for that item."
                return
            }
        }
        if queue.settingsKey == nil { queue.settingsKey = batchSettingsKey() }
        do { try batchStore.save(queue); mediaQueue = queue }
        catch { errorText = error.localizedDescription; return }
        batchCancelRequested = false
        isBatchProcessing = true
        defer { isBatchProcessing = false }
        let ids = queue.items.filter { $0.state != .completed }.map(\.id)
        do {
            try await BatchSerialRunner.run(ids: ids, shouldStop: { self.batchCancelRequested }) { id in
                guard var current = self.mediaQueue, let index = current.items.firstIndex(where: { $0.id == id }) else { throw CancellationError() }
                self.activateQueueItem(id)
                current.items[index].state = .processing
                try self.batchStore.save(current); self.mediaQueue = current
                self.batchStatusText = "Processing item \(index + 1) of \(current.items.count) • \(current.items[index].kind.rawValue)"
                DiagnosticsLogger.shared.log("Batch item begin • \(index + 1)/\(current.items.count) • \(current.items[index].displayName)")
                let render = await self.startSingle()
                if self.batchCancelRequested { render?.cancel() }
                if let render { await render.value }
                guard var updated = self.mediaQueue else { throw CancellationError() }
                if self.progress == 1, let output = self.outputURL, self.errorText == nil {
                    updated.items[index].state = .completed
                    updated.items[index].outputURL = output
                    try self.batchStore.save(updated); self.mediaQueue = updated
                    DiagnosticsLogger.shared.log("Batch item saved • \(index + 1)/\(updated.items.count) • \(output.lastPathComponent)")
                } else {
                    updated.items[index].state = self.batchCancelRequested ? .pending : .failed
                    try self.batchStore.save(updated); self.mediaQueue = updated
                    if self.batchCancelRequested { throw CancellationError() }
                    throw ProcessorError.conversionFailed(self.errorText ?? "The media item did not finish; the queue has been paused")
                }
            }
            batchStatusText = "Batch complete • \(queue.items.count) saved"
            statusText = "Finished"
        } catch is CancellationError {
            batchStatusText = "Queue paused • completed outputs saved; remaining items retained"
        } catch {
            batchStatusText = "Queue paused at this item • remaining items retained"
            errorText = error.localizedDescription
        }
    }

    private func batchSettingsKey() -> String {
        let guardEnabled = ghostProtection, cuts = sceneCutProtection
        let compressionEnabled = compressionProtection, outlineEnabled = outlineProtection
        let colorPop = colorPopEnabled, gradeStrength = colorPopStrength
        let sensitivity = ghostSensitivity, audio = preserveAudio, upscale = upscaleTo4K
        let upscaleBeforeRIFE = upscaleFirst && upscale, modelCompute = computePreference.rawValue
        let watermark = WatermarkConfiguration(enabled: watermarkRemovalEnabled, regions: watermarkRegions, paddingPixels: Int(watermarkPaddingPixels))
        return ["pipeline-v4-gif-video", "hq", "ghost=\(guardEnabled)", "cuts=\(cuts)", "compression=\(compressionEnabled)", "outline=\(outlineEnabled)", "colorpop=\(colorPop)", "colorpopStrength=\(gradeStrength)", "watermark=\(watermark.enabled)", "watermarkMasks=\(watermark.serializedRegions)", "watermarkPadding=\(watermark.paddingPixels)", "watermarkModel=\(WatermarkConfiguration.modelID)", String(format: "sensitivity=%.2f", sensitivity), "audio=\(audio)", "reddit=\(redditMode)", "videoToGIF=\(videoToGIFEnabled)", "upscale=\(upscale)", "upscaleFirst=\(upscaleBeforeRIFE)", "computeUnits=\(modelCompute)", "fps=60"].joined(separator: "|")
    }

}

