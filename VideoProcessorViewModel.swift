import Foundation
import SwiftUI
import UIKit
import PhotosUI
import CoreTransferable
import UniformTypeIdentifiers
import RifeMetal

struct PickedVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory
                .appendingPathComponent("photos-\(UUID().uuidString).\(ext)")
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
    @Published var statusText = ""
    @Published var errorText: String?
    @Published var isProcessing = false
    @Published var isImporting = false
    @Published var importProgress: Double? = nil
    @Published var ghostProtection = true
    @Published var sceneCutProtection = true
    @Published var compressionProtection = true
    @Published var outlineProtection = true
    @Published var ghostSensitivity = 1.0
    @Published var preserveAudio = true
    @Published var renderPowerMode = true
    @Published var telemetry = PerformanceTelemetry()

    private var currentTask: Task<Void, Never>?
    private var savedBrightness: CGFloat?
    private var savedIdleTimerDisabled: Bool?

    func handleImport(_ result: Result<[URL], Error>) {
        isImporting = true
        importProgress = 0
        defer {
            isImporting = false
            importProgress = nil
        }
        do {
            guard let url = try result.get().first else { return }
            importProgress = 0.5
            inputURL = url
            outputURL = nil
            errorText = nil
            importProgress = 1.0
        } catch {
            errorText = error.localizedDescription
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
            inputURL = picked.url
            outputURL = nil
            errorText = nil
            statusText = "Import complete"
        } catch {
            errorText = error.localizedDescription
            statusText = ""
        }
        isImporting = false
        importProgress = nil
    }

    func cancel() { currentTask?.cancel() }

    func handleScenePhase(_ phase: ScenePhase) {
        switch phase {
        case .active:
            if isProcessing && renderPowerMode { applyRenderPowerMode() }
        case .inactive, .background:
            restoreDisplayState()
        @unknown default:
            break
        }
    }

    func start() async {
        guard let source = inputURL, !isProcessing else { return }

        isProcessing = true
        progress = 0
        restorationProgress = 0
        telemetry = PerformanceTelemetry()
        outputURL = nil
        errorText = nil
        statusText = "Preparing RIFE 4.26 HQ…"

        if renderPowerMode { applyRenderPowerMode() }

        let guardEnabled = ghostProtection
        let cuts = sceneCutProtection
        let compressionEnabled = compressionProtection
        let outlineEnabled = outlineProtection
        let sensitivity = ghostSensitivity
        let audio = preserveAudio

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }

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

                let processor = RIFEVideoProcessor(configuration: config)
                let result = try await processor.process(
                    sourceURL: source,
                    progress: { p, message in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.progress = p
                            self.restorationProgress = min(max(p / 0.92, 0), 1)
                            self.statusText = message
                        }
                    },
                    telemetry: { sample in
                        Task { @MainActor [weak self] in
                            self?.telemetry = sample
                        }
                    }
                )

                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.outputURL = result
                    self.progress = 1
                    self.restorationProgress = 1
                    self.statusText = "Finished"
                    self.isProcessing = false
                    self.currentTask = nil
                    self.restoreDisplayState()
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.statusText = "Cancelled"
                    self.isProcessing = false
                    self.currentTask = nil
                    self.restoreDisplayState()
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.errorText = error.localizedDescription
                    self.statusText = "Failed"
                    self.isProcessing = false
                    self.currentTask = nil
                    self.restoreDisplayState()
                }
            }
        }
    }

    private func applyRenderPowerMode() {
        if savedBrightness == nil { savedBrightness = UIScreen.main.brightness }
        if savedIdleTimerDisabled == nil { savedIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled }
        UIScreen.main.brightness = 0.05
        UIApplication.shared.isIdleTimerDisabled = true
    }

    private func restoreDisplayState() {
        if let savedBrightness {
            UIScreen.main.brightness = savedBrightness
            self.savedBrightness = nil
        }
        if let savedIdleTimerDisabled {
            UIApplication.shared.isIdleTimerDisabled = savedIdleTimerDisabled
            self.savedIdleTimerDisabled = nil
        }
    }
}
