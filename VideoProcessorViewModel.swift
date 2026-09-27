import Foundation
import SwiftUI
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
    @Published var statusText = ""
    @Published var errorText: String?
    @Published var isProcessing = false
    @Published var ghostProtection = true
    @Published var sceneCutProtection = true
    @Published var compressionProtection = true
    @Published var ghostSensitivity = 1.0
    @Published var preserveAudio = true

    private var currentTask: Task<Void, Never>?

    func handleImport(_ result: Result<[URL], Error>) {
        do {
            guard let url = try result.get().first else { return }
            inputURL = url
            outputURL = nil
            errorText = nil
        } catch {
            errorText = error.localizedDescription
        }
    }

    func handlePhotoSelection(_ item: PhotosPickerItem) async {
        do {
            statusText = "Importing from Photos…"
            guard let picked = try await item.loadTransferable(type: PickedVideo.self) else {
                throw NSError(domain: "RIFE60GhostGuard", code: 1, userInfo: [NSLocalizedDescriptionKey: "The selected Photos video could not be loaded."])
            }
            inputURL = picked.url
            outputURL = nil
            errorText = nil
            statusText = ""
        } catch {
            errorText = error.localizedDescription
            statusText = ""
        }
    }

    func cancel() { currentTask?.cancel() }

    func start() async {
        guard let source = inputURL, !isProcessing else { return }

        // Update the UI immediately before any expensive RIFE/Metal initialization.
        isProcessing = true
        progress = 0
        outputURL = nil
        errorText = nil
        statusText = "Preparing RIFE 4.26…"

        let guardEnabled = ghostProtection
        let cuts = sceneCutProtection
        let compressionEnabled = compressionProtection
        let sensitivity = ghostSensitivity
        let audio = preserveAudio

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }

                let config = ProcessorConfiguration(
                    quality: .balanced,
                    ghostProtection: guardEnabled,
                    sceneCutProtection: cuts,
                    compressionProtection: compressionEnabled,
                    ghostSensitivity: sensitivity,
                    preserveAudio: audio,
                    targetFPS: 60
                )

                let processor = RIFEVideoProcessor(configuration: config)
                let result = try await processor.process(sourceURL: source) { p, message in
                    Task { @MainActor [weak self] in
                        self?.progress = p
                        self?.statusText = message
                    }
                }

                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    self?.outputURL = result
                    self?.progress = 1
                    self?.statusText = "Finished"
                    self?.isProcessing = false
                    self?.currentTask = nil
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    self?.statusText = "Cancelled"
                    self?.isProcessing = false
                    self?.currentTask = nil
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.errorText = error.localizedDescription
                    self?.statusText = "Failed"
                    self?.isProcessing = false
                    self?.currentTask = nil
                }
            }
        }
    }
}
