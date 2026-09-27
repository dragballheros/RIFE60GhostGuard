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
        guard let source = inputURL else { return }
        isProcessing = true
        progress = 0
        outputURL = nil
        errorText = nil
        statusText = "Preparing RIFE 4.26 HQ…"

        let guardEnabled = ghostProtection
        let cuts = sceneCutProtection
        let compressionEnabled = compressionProtection
        let sensitivity = ghostSensitivity
        let audio = preserveAudio

        currentTask = Task {
            do {
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }

                let config = ProcessorConfiguration(
                    quality: .hq,
                    ghostProtection: guardEnabled,
                    sceneCutProtection: cuts,
                    compressionProtection: compressionEnabled,
                    ghostSensitivity: sensitivity,
                    preserveAudio: audio,
                    targetFPS: 60
                )
                let processor = RIFEVideoProcessor(configuration: config)
                let result = try await processor.process(sourceURL: source) { [weak self] p, message in
                    Task { @MainActor in
                        self?.progress = p
                        self?.statusText = message
                    }
                }
                if Task.isCancelled { throw CancellationError() }
                outputURL = result
                progress = 1
                statusText = "Finished"
            } catch is CancellationError {
                statusText = "Cancelled"
            } catch {
                errorText = error.localizedDescription
                statusText = "Failed"
            }
            isProcessing = false
            currentTask = nil
        }
    }
}
