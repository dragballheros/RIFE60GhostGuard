import Foundation
import SwiftUI
import RifeMetal

@MainActor
final class VideoProcessorViewModel: ObservableObject {
    @Published var inputURL: URL?
    @Published var outputURL: URL?
    @Published var progress: Double = 0
    @Published var statusText = ""
    @Published var errorText: String?
    @Published var isProcessing = false
    @Published var quality: RIFEQualityChoice = .balanced
    @Published var ghostProtection = true
    @Published var sceneCutProtection = true
    @Published var ghostSensitivity = 1.0
    @Published var codec: OutputCodec = .h264
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

    func cancel() { currentTask?.cancel() }

    func start() async {
        guard let source = inputURL else { return }
        isProcessing = true
        progress = 0
        outputURL = nil
        errorText = nil
        statusText = "Preparing RIFE 4.26…"

        let q = quality
        let guardEnabled = ghostProtection
        let cuts = sceneCutProtection
        let sensitivity = ghostSensitivity
        let selectedCodec = codec
        let audio = preserveAudio

        currentTask = Task {
            do {
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }

                let config = ProcessorConfiguration(
                    quality: q.rifeTier,
                    ghostProtection: guardEnabled,
                    sceneCutProtection: cuts,
                    ghostSensitivity: sensitivity,
                    codec: selectedCodec,
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

enum RIFEQualityChoice: String, CaseIterable, Identifiable {
    case hq, balanced, fast
    var id: String { rawValue }
    var title: String {
        switch self { case .hq: return "HQ"; case .balanced: return "Balanced"; case .fast: return "Fast" }
    }
    var rifeTier: RifeQualityTier {
        switch self { case .hq: return .hq; case .balanced: return .balanced; case .fast: return .fast }
    }
}

enum OutputCodec: String, CaseIterable, Identifiable {
    case h264, hevc
    var id: String { rawValue }
    var title: String { self == .h264 ? "H.264" : "HEVC" }
}
