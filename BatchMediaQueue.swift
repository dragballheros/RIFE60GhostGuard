import Foundation
import UniformTypeIdentifiers

// Kept independent of UIKit so persistence and serial scheduling run in CI.
enum InputMediaKind: String, Codable, Sendable {
    case video
    case image
    case gif

    static func detect(for url: URL) -> InputMediaKind {
        let ext = url.pathExtension.lowercased()
        if ext == "gif" { return .gif }
        if let type = UTType(filenameExtension: ext), type.conforms(to: .image) { return .image }
        return .video
    }
}
enum BatchMediaState: String, Codable, Sendable {
    case pending, processing, completed, failed
}
struct BatchMediaItem: Codable, Identifiable, Sendable {
    var id = UUID()
    var filename: String
    var displayName: String
    var kind: InputMediaKind
    var masks: [WatermarkRegion] = []
    var removeWatermark = true
    var state: BatchMediaState = .pending
    var outputURL: URL?
}
struct BatchMediaManifest: Codable, Sendable {
    var version = 1
    var directoryName: String
    var items: [BatchMediaItem]
    var settingsKey: String?
    var nextItemID: UUID? { items.first { $0.state != .completed }?.id }
}

/// Owns imported copies; never deletes a user's Files original or saved output.
struct BatchMediaQueueStore: Sendable {
    let root: URL
    init(root: URL? = nil) {
        self.root = root ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RIFE60BatchMedia", isDirectory: true)
    }
    func sourceURL(for item: BatchMediaItem, in manifest: BatchMediaManifest) -> URL {
        root.appendingPathComponent(manifest.directoryName, isDirectory: true).appendingPathComponent(item.filename)
    }
    func load() throws -> BatchMediaManifest? {
        let url = root.appendingPathComponent("queue.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let manifest = try JSONDecoder().decode(BatchMediaManifest.self, from: Data(contentsOf: url))
        guard manifest.version == 1, !manifest.items.isEmpty, manifest.items.count <= 100,
              UUID(uuidString: manifest.directoryName) != nil,
              Set(manifest.items.map(\.id)).count == manifest.items.count,
              manifest.items.allSatisfy({ !$0.filename.contains("/") && !$0.filename.contains("..") && !$0.filename.isEmpty &&
                  WatermarkConfiguration(enabled: !$0.masks.isEmpty, regions: $0.masks).isValid }) else {
            throw NSError(domain: "RIFE60Batch", code: 1, userInfo: [NSLocalizedDescriptionKey: "The saved media queue is invalid."])
        }
        return manifest
    }
    func save(_ manifest: BatchMediaManifest) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(manifest).write(to: root.appendingPathComponent("queue.json"), options: .atomic)
    }
    func stage(_ urls: [URL], progress: @Sendable (Double) -> Void) throws -> BatchMediaManifest {
        guard !urls.isEmpty, urls.count <= 100 else {
            throw NSError(domain: "RIFE60Batch", code: 2, userInfo: [NSLocalizedDescriptionKey: "Select between 1 and 100 media items."])
        }
        let name = UUID().uuidString
        let directory = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            var items: [BatchMediaItem] = []
            for (index, source) in urls.enumerated() {
                try Task.checkCancellation()
                let secured = source.startAccessingSecurityScopedResource()
                defer { if secured { source.stopAccessingSecurityScopedResource() } }
                let filename = UUID().uuidString + "." + (source.pathExtension.isEmpty ? "mov" : source.pathExtension)
                let destination = directory.appendingPathComponent(filename)
                var failure: Error?
                NSFileCoordinator().coordinate(readingItemAt: source, options: [], error: nil) { coordinated in
                    do { try FileManager.default.copyItem(at: coordinated, to: destination) }
                    catch { failure = error }
                }
                if let failure { throw failure }
                guard FileManager.default.fileExists(atPath: destination.path) else {
                    throw NSError(domain: "RIFE60Batch", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not import \(source.lastPathComponent)."])
                }
                items.append(BatchMediaItem(filename: filename, displayName: source.lastPathComponent, kind: .detect(for: source)))
                progress(Double(index + 1) / Double(urls.count))
            }
            return BatchMediaManifest(directoryName: name, items: items)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }
    func removeInputs(_ manifest: BatchMediaManifest) {
        try? FileManager.default.removeItem(at: root.appendingPathComponent(manifest.directoryName, isDirectory: true))
    }
    func discard() throws {
        if let manifest = try load() { removeInputs(manifest) }
        let file = root.appendingPathComponent("queue.json")
        if FileManager.default.fileExists(atPath: file.path) { try FileManager.default.removeItem(at: file) }
    }
}

@MainActor
enum BatchSerialRunner {
    /// Await the entire render AND export before starting the next media item.
    static func run(ids: [UUID], shouldStop: () -> Bool,
                    process: (UUID) async throws -> Void) async throws {
        for id in ids {
            try Task.checkCancellation()
            if shouldStop() { throw CancellationError() }
            try await process(id)
        }
    }
}
