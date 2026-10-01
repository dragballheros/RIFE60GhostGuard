import Foundation
import UIKit

struct RecoveryManifest: Codable, Sendable {
    var version: Int = 1
    var sourceFilename: String
    var configurationKey: String
    var startedAt: Date
    var updatedAt: Date
    var progress: Double
    var lastMessage: String
    var status: String
}

struct RecoveryJob: Sendable {
    let directory: URL
    let sourceURL: URL
    let manifest: RecoveryManifest
    /// True when prepare() matched an existing job and kept its checkpoints; false when it started fresh.
    var resumed: Bool = false
}

final class DiagnosticsLogger: @unchecked Sendable {
    static let shared = DiagnosticsLogger()
    private let lock = NSLock()
    private let fm = FileManager.default
    private let maxBytes = 1_500_000

    private var directory: URL {
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("Diagnostics", isDirectory: true)
    }

    private var logURL: URL { directory.appendingPathComponent("latest.log") }

    private init() {
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func log(_ message: String) {
        lock.lock()
        defer { lock.unlock() }
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date())
        let line = "[\(stamp)] \(message)\n"
        if !fm.fileExists(atPath: logURL.path) {
            try? Data(line.utf8).write(to: logURL, options: .atomic)
        } else if let handle = try? FileHandle(forWritingTo: logURL) {
            defer { try? handle.close() }
            try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(line.utf8))
        }
        trimIfNeeded()
    }

    func text() -> String {
        lock.lock()
        defer { lock.unlock() }
        let header = """
        RIFE 60 Ghost Guard diagnostics
        Device: \(UIDevice.current.model)
        iOS: \(UIDevice.current.systemVersion)
        App: \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?") (\(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"))
        Thermal: \(currentThermalStateName())
        Date: \(ISO8601DateFormatter().string(from: Date()))

        """
        let body = (try? String(contentsOf: logURL, encoding: .utf8)) ?? "No diagnostic events have been recorded yet.\n"
        return header + body
    }

    private func trimIfNeeded() {
        guard let attrs = try? fm.attributesOfItem(atPath: logURL.path),
              let size = attrs[.size] as? NSNumber,
              size.intValue > maxBytes,
              let data = try? Data(contentsOf: logURL) else { return }
        let keep = data.suffix(maxBytes / 2)
        try? Data(keep).write(to: logURL, options: .atomic)
    }
}

enum RecoveryStore {
    private static let fm = FileManager.default
    private static let lock = NSLock()
    private static var lastPersist = Date.distantPast

    static var currentDirectory: URL {
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("Recovery", isDirectory: true).appendingPathComponent("Current", isDirectory: true)
    }

    private static var manifestURL: URL { currentDirectory.appendingPathComponent("manifest.json") }

    static func existingJob() -> RecoveryJob? {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(RecoveryManifest.self, from: data) else { return nil }
        let source = currentDirectory.appendingPathComponent(manifest.sourceFilename)
        guard fm.fileExists(atPath: source.path), manifest.status != "completed" else { return nil }
        return RecoveryJob(directory: currentDirectory, sourceURL: source, manifest: manifest)
    }

    // Canonicalize settings rather than invalidating build 162 checkpoints merely
    // because its manifest omitted the new, disabled-by-default grade fields.
    private static func comparableConfiguration(_ key: String) -> [String] {
        var parts = key.split(separator: "|").map(String.init)
        if !parts.contains(where: { $0.hasPrefix("colorpop=") }) { parts.append("colorpop=false") }
        if !parts.contains(where: { $0.hasPrefix("colorpopStrength=") }) { parts.append("colorpopStrength=0.50") }
        if !parts.contains(where: { $0.hasPrefix("watermark=") }) { parts.append("watermark=false") }
        if !parts.contains(where: { $0.hasPrefix("watermarkMasks=") }) { parts.append("watermarkMasks=W10=") }
        if !parts.contains(where: { $0.hasPrefix("watermarkPadding=") }) { parts.append("watermarkPadding=3") }
        if !parts.contains(where: { $0.hasPrefix("watermarkModel=") }) { parts.append("watermarkModel=\(WatermarkConfiguration.modelID)") }
        return parts.sorted()
    }

    static func prepare(source: URL, configurationKey: String) throws -> RecoveryJob {
        lock.lock()
        defer { lock.unlock() }

        if let data = try? Data(contentsOf: manifestURL),
           let existing = try? JSONDecoder().decode(RecoveryManifest.self, from: data),
           comparableConfiguration(existing.configurationKey) == comparableConfiguration(configurationKey) {
            let existingSource = currentDirectory.appendingPathComponent(existing.sourceFilename)
            if fm.fileExists(atPath: existingSource.path) && source.standardizedFileURL == existingSource.standardizedFileURL {
                var resumed = existing
                resumed.status = "running"
                resumed.configurationKey = configurationKey
                resumed.updatedAt = Date()
                try writeManifestLocked(resumed)
                DiagnosticsLogger.shared.log("Resuming recovery job at \(Int(resumed.progress * 100))%: \(resumed.lastMessage)")
                return RecoveryJob(directory: currentDirectory, sourceURL: existingSource, manifest: resumed, resumed: true)
            }
        }

        if let data = try? Data(contentsOf: manifestURL),
           let stale = try? JSONDecoder().decode(RecoveryManifest.self, from: data),
           comparableConfiguration(stale.configurationKey) != comparableConfiguration(configurationKey) {
            DiagnosticsLogger.shared.log("Recovery configuration mismatch • saved=\(stale.configurationKey) • requested=\(configurationKey) • discarding saved checkpoints.")
        }

        var copySource = source
        var stagedSource: URL?
        let recoveryPrefix = currentDirectory.standardizedFileURL.path + "/"
        if source.standardizedFileURL.path.hasPrefix(recoveryPrefix) {
            let ext = source.pathExtension.isEmpty ? "mov" : source.pathExtension
            let staged = fm.temporaryDirectory.appendingPathComponent("recovery-source-\(UUID().uuidString).\(ext)")
            try? fm.removeItem(at: staged)
            try fm.copyItem(at: source, to: staged)
            stagedSource = staged
            copySource = staged
        }
        defer { if let stagedSource { try? fm.removeItem(at: stagedSource) } }

        try? fm.removeItem(at: currentDirectory)
        try fm.createDirectory(at: currentDirectory, withIntermediateDirectories: true)
        let ext = copySource.pathExtension.isEmpty ? "mov" : copySource.pathExtension
        let localSource = currentDirectory.appendingPathComponent("source.\(ext)")
        try fm.copyItem(at: copySource, to: localSource)
        let manifest = RecoveryManifest(
            sourceFilename: localSource.lastPathComponent,
            configurationKey: configurationKey,
            startedAt: Date(),
            updatedAt: Date(),
            progress: 0,
            lastMessage: "Crash-safe source copy created",
            status: "running"
        )
        try writeManifestLocked(manifest)
        lastPersist = Date.distantPast
        DiagnosticsLogger.shared.log("Started new recovery job: \(localSource.lastPathComponent)")
        return RecoveryJob(directory: currentDirectory, sourceURL: localSource, manifest: manifest)
    }

    static func update(progress: Double, message: String, force: Bool = false) {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        if !force && now.timeIntervalSince(lastPersist) < 2.0 { return }
        guard let data = try? Data(contentsOf: manifestURL),
              var manifest = try? JSONDecoder().decode(RecoveryManifest.self, from: data) else { return }
        manifest.progress = min(max(progress, 0), 1)
        manifest.lastMessage = message
        manifest.updatedAt = now
        manifest.status = "running"
        try? writeManifestLocked(manifest)
        lastPersist = now
        DiagnosticsLogger.shared.log(String(format: "Progress %.1f%% • %@", manifest.progress * 100, message))
    }

    static func markFailed(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: manifestURL),
              var manifest = try? JSONDecoder().decode(RecoveryManifest.self, from: data) else { return }
        manifest.status = "interrupted"
        manifest.lastMessage = "Error: \(error.localizedDescription)"
        manifest.updatedAt = Date()
        try? writeManifestLocked(manifest)
        DiagnosticsLogger.shared.log("Render failed: \(error.localizedDescription)")
    }

    static func markCancelled() {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: manifestURL),
              var manifest = try? JSONDecoder().decode(RecoveryManifest.self, from: data) else { return }
        manifest.status = "cancelled"
        manifest.updatedAt = Date()
        try? writeManifestLocked(manifest)
        DiagnosticsLogger.shared.log("Render cancelled; checkpoints retained.")
    }

    static func finishAndClean() {
        lock.lock()
        defer { lock.unlock() }
        DiagnosticsLogger.shared.log("Render completed successfully; clearing recovery checkpoints.")
        try? fm.removeItem(at: currentDirectory)
    }

    static func discardCurrent() {
        lock.lock()
        defer { lock.unlock() }
        try? fm.removeItem(at: currentDirectory)
        lastPersist = Date.distantPast
    }

    private static func writeManifestLocked(_ manifest: RecoveryManifest) throws {
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL, options: .atomic)
    }
}
