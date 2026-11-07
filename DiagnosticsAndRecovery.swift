import Foundation
import UIKit

final class DiagnosticsLogger: @unchecked Sendable {
    static let shared = DiagnosticsLogger()
    private let queue = DispatchQueue(label: "RIFE60.DiagnosticsLogger")
    private let fm = FileManager.default
    private let maxBytes = 1_500_000
    private let formatter = ISO8601DateFormatter()
    private var appendCount = 0

    private var directory: URL {
        let documents = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("Diagnostics", isDirectory: true)
    }

    private var logURL: URL { directory.appendingPathComponent("latest.log") }

    private init() {
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        queue.async { self.pruneLocked() }
    }

    func log(_ message: String) {
        queue.async { self.appendLocked(message) }
    }

    /// Writes before returning so a crash on the next line is still in the log.
    func logSync(_ message: String) {
        queue.sync { self.appendLocked(message) }
    }

    func clear() {
        queue.sync {
            try? fm.removeItem(at: logURL)
            DiagnosticsExport.pruneStaleRuns(removingAllRuns: true)
        }
        CrashCapture.clearCaptureFile()
    }

    private func appendLocked(_ message: String) {
        let stamp = formatter.string(from: Date())
        let line = "[\(stamp)] \(message)\n"
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        if !fm.fileExists(atPath: logURL.path) {
            try? Data(line.utf8).write(to: logURL, options: .atomic)
        } else if let handle = try? FileHandle(forWritingTo: logURL) {
            defer { try? handle.close() }
            try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(line.utf8))
        }
        appendCount += 1
        // Do not reread and rescan the entire diagnostic log for every frame.
        // High-frequency frame telemetry can otherwise turn diagnostics into a
        // CPU/disk workload that competes with the ML pipeline.
        if appendCount % 64 == 0 || logFileSizeExceedsLimit() {
            pruneLocked()
        }
    }

    private func logFileSizeExceedsLimit() -> Bool {
        guard let attrs = try? fm.attributesOfItem(atPath: logURL.path),
              let size = attrs[.size] as? NSNumber else { return false }
        return size.intValue > maxBytes
    }

    /// Drops log lines and diagnostic run files older than two days.
    private func pruneLocked() {
        let cutoff = Date().addingTimeInterval(-2 * 24 * 60 * 60)
        if let text = try? String(contentsOf: logURL, encoding: .utf8) {
            var keep = false
            var kept = ""
            var sawTimestamp = false
            for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                if let date = timestamp(in: line) {
                    sawTimestamp = true
                    keep = date >= cutoff
                } else if !sawTimestamp {
                    keep = true
                }
                if keep {
                    kept += line
                    kept += "\n"
                }
            }
            if kept != text {
                try? Data(kept.utf8).write(to: logURL, options: .atomic)
            }
        }
        guard let attrs = try? fm.attributesOfItem(atPath: logURL.path),
              let size = attrs[.size] as? NSNumber,
              size.intValue > maxBytes,
              let data = try? Data(contentsOf: logURL) else {
            DiagnosticsExport.pruneStaleRuns(removingAllRuns: false)
            return
        }
        try? Data(data.suffix(maxBytes / 2)).write(to: logURL, options: .atomic)
        DiagnosticsExport.pruneStaleRuns(removingAllRuns: false)
    }

    private func timestamp(in line: Substring) -> Date? {
        guard line.hasPrefix("["), let end = line.firstIndex(of: "]"), line.startIndex < end else { return nil }
        let stamp = line[line.index(after: line.startIndex)..<end]
        return formatter.date(from: String(stamp))
    }

    func text() -> String {
        queue.sync {
            pruneLocked()
            let header = """
            RIFE 60 Ghost Guard diagnostics
            Device: \(UIDevice.current.model)
            iOS: \(UIDevice.current.systemVersion)
            App: \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?") (\(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"))
            Thermal: \(currentThermalStateName())
            Date: \(formatter.string(from: Date()))

            """
            let body = (try? String(contentsOf: logURL, encoding: .utf8)) ?? "No diagnostic events have been recorded yet.\n"
            return header + body
        }
    }
}
