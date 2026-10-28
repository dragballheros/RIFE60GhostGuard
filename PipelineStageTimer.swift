import Foundation

/// Records per-stage durations for one video run and writes a JSON report the owner can export.
final class PipelineStageTimer: @unchecked Sendable {
    private let lock = NSLock()
    private let started = Date()
    private var marks: [(name: String, seconds: Double)] = []
    private var notes: [String] = []
    private var metrics: [String: Double] = [:]
    private var rejections: [[String: Any]] = []
    private var open: [String: Date] = [:]
    private static let currentLock = NSLock()
    static weak var current: PipelineStageTimer?

    func begin(_ name: String) {
        lock.lock()
        open[name] = Date()
        lock.unlock()
    }

    func end(_ name: String) {
        lock.lock()
        let start = open.removeValue(forKey: name) ?? started
        marks.append((name, Date().timeIntervalSince(start)))
        lock.unlock()
    }

    func record(_ name: String, milliseconds: Double) {
        lock.lock()
        metrics[name] = milliseconds
        lock.unlock()
    }

    func note(_ text: String) {
        lock.lock()
        notes.append(text)
        lock.unlock()
    }

    func recordRejection(seconds: Double, reason: String, disposition: String) {
        lock.lock()
        rejections.append(["seconds": seconds, "reason": reason, "disposition": disposition])
        lock.unlock()
    }

    static func recordRejection(seconds: Double, reason: String, disposition: String) {
        currentLock.lock()
        let timer = current
        currentLock.unlock()
        timer?.recordRejection(seconds: seconds, reason: reason, disposition: disposition)
    }

    static func note(_ text: String) {
        currentLock.lock()
        let timer = current
        currentLock.unlock()
        timer?.note(text)
        if timer == nil {
            let payload: [String: Any] = ["notes": [text]]
            guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]),
                  let docs = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true) else { return }
            let directory = docs.appendingPathComponent("Diagnostics", isDirectory: true)
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            try? data.write(to: directory.appendingPathComponent("run-fallback-\(stamp).json"), options: .atomic)
        }
    }

    func finish() {
        lock.lock()
        let snapshot = marks
        let recordedNotes = notes
        let recordedMetrics = metrics
        let recordedRejections = rejections
        let total = Date().timeIntervalSince(started)
        lock.unlock()
        let stages = snapshot.map { ["stage": $0.name, "seconds": $0.seconds] as [String: Any] }
        let payload: [String: Any] = ["totalSeconds": total, "stages": stages, "notes": recordedNotes, "metrics": recordedMetrics, "rifeRejections": recordedRejections]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) else { return }
        let fm = FileManager.default
        guard let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true) else { return }
        let directory = docs.appendingPathComponent("Diagnostics", isDirectory: true)
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = directory.appendingPathComponent("run-\(stamp).json")
        try? data.write(to: url, options: .atomic)
        DiagnosticsLogger.shared.log("Stage timing report \(url.lastPathComponent) • total \(Int(total))s")
    }
}
