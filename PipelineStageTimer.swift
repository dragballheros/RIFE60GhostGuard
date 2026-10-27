import Foundation

/// Records per-stage durations for one video run and writes a JSON report the owner can export.
final class PipelineStageTimer: @unchecked Sendable {
    private let lock = NSLock()
    private let started = Date()
    private var marks: [(name: String, seconds: Double)] = []
    private var open: [String: Date] = [:]

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

    func finish() {
        lock.lock()
        let snapshot = marks
        let total = Date().timeIntervalSince(started)
        lock.unlock()
        let stages = snapshot.map { ["stage": $0.name, "seconds": $0.seconds] as [String: Any] }
        let payload: [String: Any] = ["totalSeconds": total, "stages": stages]
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
