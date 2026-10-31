import Foundation
import SwiftUI
import UIKit

enum DiagnosticsExport {
    static func ensureFolder() {
        guard let directory = directoryURL() else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        pruneStaleRuns(removingAllRuns: false)
    }

    /// `removingAllRuns` clears every run file. Otherwise only files older than two days are removed.
    static func pruneStaleRuns(removingAllRuns: Bool) {
        guard let directory = directoryURL(),
              let urls = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles]) else {
            return
        }
        let cutoff = Date().addingTimeInterval(-2 * 24 * 60 * 60)
        for url in urls {
            let name = url.lastPathComponent
            let isRun = name.hasPrefix("run-") && url.pathExtension == "json"
            let isZip = name.hasPrefix("diagnostics-last-") && url.pathExtension == "zip"
            guard isRun || isZip else { continue }
            if removingAllRuns || modificationDate(url) < cutoff {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }

    static func directoryURL() -> URL? {
        guard let docs = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true) else { return nil }
        return docs.appendingPathComponent("Diagnostics", isDirectory: true)
    }

    static func runFiles() -> [URL] {
        guard let directory = directoryURL(),
              let urls = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles]) else {
            return []
        }
        return urls
            .filter { $0.lastPathComponent.hasPrefix("run-") && $0.pathExtension == "json" }
            .sorted { modificationDate($0) > modificationDate($1) }
    }

    static func newestJSONText() -> String? {
        guard let newest = runFiles().first else { return nil }
        return try? String(contentsOf: newest, encoding: .utf8)
    }

    /// Newest run JSON, plus a zip of the last five runs.
    static func shareItems() -> [URL] {
        ensureFolder()
        let runs = Array(runFiles().prefix(5))
        guard let newest = runs.first else { return [] }
        var items = [newest]
        if let zip = zip(runs) { items.append(zip) }
        return items
    }

    private static func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    private static func zip(_ files: [URL]) -> URL? {
        guard let directory = directoryURL() else { return nil }
        let destination = directory.appendingPathComponent("diagnostics-last-\(files.count).zip")
        try? FileManager.default.removeItem(at: destination)
        var archive = Data()
        var catalog = Data()
        var offset = 0
        for file in files {
            guard let payload = try? Data(contentsOf: file) else { continue }
            let name = Data(file.lastPathComponent.utf8)
            var local = Data()
            local.append(contentsOf: [0x50, 0x4b, 0x03, 0x04])
            local.append(contentsOf: [20, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            local.append(contentsOf: le32(0))
            local.append(contentsOf: le32(payload.count))
            local.append(contentsOf: le32(payload.count))
            local.append(contentsOf: le16(name.count))
            local.append(contentsOf: le16(0))
            local.append(name)
            local.append(payload)
            archive.append(local)
            var central = Data()
            central.append(contentsOf: [0x50, 0x4b, 0x01, 0x02])
            central.append(contentsOf: [20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            central.append(contentsOf: le32(0))
            central.append(contentsOf: le32(payload.count))
            central.append(contentsOf: le32(payload.count))
            central.append(contentsOf: le16(name.count))
            central.append(contentsOf: le16(0))
            central.append(contentsOf: le16(0))
            central.append(contentsOf: le16(0))
            central.append(contentsOf: le16(0))
            central.append(contentsOf: le32(0))
            central.append(contentsOf: le32(offset))
            central.append(name)
            catalog.append(central)
            offset += local.count
        }
        let catalogStart = archive.count
        archive.append(catalog)
        var end = Data([0x50, 0x4b, 0x05, 0x06])
        end.append(contentsOf: [0, 0, 0, 0])
        end.append(contentsOf: le16(files.count))
        end.append(contentsOf: le16(files.count))
        end.append(contentsOf: le32(catalog.count))
        end.append(contentsOf: le32(catalogStart))
        end.append(contentsOf: le16(0))
        archive.append(end)
        do {
            try archive.write(to: destination, options: .atomic)
            return destination
        } catch {
            return nil
        }
    }

    private static func le16(_ value: Int) -> [UInt8] {
        [UInt8(value & 0xff), UInt8((value >> 8) & 0xff)]
    }

    private static func le32(_ value: Int) -> [UInt8] {
        [UInt8(value & 0xff), UInt8((value >> 8) & 0xff), UInt8((value >> 16) & 0xff), UInt8((value >> 24) & 0xff)]
    }
}

struct DiagnosticsActivityView: UIViewControllerRepresentable {
    let items: [URL]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
