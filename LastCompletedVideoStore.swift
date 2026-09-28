import Foundation

/// Finds the newest completed render that is still reachable after the app is relaunched.
/// The app-owned Exports directory is always available. If the user granted a visible
/// Files folder (including the On My iPhone root), its security-scoped bookmark is also
/// checked so the last render can be exported again without reprocessing.
enum LastCompletedVideoStore {
    private static let exportFolderBookmarkKey = "RIFE60.ExportFolderBookmark.v1"
    private static let supportedExtensions: Set<String> = ["mov", "mp4", "m4v"]

    static func latestCompletedVideo() -> URL? {
        var candidates: [URL] = []

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        candidates.append(contentsOf: videos(in: exports))

        if let folder = selectedExportFolder() {
            let secured = folder.startAccessingSecurityScopedResource()
            candidates.append(contentsOf: videos(in: folder))
            if secured { folder.stopAccessingSecurityScopedResource() }
        }

        return candidates.max { lhs, rhs in
            modificationDate(lhs) < modificationDate(rhs)
        }
    }

    private static func videos(in folder: URL) -> [URL] {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return urls.filter { url in
            supportedExtensions.contains(url.pathExtension.lowercased()) &&
            ((try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false)
        }
    }

    private static func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    private static func selectedExportFolder() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: exportFolderBookmarkKey) else { return nil }
        var stale = false
        return try? URL(
            resolvingBookmarkData: data,
            options: [.withoutUI],
            relativeTo: nil,
            bookmarkDataIsStale: &stale
        )
    }
}
