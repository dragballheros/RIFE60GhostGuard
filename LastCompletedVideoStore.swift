import Foundation

/// Persists the exact last completed output, rather than guessing by modification
/// date from every video in a user-selected Files folder. Selected folders may
/// contain original source videos, so scanning them can accidentally offer the
/// wrong file as the last export.
enum LastCompletedVideoStore {
    private static let exportFolderBookmarkKey = "RIFE60.ExportFolderBookmark.v1"
    private static let lastCompletedBookmarkKey = "RIFE60.LastCompletedVideoBookmark.v1"
    private static let lastCompletedURLKey = "RIFE60.LastCompletedVideoURL.v1"
    private static let supportedExtensions: Set<String> = ["mov", "mp4", "m4v"]

    /// Records an output only after the render has been persisted and validated.
    /// Call this while the selected folder's security scope is active whenever
    /// the URL points outside the app container.
    @discardableResult
    static func recordCompletedVideo(_ url: URL) -> Bool {
        guard isUsableVideoFile(url) else { return false }
        UserDefaults.standard.set(url.absoluteString, forKey: lastCompletedURLKey)
        do {
            let bookmark = try url.bookmarkData(
                options: .minimalBookmark,
                includingResourceValuesForKeys: [.isRegularFileKey, .fileSizeKey],
                relativeTo: nil
            )
            UserDefaults.standard.set(bookmark, forKey: lastCompletedBookmarkKey)
        } catch {
            // The stored URL remains a fallback for app-container exports. A
            // security-scoped external URL may require a fresh folder permission.
            UserDefaults.standard.removeObject(forKey: lastCompletedBookmarkKey)
        }
        return true
    }

    static func latestCompletedVideo() -> URL? {
        if let data = UserDefaults.standard.data(forKey: lastCompletedBookmarkKey) {
            var stale = false
            if let url = try? URL(
                resolvingBookmarkData: data,
                options: [.withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            ), isUsableVideoFile(url) {
                if stale {
                    _ = recordCompletedVideo(url)
                }
                return url
            }
        }

        if let rawURL = UserDefaults.standard.string(forKey: lastCompletedURLKey),
           let url = URL(string: rawURL),
           isUsableVideoFile(url) {
            return url
        }

        // Compatibility fallback for releases before explicit output bookmarks
        // existed. Only inspect the app's private Exports folder; never search
        // the user-selected folder, which can contain unrelated input videos.
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        return videos(in: exports).max { modificationDate($0) < modificationDate($1) }
    }

    private static func isUsableVideoFile(_ url: URL) -> Bool {
        guard supportedExtensions.contains(url.pathExtension.lowercased()),
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              (attributes[.type] as? FileAttributeType) == .typeRegular,
              let size = attributes[.size] as? NSNumber,
              size.int64Value > 0 else {
            return false
        }
        return true
    }

    private static func videos(in folder: URL) -> [URL] {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return urls.filter(isUsableVideoFile)
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
