import Foundation

@main
enum LastCompletedVideoStoreTests {
    static func main() throws {
        let suite = "RIFE60.LastCompletedVideoStoreTests.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suite) else {
            fatalError("FAIL: could not create isolated UserDefaults suite")
        }
        defaults.removePersistentDomain(forName: suite)

        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("last-video-store-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: folder)
            defaults.removePersistentDomain(forName: suite)
            // The test store uses standard defaults; remove its uniquely scoped keys.
            UserDefaults.standard.removeObject(forKey: "RIFE60.LastCompletedVideoBookmark.v1")
            UserDefaults.standard.removeObject(forKey: "RIFE60.LastCompletedVideoURL.v1")
            UserDefaults.standard.removeObject(forKey: "RIFE60.ExportFolderBookmark.v1")
        }

        let output = folder.appendingPathComponent("RIFE60-completed-output.mov")
        let unrelatedInput = folder.appendingPathComponent("new-source-input.mp4")
        try Data(repeating: 17, count: 4096).write(to: output)
        guard LastCompletedVideoStore.recordCompletedVideo(output) else {
            fatalError("FAIL: valid completed output should be recordable")
        }

        // The source video is created after the result, so a "newest file in the
        // selected directory" implementation incorrectly chooses this input.
        try Data(repeating: 99, count: 8192).write(to: unrelatedInput)
        let folderBookmark = try folder.bookmarkData(
            options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil
        )
        UserDefaults.standard.set(folderBookmark, forKey: "RIFE60.ExportFolderBookmark.v1")

        guard let selected = LastCompletedVideoStore.latestCompletedVideo(),
              selected.standardizedFileURL == output.standardizedFileURL else {
            fatalError("FAIL: last completed export must remain the recorded output, not the newer source video")
        }

        try FileManager.default.removeItem(at: output)
        guard LastCompletedVideoStore.latestCompletedVideo()?.standardizedFileURL != output.standardizedFileURL else {
            fatalError("FAIL: deleted output must not be returned")
        }
        print("PASS: export selection tracks exact validated output, not newest source file")
    }
}
