import Foundation

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "BatchCheck", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}
@main
struct BatchChecks {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("batch-check-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let image = root.appendingPathComponent("image.jpeg"), video = root.appendingPathComponent("video.mov")
        try Data([1, 2, 3]).write(to: image); try Data([4, 5, 6]).write(to: video)
        let store = BatchMediaQueueStore(root: root.appendingPathComponent("queue"))
        var queue = try store.stage([image, video, image], progress: { _ in })
        try require(queue.items.map(\.kind) == [.image, .video, .image], "mixed selection order")
        let stroke = WatermarkBrushStroke(points: [WatermarkBrushPoint(x: 0.2, y: 0.3)], radiusX: 0.02, radiusY: 0.03, erase: false)
        guard let mask = WatermarkBrushMask(strokes: [stroke]).region else { throw CancellationError() }
        queue.items[0].masks = [mask]
        queue.items[1].removeWatermark = false
        queue.settingsKey = "pipeline-v3-final-size|watermark=true|colorpop=true|colorpopStrength=1.0"
        try store.save(queue)
        guard let loaded = try store.load() else { throw CancellationError() }
        try require(loaded.items[0].masks == [mask] && loaded.items[2].masks.isEmpty && !loaded.items[1].removeWatermark, "independent per-item masks and opt-out persist")
        try require(loaded.settingsKey == queue.settingsKey, "processing settings persist")
        try require(FileManager.default.fileExists(atPath: store.sourceURL(for: loaded.items[0], in: loaded).path), "owned source copy persists")
        let ids = queue.items.map(\.id)
        var active = 0, maximumActive = 0
        var events: [String] = []
        try await BatchSerialRunner.run(ids: ids, shouldStop: { false }) { id in
            active += 1; maximumActive = max(maximumActive, active)
            events.append("start:\(id)")
            try await Task.sleep(nanoseconds: 5_000_000)
            events.append("saved:\(id)"); active -= 1
        }
        try require(maximumActive == 1 && active == 0, "no simultaneous processing")
        try require(events == ids.flatMap { ["start:\($0)", "saved:\($0)"] }, "each item finishes exporting before the next begins")
        queue.items[0].state = .completed; queue.items[0].outputURL = image
        queue.items[1].state = .processing
        try store.save(queue)
        let resumed = try store.load()
        try require(resumed?.nextItemID == ids[1] && resumed?.items[0].outputURL == image, "resume skips completed output and retains active item")
        var stopped = false, cancelledIDs: [UUID] = []
        do {
            try await BatchSerialRunner.run(ids: ids, shouldStop: { stopped }) { id in cancelledIDs.append(id); stopped = true }
            try require(false, "cancellation must stop")
        } catch is CancellationError { }
        try require(cancelledIDs == [ids[0]], "cancel cannot start next item")
        var failedIDs: [UUID] = []
        do {
            try await BatchSerialRunner.run(ids: ids, shouldStop: { false }) { id in
                failedIDs.append(id)
                if id == ids[1] { throw NSError(domain: "Fixture", code: 2) }
            }
            try require(false, "failure must stop")
        } catch { }
        try require(failedIDs == [ids[0], ids[1]], "failure cannot start later item")
        try store.discard()
        try require(try store.load() == nil, "queue cleared")
        try require(FileManager.default.fileExists(atPath: image.path) && FileManager.default.fileExists(atPath: video.path), "clearing queue preserves originals / exported files")
        print("BATCH_MEDIA_PASS: mixed import order; source copies; per-item brush masks/settings; serial render+export; restart skip; cancellation; failure; safe cleanup")
    }
}
