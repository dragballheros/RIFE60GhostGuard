import Foundation

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "X265CacheCheck", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}

@main
struct X265PresetCacheChecks {
    static func main() throws {
        try run()
        print("x265 preset cache checks passed")
    }
}

private func run() throws {
let digest = "abc"
let preset = "preset-a"
let match = X265PresetCacheRecord(sourceSHA256: digest, preset: preset, complete: true, outputBytes: 42)
try require(X265PresetCache.reusableOutput(sourceSHA256: digest, preset: preset, record: match, outputBytes: 42) != nil, "finished preset file is reused")
try require(X265PresetCache.reusableOutput(sourceSHA256: digest, preset: "other", record: match, outputBytes: 42) == nil, "a different preset is encoded again")
var partial = match
partial.complete = false
try require(X265PresetCache.reusableOutput(sourceSHA256: digest, preset: preset, record: partial, outputBytes: 42) == nil, "an interrupted encode is not reused")
try require(X265PresetCache.reusableOutput(sourceSHA256: digest, preset: preset, record: match, outputBytes: 7) == nil, "a truncated file is not reused")
try require(InputMediaKind.detect(for: URL(fileURLWithPath: "/tmp/source.mkv")) == .video, "mkv enters the video pipeline")
try require(InputMediaKind.detect(for: URL(fileURLWithPath: "/tmp/source.webm")) == .video, "webm enters the video pipeline")
}
