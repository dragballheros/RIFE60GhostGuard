import CryptoKit
import Darwin
import Foundation

/// Tier 1 MiniMax H3 install. No paid API, no cloud inference, no encoder-student
/// distillation, and no DMD training. GitHub Actions is not used to quantize
/// weights: a free runner cannot hold the official tree, and weights are not
/// uploaded as artifacts.
///
/// The phone downloads one manifest file into a scratch path, checks it, keeps
/// it only when the stored set stays within the Tier 1 budget, then deletes the
/// scratch file before the next file starts.
enum H3Tier1OnDeviceConverter {
    static let targetStoreBytes: Int64 = 44 * 1_073_741_824
    /// Raw manifest on the repository default branch. The app also ships the same JSON.
    static let defaultManifestURLString = "https://raw.githubusercontent.com/dragballheros/RIFE60GhostGuard/main/Models/h3-tier1-manifest.json"

    static func stageTier1Layout(into modelDirectory: URL) throws {
        let fm = FileManager.default
        let roots = [modelDirectory, modelDirectory.appendingPathComponent("tier1-weights", isDirectory: true)]
        guard let found = roots.first(where: {
            fm.fileExists(atPath: $0.appendingPathComponent("minimax_h3_ref2va_pruned-Q8_0.gguf").path)
        }) else { return }
        func source(_ names: [String]) -> URL? {
            for name in names {
                let candidate = found.appendingPathComponent(name)
                if fm.fileExists(atPath: candidate.path) { return candidate }
            }
            return nil
        }
        func link(directory: String, name: String, from source: URL) throws {
            let folder = modelDirectory.appendingPathComponent(directory, isDirectory: true)
            try fm.createDirectory(at: folder, withIntermediateDirectories: true)
            let destination = folder.appendingPathComponent(name)
            if fm.fileExists(atPath: destination.path) { return }
            try fm.createSymbolicLink(at: destination, withDestinationURL: source)
        }
        let ref = found.appendingPathComponent("minimax_h3_ref2va_pruned-Q8_0.gguf")
        try link(directory: "FL2VA/transformer", name: ref.lastPathComponent, from: ref)
        try link(directory: "Ref2VA/transformer", name: ref.lastPathComponent, from: ref)
        if let qwen = source(["qwen3vl_32b_minimax_h3-Q4_K_M.gguf"]) {
            try link(directory: "FL2VA/text_encoder", name: qwen.lastPathComponent, from: qwen)
            try link(directory: "Ref2VA/text_encoder", name: qwen.lastPathComponent, from: qwen)
        }
        if let video = source(["vae_minimax_h3_video_vae_fp16.safetensors", "vae/minimax_h3_video_vae_fp16.safetensors"]) {
            try link(directory: "FL2VA/video_vae/source", name: "minimax_h3_video_vae_fp16.safetensors", from: video)
            try link(directory: "Ref2VA/video_vae/source", name: "minimax_h3_video_vae_fp16.safetensors", from: video)
        }
        if let audio = source(["vae_minimax_h3_audio_vae_fp32.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors"]) {
            try link(directory: "FL2VA/audio_vae", name: "minimax_h3_audio_vae_fp32.safetensors", from: audio)
            try link(directory: "Ref2VA/audio_vae", name: "minimax_h3_audio_vae_fp32.safetensors", from: audio)
        }
        let marker = Data("{\"architectures\":[\"MiniMaxH3\"],\"weight_format\":\"gguf-stream\"}\n".utf8)
        for relative in [
            "FL2VA/transformer/config.json",
            "Ref2VA/transformer/config.json",
            "FL2VA/text_encoder/config.json",
            "Ref2VA/text_encoder/config.json"
        ] {
            try writeIfAbsent(relative, data: marker, into: modelDirectory)
        }
        // Official MiniMax-H3 VAE configs. The runtime opens
        // {audio_vae}/config.json and {video_vae/source}/../config.json and
        // parses latents_mean / latents_std. A marker file is not enough.
        let audioConfig = Data(Self.audioVAEConfigJSON.utf8)
        let videoConfig = Data(Self.videoVAEConfigJSON.utf8)
        for relative in ["FL2VA/audio_vae/config.json", "Ref2VA/audio_vae/config.json"] {
            try writeLatentConfig(relative, data: audioConfig, into: modelDirectory)
        }
        for relative in ["FL2VA/video_vae/config.json", "Ref2VA/video_vae/config.json"] {
            try writeLatentConfig(relative, data: videoConfig, into: modelDirectory)
        }
        let indexURL = modelDirectory.appendingPathComponent("Ref2VA/transformer/model.safetensors.index.json")
        if !fm.fileExists(atPath: indexURL.path) {
            let index = Data("{\"weight_map\":{\"ref2va\":\"minimax_h3_ref2va_pruned-Q8_0.gguf\"}}\n".utf8)
            try fm.createDirectory(at: indexURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try index.write(to: indexURL, options: .atomic)
        }
        let bundledSize = (try? Bundle.main.url(forResource: "h3-tokenizer", withExtension: "json")
            .flatMap { try fm.attributesOfItem(atPath: $0.path)[.size] as? NSNumber }?.int64Value) ?? 0
        CrashCapture.step("H3 copying bundled tokenizer bytes=\(bundledSize)")
        try installBundledTokenizer(into: modelDirectory)
        CrashCapture.step("H3 staged Q8_0 transformer, Q4_K_M text encoder, both VAE files, and the configs the runtime opens")
    }

    private static func writeIfAbsent(_ relative: String, data: Data, into modelDirectory: URL) throws {
        let url = modelDirectory.appendingPathComponent(relative)
        if FileManager.default.fileExists(atPath: url.path) { return }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    /// Replaces a missing or marker-only config so an install that already
    /// downloaded the VAE weights picks up the latent stats without a new download.
    private static func writeLatentConfig(_ relative: String, data: Data, into modelDirectory: URL) throws {
        let url = modelDirectory.appendingPathComponent(relative)
        if FileManager.default.fileExists(atPath: url.path),
           let existing = try? String(contentsOf: url, encoding: .utf8),
           existing.contains("\"latents_mean\""),
           existing.contains("\"latents_std\"") {
            return
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    /// MiniMaxAI/MiniMax-H3 audio_vae/config.json (32 channels).
    static let audioVAEConfigJSON = """
    {"_class_name":"AutoencoderKLMiniMaxH3Audio","latent_channels":32,"sampling_rate":32000,"latents_mean":[-0.020211687488382354,0.3876466479950502,-0.04398279799186767,-0.28591514936373,0.08179686214561671,-0.35782641352446604,0.040623809960919084,-0.01552534501956604,-0.223362481667332,0.1821006842509091,0.2941778783780663,-0.07901167601970885,-0.056815072777201,-0.3699028221860095,-0.31616315591624855,0.5905951377425391,-0.052139568068853864,0.013673160263486295,-0.03691647864630577,0.09732660653298163,-0.3394662328788498,-0.30685677538541667,-0.24504598907458763,-0.034698524462007344,0.02868032184767538,-0.21217779266454084,-0.1678263169941987,0.3221287889040614,-0.1223055851554907,0.4356604928128464,-0.0502599202236253,0.3979258376211797],"latents_std":[1.6895524230479284,2.76263727217653,1.7945344281264435,1.6801681847309828,1.6390226546605453,2.7788298348882177,1.7659090095747236,1.6199757612137327,2.6336525640336896,1.8539356672817833,2.5056497896915633,1.811019237886178,1.9579657790720237,1.6685498243529284,1.4922469314453364,3.298670198067373,1.9491804496832168,1.8720003270431442,1.8334080103291832,1.6488070416529093,1.6176957696319716,1.9131449234774398,1.5695245398428617,1.6943659940415912,1.8318420762504692,1.5540637421583379,1.9344930328968526,1.599198216109855,1.718045989838149,1.6307219190837705,1.8661226051202384,1.5613768203168363]}
    """

    /// MiniMaxAI/MiniMax-H3 FL2VA/video_vae/config.json (24 channels).
    static let videoVAEConfigJSON = """
    {"_class_name":"MiniMaxH3VideoVAE","latent_channels":24,"source_path":"source","latents_mean":[0.858090341091156,-0.9606591463088989,1.0661640167236328,-0.5090325474739075,-0.2727581858634949,-1.3675414323806763,-0.2553254961967468,-0.26907554268836975,-0.5376840829849243,-0.0464097298681736,0.6657370328903198,0.19690127670764923,-0.5460608005523682,-0.4035342037677765,-0.23683024942874908,0.25928452610969543,-0.30133944749832153,0.211341992020607,-1.1206848621368408,0.3581933379173279,-0.04225143790245056,0.2604829967021942,0.22864092886447906,0.7056031823158264],"latents_std":[1.2223774194717407,1.2767263650894165,1.68317747116088865,1.7549455165863037,1.5636216402053833,2.194143533706665,0.96531379222869875,1.05698859691619875,0.841948926448822,0.7729952931404114,1.8955937623977661,0.946841835975647,0.7996809482574463,0.44988900423049925,0.7197399735450745,0.69362932443618775,2.961095094680786,2.7694199085235595,3.0496184825897215,2.1088054180145265,3.276226282119751,3.1627357006073,2.28168129920959475,2.6127843856811525]}
    """

    /// The Q8 manifest does not include a tokenizer. The official tokenizer ships in the app
    /// and is copied next to the weights so generation can encode a prompt.
    static func installBundledTokenizer(into modelDirectory: URL) throws {
        guard let bundled = Bundle.main.url(forResource: "h3-tokenizer", withExtension: "json") else {
            return
        }
        let folders = ["FL2VA/tokenizer", "Ref2VA/tokenizer"]
        for relative in folders {
            let directory = modelDirectory.appendingPathComponent(relative, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: destination.path) { continue }
            try FileManager.default.copyItem(at: bundled, to: destination)
        }
    }
    static let peakScratchCeilingBytes: Int64 = 150 * 1_073_741_824
    static let manifestDefaultsKey = "RIFE60.H3.Tier1ManifestURL"

    struct Manifest: Codable, Equatable, Sendable {
        var files: [File]

        struct File: Codable, Equatable, Sendable {
            var path: String
            var url: String
            var sha256: String
            var bytes: Int64
        }
    }

    enum Failure: LocalizedError {
        case missingManifest
        case unsafePath(String)
        case shardTooLarge(String)
        case notEnoughFreeSpace
        case storeBudgetExceeded(String)
        case checksum(String)
        case http(String, Int)

        var errorDescription: String? {
            switch self {
            case .missingManifest:
                return "Set a Tier 1 manifest URL for already-published compressed H3 weights. This phone will not download the full official tree as the stored model."
            case .unsafePath(let path):
                return "Refusing model path \(path)."
            case .shardTooLarge(let path):
                return "\(path) is larger than the 150 GB scratch ceiling. It was not downloaded."
            case .notEnoughFreeSpace:
                return "Not enough free space to hold this one shard plus a 2 GB margin and still delete the scratch file. The kept store can grow to about 44 GB, one file at a time."
            case .storeBudgetExceeded(let path):
                return "Keeping \(path) would pass the about 44 GB Tier 1 store. The scratch file was deleted."
            case .checksum(let path):
                return "Checksum failed for \(path). The scratch file was deleted."
            case .http(let path, let status):
                return "Download failed for \(path) (HTTP \(status))."
            }
        }
    }

    static func convert(
        manifest: Manifest,
        into directory: URL,
        progress: @escaping @Sendable (String) -> Void
    ) async throws {
        let fm = FileManager.default
        let scratch = directory.appendingPathComponent("tier1-scratch", isDirectory: true)
        let kept = directory.appendingPathComponent("tier1-weights", isDirectory: true)
        try fm.createDirectory(at: scratch, withIntermediateDirectories: true)
        try fm.createDirectory(at: kept, withIntermediateDirectories: true)
        var stored = try directorySize(kept)
        for file in manifest.files {
            try Task.checkCancellation()
            let scratchURL = scratch.appendingPathComponent(safeName(file.path))
            let keptURL = kept.appendingPathComponent(safeName(file.path))
            CrashCapture.step("H3 shard \(file.path) bytes=\(file.bytes)")
            if fm.fileExists(atPath: keptURL.path), try await alreadyVerified(file, at: keptURL, directory: directory) {
                progress("Kept \(file.path). Scratch deleted.")
                continue
            }
            if fm.fileExists(atPath: keptURL.path) {
                let digest = try await hashFile(keptURL) { read in
                    progress("bytes \(read)/\(file.bytes) · \(file.path)")
                }
                guard digest == file.sha256.lowercased() else { throw Failure.checksum(file.path) }
                try rememberVerified(file, at: keptURL, digest: digest, directory: directory)
                progress("Kept \(file.path). Scratch deleted.")
                continue
            }
            let present = (try? fm.attributesOfItem(atPath: scratchURL.path)[.size] as? NSNumber)?.int64Value ?? 0
            let bytesStillNeeded = present > 0 && present < file.bytes ? file.bytes - present : (present == file.bytes ? 0 : file.bytes)
            try validate(file, bytesStillNeeded: bytesStillNeeded)
            // A killed app must keep the partial shard. A failed checksum still deletes it.
            defer {
                if !Task.isCancelled {
                    try? fm.removeItem(at: scratchURL)
                }
            }
            let scratchSize = (try? fm.attributesOfItem(atPath: scratchURL.path)[.size] as? NSNumber)?.int64Value ?? 0
            var digest: String?
            if scratchSize == file.bytes, try await alreadyVerified(file, at: scratchURL, directory: directory) {
                digest = file.sha256.lowercased()
                progress("bytes \(file.bytes)/\(file.bytes) · \(file.path)")
            } else if scratchSize == file.bytes {
                CrashCapture.step("H3 checksum \(file.path)")
                digest = try await hashFile(scratchURL) { read in
                    progress("bytes \(read)/\(file.bytes) · \(file.path)")
                }
            }
            if digest == file.sha256.lowercased() {
                progress("bytes \(file.bytes)/\(file.bytes) · \(file.path)")
            } else {
                if scratchSize > file.bytes { try? fm.removeItem(at: scratchURL) }
                CrashCapture.step("H3 downloading \(file.path)")
                progress("Downloading \(file.path)")
                try await download(file, to: scratchURL) { written, expected in
                    progress("bytes \(written)/\(expected) · \(file.path)")
                }
                CrashCapture.step("H3 checksum \(file.path)")
                digest = try await hashFile(scratchURL) { read in
                    progress("bytes \(read)/\(file.bytes) · \(file.path)")
                }
            }
            guard digest == file.sha256.lowercased() else {
                throw Failure.checksum(file.path)
            }
            if file.path.lowercased().hasSuffix(".gguf") {
                CrashCapture.step("H3 verify GGUF header \(file.path)")
                try await Task.detached(priority: .utility) {
                    try H3GGUFStream.confirmDownloadableShard(at: scratchURL, path: file.path)
                }.value
            }
            guard stored + file.bytes <= targetStoreBytes else {
                throw Failure.storeBudgetExceeded(file.path)
            }
            let destination = keptURL
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.moveItem(at: scratchURL, to: destination)
            if let digest {
                try rememberVerified(file, at: destination, digest: digest, directory: directory)
            }
            stored += file.bytes
            progress("Kept \(file.path). Scratch deleted.")
        }
    }

    private static func validate(_ file: Manifest.File, bytesStillNeeded: Int64) throws {
        if file.path.contains("..") || file.path.contains("\\") || file.path.hasPrefix("/") {
            throw Failure.unsafePath(file.path)
        }
        if file.bytes <= 0 || file.bytes > peakScratchCeilingBytes {
            throw Failure.shardTooLarge(file.path)
        }
        let margin = bytesStillNeeded == 0 ? Int64(0) : 2 * 1_073_741_824
        if freeBytes() < bytesStillNeeded + margin {
            throw Failure.notEnoughFreeSpace
        }
    }

    /// One ranged piece. A single background task for shard 1 (about 21 GB) is what
    /// iOS kept relaunching; this stays on a foreground session and survives a kill.
    private static let pieceBytes: Int64 = 64 * 1_024 * 1_024

    private static let pieceSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 180
        configuration.timeoutIntervalForResource = 60 * 60
        configuration.waitsForConnectivity = true
        return URLSession(configuration: configuration)
    }()

    private static func download(
        _ file: Manifest.File,
        to destination: URL,
        onBytes: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws {
        guard let url = URL(string: file.url) else { throw Failure.http(file.path, -1) }
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
        }
        var written = (try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? NSNumber)?.int64Value ?? 0
        if written > file.bytes {
            try FileManager.default.removeItem(at: destination)
            FileManager.default.createFile(atPath: destination.path, contents: nil)
            written = 0
        }
        onBytes(written, file.bytes)
        while written < file.bytes {
            try Task.checkCancellation()
            let pieceEnd = min(written + pieceBytes, file.bytes) - 1
            var request = H3ModelDownloader.resumeRequest(for: url, existingBytes: written, end: pieceEnd)
            request.timeoutInterval = 180
            let (temporary, response) = try await pieceSession.download(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            let body = (try? FileManager.default.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber)?.int64Value ?? 0
            if status == 206 {
                let expectedPiece = pieceEnd - written + 1
                guard body == expectedPiece else {
                    try? FileManager.default.removeItem(at: temporary)
                    throw Failure.http(file.path, status)
                }
                written = try appendFile(temporary, to: destination, already: written, expected: file.bytes, onBytes: onBytes)
            } else if status == 200, body == file.bytes {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: temporary, to: destination)
                written = file.bytes
                onBytes(written, file.bytes)
            } else {
                try? FileManager.default.removeItem(at: temporary)
                throw Failure.http(file.path, status)
            }
        }
        guard written == file.bytes else { throw Failure.http(file.path, -1) }
    }

    private static func appendFile(
        _ temporary: URL,
        to destination: URL,
        already: Int64,
        expected: Int64,
        onBytes: @escaping @Sendable (Int64, Int64) -> Void
    ) throws -> Int64 {
        let input = try FileHandle(forReadingFrom: temporary)
        defer {
            try? input.close()
            try? FileManager.default.removeItem(at: temporary)
        }
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        try output.seekToEnd()
        var written = already
        while true {
            let chunk = try input.read(upToCount: 8 * 1024 * 1024) ?? Data()
            if chunk.isEmpty { break }
            try output.write(contentsOf: chunk)
            written += Int64(chunk.count)
            if written == expected || written % (32 * 1_024 * 1_024) < Int64(chunk.count) {
                onBytes(written, expected)
            }
        }
        return written
    }

    private static let checksumChunkBytes = 4 * 1024 * 1024

    private struct VerifiedRecord: Codable {
        var bytes: Int64
        var modifiedMillis: Int64
        var sha256: String
    }

    private static func verifiedMarkerURL(_ directory: URL) -> URL {
        directory.appendingPathComponent("tier1-verified.json")
    }

    private static func modificationMillis(_ url: URL) -> Int64? {
        guard let date = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate else { return nil }
        return Int64((date.timeIntervalSince1970 * 1000).rounded())
    }

    private static func alreadyVerified(_ file: Manifest.File, at url: URL, directory: URL) async throws -> Bool {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value,
              size == file.bytes,
              let modified = modificationMillis(url),
              let data = try? Data(contentsOf: verifiedMarkerURL(directory)),
              let map = try? JSONDecoder().decode([String: VerifiedRecord].self, from: data),
              let record = map[file.path],
              record.bytes == size,
              record.modifiedMillis == modified,
              record.sha256 == file.sha256.lowercased() else {
            return false
        }
        CrashCapture.step("H3 checksum skipped \(file.path); verified marker matches size and modification date")
        return true
    }

    private static func rememberVerified(_ file: Manifest.File, at url: URL, digest: String, directory: URL) throws {
        guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value,
              let modified = modificationMillis(url) else { return }
        let markerURL = verifiedMarkerURL(directory)
        var map = (try? Data(contentsOf: markerURL)).flatMap { try? JSONDecoder().decode([String: VerifiedRecord].self, from: $0) } ?? [:]
        map[file.path] = VerifiedRecord(bytes: size, modifiedMillis: modified, sha256: digest.lowercased())
        let data = try JSONEncoder().encode(map)
        try data.write(to: markerURL, options: .atomic)
    }

    private static func hashFile(_ url: URL, progress: @escaping @Sendable (Int64) -> Void) async throws -> String {
        try await Task.detached(priority: .utility) {
            try await sha256Streaming(of: url, progress: progress)
        }.value
    }

    private static func sha256Streaming(of url: URL, progress: @escaping @Sendable (Int64) -> Void) async throws -> String {
        var handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        armUncached(handle)
        CrashCapture.step("H3 allocating checksum buffer bytes=\(checksumChunkBytes) for \(url.lastPathComponent)")
        var hasher = SHA256()
        var readTotal: Int64 = 0
        var loggedFirstChunk = false
        while true {
            if CrashCapture.memoryWarningActive {
                try handle.close()
                CrashCapture.step("H3 checksum paused; released \(checksumChunkBytes)B read buffer")
                try await CrashCapture.waitWhileMemoryWarning()
                handle = try FileHandle(forReadingFrom: url)
                armUncached(handle)
                try handle.seek(toOffset: UInt64(readTotal))
                CrashCapture.step("H3 checksum resumed at byte \(readTotal)")
            }
            let chunk: Data = try autoreleasepool {
                try handle.read(upToCount: checksumChunkBytes) ?? Data()
            }
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
            readTotal += Int64(chunk.count)
            if !loggedFirstChunk {
                loggedFirstChunk = true
                CrashCapture.step("H3 checksum first chunk bytes=\(chunk.count) released")
            }
            if readTotal % (32 * 1_024 * 1_024) < Int64(chunk.count) {
                progress(readTotal)
                CrashCapture.note("H3 checksum \(url.lastPathComponent) \(readTotal)")
            }
        }
        CrashCapture.step("H3 checksum finished \(url.lastPathComponent) bytes=\(readTotal)")
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func armUncached(_ handle: FileHandle) {
        _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        armUncached(handle)
        var hasher = SHA256()
        while true {
            let chunk: Data = try autoreleasepool {
                try handle.read(upToCount: checksumChunkBytes) ?? Data()
            }
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func directorySize(_ url: URL) throws -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let item as URL in enumerator {
            let size = try item.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            total += Int64(size)
        }
        return total
    }

    private static func freeBytes() -> Int64 {
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let attributes = try? FileManager.default.attributesOfFileSystem(forPath: root.path)
        return (attributes?[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
    }

    private static func safeName(_ path: String) -> String {
        path.replacingOccurrences(of: "/", with: "_")
    }
}

/// Drains the old whole-file background session. New shards use ranged foreground
/// downloads. The session is created only after the background completion handler
/// is stored, then every legacy task is cancelled so iOS stops relaunching on shard 1.
final class H3Tier1BackgroundTransfer: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    static let shared = H3Tier1BackgroundTransfer()
    static let identifier = "com.za.rife60ghostguard.h3.tier1"

    private let lock = NSLock()
    private var session: URLSession?
    private var backgroundCompletion: (() -> Void)?
    private var handlingBackgroundEvents = false
    private var deliveredCompletion = false

    func beginBackgroundEvents(_ completion: @escaping () -> Void) {
        lock.lock()
        backgroundCompletion = completion
        handlingBackgroundEvents = true
        deliveredCompletion = false
        createSessionLocked()
        let active = session
        lock.unlock()
        active?.invalidateAndCancel()
    }

    private func createSessionLocked() {
        guard session == nil else { return }
        let configuration = URLSessionConfiguration.background(withIdentifier: Self.identifier)
        configuration.sessionSendsLaunchEvents = false
        configuration.isDiscretionary = false
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        downloadTask.cancel()
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // Leave the file in place so the system can delete it. Moving a finished
        // multi-gigabyte shard here blocks the delegate queue and iOS relaunches the app.
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {}

    func urlSession(_ session: URLSession, didBecomeInvalidWithError error: Error?) {
        lock.lock()
        self.session = nil
        lock.unlock()
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        lock.lock()
        if deliveredCompletion {
            lock.unlock()
            return
        }
        deliveredCompletion = true
        let handler = backgroundCompletion
        backgroundCompletion = nil
        lock.unlock()
        DispatchQueue.main.async { handler?() }
    }

}
