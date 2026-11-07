import AVFoundation

/// Normalizes every processing input to a 720p working raster before any
/// restoration, RIFE, CUGAN, or outline stage runs.
///
/// This is a spatial working-resolution step, not a final delivery bitrate
/// setting. The original source URL is still retained by the caller for final
/// audio restoration when required.
final class Video720pPreprocessor {
    private let fm = FileManager.default

    func prepare(sourceURL: URL) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ProcessorError.missingVideoTrack
        }

        let naturalSize = try await track.load(.naturalSize)
        let width = Int(abs(naturalSize.width).rounded())
        let height = Int(abs(naturalSize.height).rounded())
        guard width > 0, height > 0 else {
            throw ProcessorError.conversionFailed("720p preprocessor received an invalid video size")
        }

        let longSide = max(width, height)
        let shortSide = min(width, height)
        if longSide == 1280 && shortSide == 720 {
            DiagnosticsLogger.shared.log("720p input normalization • source already 720p • no resize required")
            return sourceURL
        }

        guard let exporter = AVAssetExportSession(asset: asset,
                                                  presetName: AVAssetExportPreset1280x720) else {
            throw ProcessorError.conversionFailed("could not create 720p export session")
        }

        let outputURL = fm.temporaryDirectory
            .appendingPathComponent("RIFE60-720p-working-" + UUID().uuidString)
            .appendingPathExtension("mov")
        try? fm.removeItem(at: outputURL)

        exporter.shouldOptimizeForNetworkUse = false
        exporter.allowsParallelizedExport = false

        DiagnosticsLogger.shared.log(
            "720p input normalization • \(width)x\(height) → 720p working target • no final bitrate target applied"
        )

        let outputType: AVFileType = exporter.supportedFileTypes.contains(.mov) ? .mov : (exporter.supportedFileTypes.first ?? .mov)
        do {
            try await exporter.export(to: outputURL, as: outputType)
        } catch {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.writer("720p input normalization failed: \(error.localizedDescription)")
        }

        let normalized = AVURLAsset(url: outputURL)
        guard let normalizedTrack = try await normalized.loadTracks(withMediaType: .video).first else {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.conversionFailed("720p normalization produced no video track")
        }

        let normalizedSize = try await normalizedTrack.load(.naturalSize)
        let normalizedWidth = Int(abs(normalizedSize.width).rounded())
        let normalizedHeight = Int(abs(normalizedSize.height).rounded())
        let normalizedLong = max(normalizedWidth, normalizedHeight)
        let normalizedShort = min(normalizedWidth, normalizedHeight)

        guard normalizedLong == 1280 && normalizedShort == 720 else {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.conversionFailed(
                "720p normalization produced \(normalizedWidth)x\(normalizedHeight) instead of a 720p raster"
            )
        }

        DiagnosticsLogger.shared.log(
            "720p input normalization complete • working source=\(normalizedWidth)x\(normalizedHeight) • original remains untouched"
        )
        return outputURL
    }
}
