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

        // AVAssetExportPreset1280x720 is the spatial normalization target.
        // If the source is already 1280x720, avoid an unnecessary generation loss.
        let orientedWidth = Int(abs(try await track.load(.naturalSize).width).rounded())
        let orientedHeight = Int(abs(try await track.load(.naturalSize).height).rounded())
        if orientedWidth == 1280 && orientedHeight == 720 {
            DiagnosticsLogger.shared.log("720p input normalization • source already 1280x720 • no resize required")
            return sourceURL
        }

        guard AVAssetExportSession.compatibility(ofExportPreset: AVAssetExportPreset1280x720,
                                                 with: asset,
                                                 outputFileType: .mov) else {
            throw ProcessorError.conversionFailed("720p preprocessor is not compatible with this source video")
        }

        let outputURL = fm.temporaryDirectory
            .appendingPathComponent("RIFE60-720p-working-(UUID().uuidString)")
            .appendingPathExtension("mov")
        try? fm.removeItem(at: outputURL)

        guard let exporter = AVAssetExportSession(asset: asset,
                                                  presetName: AVAssetExportPreset1280x720) else {
            throw ProcessorError.conversionFailed("could not create 720p export session")
        }

        exporter.shouldOptimizeForNetworkUse = false
        exporter.allowsParallelizedExport = false

        DiagnosticsLogger.shared.log(
            "720p input normalization • (width)x(height) → 1280x720 working target • bitrate not used as a processing quality control"
        )

        let outputType: AVFileType = exporter.supportedFileTypes.contains(.mov) ? .mov : exporter.supportedFileTypes.first ?? .mov
        do {
            try await exporter.export(to: outputURL, as: outputType)
        } catch {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.writer("720p input normalization failed: (error.localizedDescription)")
        }

        let normalized = AVURLAsset(url: outputURL)
        guard let normalizedTrack = try await normalized.loadTracks(withMediaType: .video).first else {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.conversionFailed("720p normalization produced no video track")
        }

        let normalizedSize = try await normalizedTrack.load(.naturalSize)
        let normalizedWidth = Int(abs(normalizedSize.width).rounded())
        let normalizedHeight = Int(abs(normalizedSize.height).rounded())

        guard normalizedWidth == 1280 && normalizedHeight == 720 else {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.conversionFailed(
                "720p normalization produced (normalizedWidth)x(normalizedHeight) instead of 1280x720"
            )
        }

        DiagnosticsLogger.shared.log(
            "720p input normalization complete • working source=1280x720 • original remains untouched"
        )
        return outputURL
    }
}
