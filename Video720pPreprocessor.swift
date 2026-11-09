import AVFoundation

/// Normalizes every processing input to a 1080p working raster before any
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
            throw ProcessorError.conversionFailed("1080p preprocessor received an invalid video size")
        }

        let longSide = max(width, height)
        let shortSide = min(width, height)
        if longSide == 1920 && shortSide == 1080 {
            DiagnosticsLogger.shared.log("1080p input normalization • source already 1080p • no resize required")
            return sourceURL
        }

        guard let exporter = AVAssetExportSession(
            asset: asset,
            presetName: AVAssetExportPreset1920x1080
        ) else {
            throw ProcessorError.conversionFailed("could not create 1080p export session")
        }

        let outputURL = fm.temporaryDirectory
            .appendingPathComponent("RIFE60-1080p-working-" + UUID().uuidString)
            .appendingPathExtension("mov")
        try? fm.removeItem(at: outputURL)

        exporter.shouldOptimizeForNetworkUse = false

        DiagnosticsLogger.shared.log(
            "1080p input normalization • \(width)x\(height) → 1080p working target • no final bitrate target applied"
        )

        let outputType: AVFileType = exporter.supportedFileTypes.contains(.mov)
            ? .mov
            : (exporter.supportedFileTypes.first ?? .mov)

        exporter.outputURL = outputURL
        exporter.outputFileType = outputType

        do {
            // Use the iOS 16-compatible export API. The async
            // export(to:as:) API is only back-deployed from iOS 26 and
            // therefore must not be referenced directly by this iOS 16+
            // target's deployment path.
            try await withCheckedThrowingContinuation { continuation in
                exporter.exportAsynchronously {
                    switch exporter.status {
                    case .completed:
                        continuation.resume(returning: ())
                    case .cancelled:
                        continuation.resume(throwing: CancellationError())
                    case .failed, .unknown, .waiting, .exporting:
                        continuation.resume(
                            throwing: ProcessorError.writer(
                                "1080p input normalization failed: \(exporter.error?.localizedDescription ?? "export did not complete")"
                            )
                        )
                    @unknown default:
                        continuation.resume(
                            throwing: ProcessorError.writer(
                                "1080p input normalization failed with an unknown export status"
                            )
                        )
                    }
                }
            }
        } catch {
            try? fm.removeItem(at: outputURL)
            if error is CancellationError {
                throw error
            }
            if let processorError = error as? ProcessorError {
                throw processorError
            }
            throw ProcessorError.writer("1080p input normalization failed: \(error.localizedDescription)")
        }

        let normalized = AVURLAsset(url: outputURL)
        guard let normalizedTrack = try await normalized.loadTracks(withMediaType: .video).first else {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.conversionFailed("1080p normalization produced no video track")
        }

        let normalizedSize = try await normalizedTrack.load(.naturalSize)
        let normalizedWidth = Int(abs(normalizedSize.width).rounded())
        let normalizedHeight = Int(abs(normalizedSize.height).rounded())
        let normalizedLong = max(normalizedWidth, normalizedHeight)
        let normalizedShort = min(normalizedWidth, normalizedHeight)

        guard normalizedLong == 1920 && normalizedShort == 1080 else {
            try? fm.removeItem(at: outputURL)
            throw ProcessorError.conversionFailed(
                "1080p normalization produced \(normalizedWidth)x\(normalizedHeight) instead of a 1080p raster"
            )
        }

        DiagnosticsLogger.shared.log(
            "1080p input normalization complete • working source=\(normalizedWidth)x\(normalizedHeight) • original remains untouched"
        )
        return outputURL
    }
}
