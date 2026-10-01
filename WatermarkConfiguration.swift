import Foundation
import CoreGraphics

/// A rectangle in the upright preview: normalized 0...1, origin at top-left.
struct WatermarkRegion: Codable, Sendable, Equatable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double

    var isValid: Bool {
        x.isFinite && y.isFinite && width.isFinite && height.isFinite &&
        x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1.000001 && y + height <= 1.000001
    }
}

struct WatermarkConfiguration: Sendable {
    var enabled = false
    var regions: [WatermarkRegion] = []
    var paddingPixels = 3
    static let modelID = "anime-lama-512-fp32-v1"

    var isValid: Bool { !enabled || (!regions.isEmpty && regions.count <= 8 && regions.allSatisfy { $0.isValid }) }

    var serializedRegions: String {
        // Fixed order and Double precision preserve masks when a job resumes.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return (try? encoder.encode(regions).base64EncodedString()) ?? "W10="
    }

    static func decodeRegions(_ encoded: String) -> [WatermarkRegion] {
        guard let data = Data(base64Encoded: encoded),
              let regions = try? JSONDecoder().decode([WatermarkRegion].self, from: data),
              regions.count <= 8, regions.allSatisfy({ $0.isValid }) else { return [] }
        return regions
    }

    /// AVAssetImageGenerator shows the preferred-transform preview. Decode frames
    /// remain in encoded orientation, so map the selected rectangles back once.
    func inEncodedOrientation(size: CGSize, transform: CGAffineTransform) -> WatermarkConfiguration {
        guard enabled else { return self }
        let rawBounds = CGRect(origin: .zero, size: size)
        let upright = rawBounds.applying(transform).standardized
        let inverse = transform.inverted()
        var result = self
        result.regions = regions.compactMap { region in
            let selected = CGRect(x: upright.minX + region.x * upright.width,
                                  y: upright.minY + region.y * upright.height,
                                  width: region.width * upright.width, height: region.height * upright.height)
            let raw = selected.applying(inverse).standardized.intersection(rawBounds)
            guard !raw.isNull, raw.width > 0, raw.height > 0, size.width > 0, size.height > 0 else { return nil }
            return WatermarkRegion(x: raw.minX / size.width, y: raw.minY / size.height,
                                   width: raw.width / size.width, height: raw.height / size.height)
        }
        return result
    }
}
