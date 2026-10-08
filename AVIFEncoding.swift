import Foundation
import CoreGraphics
import UIKit
import avif

enum AVIFEncodingError: LocalizedError {
    case emptyOutput
    case invalidContainer
    case encoderFailed(String)

    var errorDescription: String? {
        switch self {
        case .emptyOutput:
            return "The AVIF encoder returned an empty file."
        case .invalidContainer:
            return "The AVIF encoder returned data without a valid AVIF container signature."
        case .encoderFailed(let message):
            return "The AVIF encoder failed: \(message)"
        }
    }
}

/// Serializes access to the native AVIF encoder. The encoder is shared by regular
/// image export and Reddit delivery, so concurrent native encoder calls are avoided.
final class AVIFEncoderGate: @unchecked Sendable {
    static let shared = AVIFEncoderGate()

    private let lock = NSLock()

    private init() {}

    func encode(_ image: CGImage, quality: Double) throws -> Data {
        lock.lock()
        defer { lock.unlock() }

        var lastError: Error?
        let qualities = [quality, max(0, quality - 5), max(0, quality - 10)]

        for candidateQuality in qualities {
            do {
                let data = try autoreleasepool {
                    try AVIFEncoder.encode(
                        image: UIImage(cgImage: image),
                        quality: candidateQuality
                    )
                }

                guard !data.isEmpty else {
                    lastError = AVIFEncodingError.emptyOutput
                    continue
                }

                guard Self.hasAVIFSignature(data) else {
                    lastError = AVIFEncodingError.invalidContainer
                    continue
                }

                return data
            } catch {
                lastError = error
            }
        }

        throw AVIFEncodingError.encoderFailed(lastError?.localizedDescription ?? "unknown encoder error")
    }

    private static func hasAVIFSignature(_ data: Data) -> Bool {
        guard data.count >= 12 else { return false }

        // ISO-BMFF AVIF files contain an ftyp box at byte 4 and an avif/avis
        // compatible brand in the compatible-brand list.
        let bytes = [UInt8](data.prefix(min(data.count, 128)))
        guard bytes.count >= 12,
              bytes[4] == 0x66, bytes[5] == 0x74,
              bytes[6] == 0x79, bytes[7] == 0x70 else {
            return false
        }

        var offset = 8
        while offset + 4 <= bytes.count {
            let brand = bytes[offset...offset + 3]
            if brand == [0x61, 0x76, 0x69, 0x66] || brand == [0x61, 0x76, 0x69, 0x73] {
                return true
            }
            offset += 4
        }
        return false
    }
}
