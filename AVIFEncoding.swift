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

        let pixelCount = image.width * image.height
        // avif.swift exposes libaom's speed control. Omitting it selects the codec
        // default, which can make high-quality 4K-class stills appear to hang on-device.
        // Speed 6 is substantially more practical on iPhone while retaining the
        // requested quality setting.
        let speed: Int
        if pixelCount >= 24_000_000 {
            speed = 7
        } else if pixelCount >= 8_000_000 {
            speed = 6
        } else {
            speed = 5
        }

        DiagnosticsLogger.shared.log(
            "AVIF encode begin • \(image.width)x\(image.height) • pixels=\(pixelCount) • quality=\(quality) • speed=\(speed)"
        )

        do {
            let data = try autoreleasepool {
                try AVIFEncoder.encode(
                    image: UIImage(cgImage: image),
                    quality: quality,
                    speed: speed,
                    preferredCodec: .AOM
                )
            }

            guard !data.isEmpty else {
                throw AVIFEncodingError.emptyOutput
            }

            guard Self.hasAVIFSignature(data) else {
                throw AVIFEncodingError.invalidContainer
            }

            DiagnosticsLogger.shared.log(
                "AVIF encode complete • \(image.width)x\(image.height) • bytes=\(data.count) • quality=\(quality) • speed=\(speed)"
            )
            return data
        } catch {
            DiagnosticsLogger.shared.log(
                "AVIF encode failed • \(image.width)x\(image.height) • quality=\(quality) • speed=\(speed) • error=\(error.localizedDescription)"
            )
            throw AVIFEncodingError.encoderFailed(error.localizedDescription)
        }
    }

    static func isAVIFData(_ data: Data) -> Bool {
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
