import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import UIKit
import Vision

enum VACEMaskMode: String, CaseIterable, Identifiable {
    case autoSubject = "Auto (subject)"
    case background = "Background"
    case wholeFrame = "Whole frame"
    case drawManually = "Draw manually"

    var id: String { rawValue }

    var needsHandDrawnMask: Bool { self == .drawManually }

    var caption: String {
        switch self {
        case .autoSubject:
            return "People and other foreground subjects are masked on this iPhone. White is edited; the rest of the frame is kept."
        case .background:
            return "The subject stays. The background is the edit region."
        case .wholeFrame:
            return "No mask. The prompt restyles the whole frame."
        case .drawManually:
            return "Optional. White in the image is edited; black is kept. Optical flow tracks that mask."
        }
    }
}

enum VACESubjectSegmentationError: LocalizedError {
    case unreadableFrame
    case visionFailed(String)

    var errorDescription: String? {
        switch self {
        case .unreadableFrame: return "The video frame could not be prepared for on-device segmentation."
        case .visionFailed(let message): return message
        }
    }
}

/// On-device subject mask. Person segmentation first, then the iOS 17 foreground
/// instance mask for subjects Vision does not label as people. The result stays
/// at the source frame size.
enum VACESubjectSegmenter {
    private static let minimumSubjectPixels = 48

    static func editMask(cgImage: CGImage, mode: VACEMaskMode) throws -> (mask: VACEMaskPlane, foundSubject: Bool) {
        let width = cgImage.width
        let height = cgImage.height
        if mode == .wholeFrame {
            return (try solid(width: width, height: height, value: 255), true)
        }
        let subject = try subjectConfidence(cgImage: cgImage)
        let binary = threshold(subject, above: 128)
        let found = binary.reduce(into: 0) { if $1 == 255 { $0 += 1 } } >= minimumSubjectPixels
        guard found else {
            return (try solid(width: width, height: height, value: 0), false)
        }
        let radius = max(2, min(width, height) / 180)
        var soft = feather(dilate(binary, width: width, height: height, radius: radius), width: width, height: height, radius: max(2, radius / 2))
        if mode == .background {
            for index in soft.indices { soft[index] = 255 &- soft[index] }
        }
        let plane = try VACEMaskPlane(width: width, height: height, pixels: soft)
        let editable = plane.pixels.reduce(into: 0) { if $1 > 16 { $0 += 1 } }
        if editable < minimumSubjectPixels {
            return (try solid(width: width, height: height, value: 0), false)
        }
        return (plane, true)
    }

    static func editMask(frame: VACERGBAFrame, mode: VACEMaskMode) throws -> (mask: VACEMaskPlane, foundSubject: Bool) {
        guard let image = cgImage(from: frame) else { throw VACESubjectSegmentationError.unreadableFrame }
        return try editMask(cgImage: image, mode: mode)
    }

    static func previewImage(base: CGImage, mask: VACEMaskPlane) -> UIImage? {
        let width = base.width
        let height = base.height
        guard mask.width == width, mask.height == height,
              let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else { return nil }
        context.draw(base, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { return nil }
        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for index in 0..<(width * height) {
            let coverage = Int(mask.pixels[index])
            guard coverage > 0 else { continue }
            let offset = index * 4
            let keep = 255 - coverage
            pixels[offset] = UInt8(min(255, (Int(pixels[offset]) * keep + 255 * coverage) / 255))
            pixels[offset + 1] = UInt8((Int(pixels[offset + 1]) * keep) / 255)
            pixels[offset + 2] = UInt8((Int(pixels[offset + 2]) * keep) / 255)
        }
        guard let composited = context.makeImage() else { return nil }
        return UIImage(cgImage: composited)
    }

    static func firstFrameImage(url: URL) async throws -> CGImage {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let result = try await generator.image(at: .zero)
        return result.image
    }

    private static func subjectConfidence(cgImage: CGImage) throws -> [UInt8] {
        let width = cgImage.width
        let height = cgImage.height
        do {
            let person = try confidence(cgImage: cgImage, request: personRequest())
            let strong = person.reduce(into: 0) { if $1 > 128 { $0 += 1 } }
            if strong >= minimumSubjectPixels { return person }
        } catch {
            CrashCapture.step("VACE segmentation failed: \(error.localizedDescription)")
        }
        if #available(iOS 17.0, *) {
            do {
                if let foreground = try foregroundConfidence(cgImage: cgImage) {
                    let strong = foreground.reduce(into: 0) { if $1 > 128 { $0 += 1 } }
                    if strong >= minimumSubjectPixels { return foreground }
                }
            } catch {
                CrashCapture.step("VACE segmentation failed: \(error.localizedDescription)")
            }
        }
        return [UInt8](repeating: 0, count: width * height)
    }

    private static func personRequest() -> VNGeneratePersonSegmentationRequest {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return request
    }

    private static func confidence(cgImage: CGImage, request: VNGeneratePersonSegmentationRequest) throws -> [UInt8] {
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])
        guard let buffer = request.results?.first?.pixelBuffer else {
            throw VACESubjectSegmentationError.visionFailed("Person segmentation returned no mask.")
        }
        return try scaledGray(buffer, width: cgImage.width, height: cgImage.height)
    }

    @available(iOS 17.0, *)
    private static func foregroundConfidence(cgImage: CGImage) throws -> [UInt8]? {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])
        guard let observation = request.results?.first else { return nil }
        let instances = observation.allInstances
        guard !instances.isEmpty else { return nil }
        let buffer = try observation.generateScaledMaskForImage(forInstances: instances, from: handler)
        return try scaledGray(buffer, width: cgImage.width, height: cgImage.height)
    }

    private static func scaledGray(_ buffer: CVPixelBuffer, width: Int, height: Int) throws -> [UInt8] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let sourceWidth = CVPixelBufferGetWidth(buffer)
        let sourceHeight = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer), sourceWidth > 0, sourceHeight > 0 else {
            throw VACESubjectSegmentationError.visionFailed("Segmentation mask was empty.")
        }
        var packed = [UInt8](repeating: 0, count: sourceWidth * sourceHeight)
        let source = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<sourceHeight {
            let row = source.advanced(by: y * rowBytes)
            packed.replaceSubrange((y * sourceWidth)..<((y + 1) * sourceWidth), with: UnsafeBufferPointer(start: row, count: sourceWidth))
        }
        if sourceWidth == width && sourceHeight == height { return packed }
        guard let provider = CGDataProvider(data: Data(packed) as CFData),
              let image = CGImage(
                width: sourceWidth, height: sourceHeight,
                bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: sourceWidth,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
              ),
              let context = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue
              ) else {
            throw VACESubjectSegmentationError.visionFailed("Could not scale the segmentation mask to the frame.")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let scaled = context.data else {
            throw VACESubjectSegmentationError.visionFailed("Could not read the scaled segmentation mask.")
        }
        return Array(UnsafeBufferPointer(start: scaled.bindMemory(to: UInt8.self, capacity: width * height), count: width * height))
    }

    private static func threshold(_ pixels: [UInt8], above: UInt8) -> [UInt8] {
        pixels.map { $0 > above ? 255 : 0 }
    }

    private static func dilate(_ pixels: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0 else { return pixels }
        var horizontal = [UInt8](repeating: 0, count: pixels.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                var value: UInt8 = 0
                let start = max(0, x - radius)
                let end = min(width - 1, x + radius)
                for sample in start...end {
                    value = max(value, pixels[row + sample])
                }
                horizontal[row + x] = value
            }
        }
        var vertical = [UInt8](repeating: 0, count: pixels.count)
        for x in 0..<width {
            for y in 0..<height {
                var value: UInt8 = 0
                let start = max(0, y - radius)
                let end = min(height - 1, y + radius)
                for sample in start...end {
                    value = max(value, horizontal[sample * width + x])
                }
                vertical[y * width + x] = value
            }
        }
        return vertical
    }

    private static func feather(_ pixels: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0 else { return pixels }
        let window = radius * 2 + 1
        var horizontal = [UInt8](repeating: 0, count: pixels.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                var sum = 0
                for sample in (x - radius)...(x + radius) {
                    sum += Int(pixels[row + min(width - 1, max(0, sample))])
                }
                horizontal[row + x] = UInt8(sum / window)
            }
        }
        var vertical = [UInt8](repeating: 0, count: pixels.count)
        for x in 0..<width {
            for y in 0..<height {
                var sum = 0
                for sample in (y - radius)...(y + radius) {
                    sum += Int(horizontal[min(height - 1, max(0, sample)) * width + x])
                }
                vertical[y * width + x] = UInt8(sum / window)
            }
        }
        return vertical
    }

    private static func solid(width: Int, height: Int, value: UInt8) throws -> VACEMaskPlane {
        try VACEMaskPlane(width: width, height: height, pixels: [UInt8](repeating: value, count: width * height))
    }

    private static func cgImage(from frame: VACERGBAFrame) -> CGImage? {
        let data = Data(frame.bytes) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(
            width: frame.width, height: frame.height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: frame.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}

/// Per-frame on-device mask. A frame with no subject keeps the previous mask so
/// the edit region stays tracked. If the first frame has no subject, every frame
/// uses the whole frame.
final class VACEAutoMaskProvider: VACEFrameMaskProvider {
    let mode: VACEMaskMode
    private var previous: VACEMaskPlane?
    private var useWholeFrame = false
    private(set) var fellBackToWholeFrame = false
    private var loggedFailure = false

    init(mode: VACEMaskMode) {
        self.mode = mode
    }

    func mask(for frameIndex: Int, presentationTime: CMTime, source: VACERGBAFrame) async throws -> VACEMaskPlane {
        _ = presentationTime
        if mode == .wholeFrame || useWholeFrame {
            return try VACEMaskPlane(width: source.width, height: source.height, pixels: [UInt8](repeating: 255, count: source.width * source.height))
        }
        do {
            let result = try VACESubjectSegmenter.editMask(frame: source, mode: mode)
            if result.foundSubject, result.mask.activePixelCount > 0 {
                previous = result.mask
                return result.mask
            }
            if let previous { return previous }
            useWholeFrame = true
            fellBackToWholeFrame = true
            CrashCapture.step("VACE segmentation found nothing; using whole frame")
            return try VACEMaskPlane(width: source.width, height: source.height, pixels: [UInt8](repeating: 255, count: source.width * source.height))
        } catch {
            if !loggedFailure {
                loggedFailure = true
                CrashCapture.step("VACE segmentation failed: \(error.localizedDescription)")
            }
            if let previous { return previous }
            useWholeFrame = true
            fellBackToWholeFrame = true
            return try VACEMaskPlane(width: source.width, height: source.height, pixels: [UInt8](repeating: 255, count: source.width * source.height))
        }
    }
}
