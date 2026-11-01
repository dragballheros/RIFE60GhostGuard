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
            return "An on-device anime segmenter masks the character. Vision is the fallback. White is edited; the rest of the frame is kept."
        case .background:
            return "The subject stays. The background is the edit region."
        case .wholeFrame:
            return "No mask. The prompt restyles the whole frame."
        case .drawManually:
            return "Tap the character on any frame, or brush a few strokes. That one frame is tracked through the clip on device and re-anchored at each shot cut."
        }
    }
}

enum VACEMaskPreviewLabel {
    static func displayName(for method: String) -> String {
        switch method {
        case "anime-isnet": return "anime-segmenter"
        case "tap-select": return "tap-select"
        case "manual-tracked", "draw-manually": return "manual-tracked"
        case "vision-person", "vision-foreground": return "vision-fallback"
        default: return method
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

struct VACESegmentationDecision {
    var mask: VACEMaskPlane
    var foundSubject: Bool
    var method: String
    var coverage: Double
}

/// On-device subject mask. Anime ISNet is the primary path. Vision person and
/// foreground masks are fallbacks. The result stays at the source frame size.
enum VACESubjectSegmenter {
    static let minimumCoverage = 0.03

    static func editMask(cgImage: CGImage, mode: VACEMaskMode) throws -> VACESegmentationDecision {
        let width = cgImage.width
        let height = cgImage.height
        if mode == .wholeFrame {
            let plane = try solid(width: width, height: height, value: 255)
            return VACESegmentationDecision(mask: plane, foundSubject: true, method: "whole-frame", coverage: 1)
        }
        let chosen = try bestSubjectMask(cgImage: cgImage)
        guard var refined = chosen else {
            return VACESegmentationDecision(
                mask: try solid(width: width, height: height, value: 0),
                foundSubject: false,
                method: "none",
                coverage: 0
            )
        }
        if mode == .background {
            for index in refined.pixels.indices { refined.pixels[index] = 255 &- refined.pixels[index] }
            refined.coverage = coverage(of: refined.pixels)
            if refined.coverage < minimumCoverage {
                return VACESegmentationDecision(
                    mask: try solid(width: width, height: height, value: 0),
                    foundSubject: false,
                    method: refined.method,
                    coverage: refined.coverage
                )
            }
        }
        let plane = try VACEMaskPlane(width: width, height: height, pixels: refined.pixels)
        CrashCapture.step(String(format: "VACE mask method %@ coverage %.1f%%", refined.method, refined.coverage * 100))
        return VACESegmentationDecision(mask: plane, foundSubject: true, method: refined.method, coverage: refined.coverage)
    }

    static func editMask(frame: VACERGBAFrame, mode: VACEMaskMode) throws -> VACESegmentationDecision {
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

    private struct RefinedMask {
        var pixels: [UInt8]
        var method: String
        var coverage: Double
    }

    /// Keeps the connected region under a tap. A close-up that fills most of the frame stays.
    static func tapSelect(cgImage: CGImage, x: Int, y: Int) throws -> VACESegmentationDecision {
        let width = cgImage.width
        let height = cgImage.height
        let clampedX = min(max(0, x), width - 1)
        let clampedY = min(max(0, y), height - 1)
        if let confidence = try VACEAnimeSegmenter.confidence(cgImage: cgImage),
           let refined = refineComponent(confidence, width: width, height: height, x: clampedX, y: clampedY, method: "tap-select") {
            let plane = try VACEMaskPlane(width: width, height: height, pixels: refined.pixels)
            CrashCapture.step(String(format: "VACE tap-select anime-segmenter coverage %.1f%%", refined.coverage * 100))
            return VACESegmentationDecision(mask: plane, foundSubject: true, method: "tap-select", coverage: refined.coverage)
        }
        let person = try self.confidence(cgImage: cgImage, request: personRequest())
        if let refined = refineComponent(person, width: width, height: height, x: clampedX, y: clampedY, method: "tap-select") {
            let plane = try VACEMaskPlane(width: width, height: height, pixels: refined.pixels)
            return VACESegmentationDecision(mask: plane, foundSubject: true, method: "tap-select", coverage: refined.coverage)
        }
        return VACESegmentationDecision(
            mask: try solid(width: width, height: height, value: 0),
            foundSubject: false,
            method: "tap-select",
            coverage: 0
        )
    }

    static func frameImage(url: URL, at seconds: Double) async throws -> CGImage {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let time = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        let result = try await generator.image(at: time)
        return result.image
    }

    static func paint(_ mask: VACEMaskPlane, x: Int, y: Int, radius: Int, erase: Bool) -> VACEMaskPlane {
        var pixels = mask.pixels
        let value: UInt8 = erase ? 0 : 255
        let r = max(1, radius)
        for dy in -r...r {
            for dx in -r...r where dx * dx + dy * dy <= r * r {
                let px = x + dx
                let py = y + dy
                guard px >= 0, py >= 0, px < mask.width, py < mask.height else { continue }
                pixels[py * mask.width + px] = value
            }
        }
        return (try? VACEMaskPlane(width: mask.width, height: mask.height, pixels: pixels)) ?? mask
    }

    private static func bestSubjectMask(cgImage: CGImage) throws -> RefinedMask? {
        do {
            if let confidence = try VACEAnimeSegmenter.confidence(cgImage: cgImage) {
                let raw = coverage(of: threshold(confidence, above: 128))
                CrashCapture.step(String(format: "VACE anime frame coverage before cleanup %.1f%%", raw * 100))
                if let refined = refine(confidence, width: cgImage.width, height: cgImage.height, method: "anime-isnet") {
                    CrashCapture.step(String(format: "VACE preferred anime-segmenter coverage %.1f%%", refined.coverage * 100))
                    return refined
                }
                CrashCapture.step("VACE anime segmenter ran but the mask was below 3% after cleanup")
            } else {
                CrashCapture.step("VACE anime segmenter unavailable; using Vision")
            }
        } catch {
            CrashCapture.step("VACE segmentation failed: anime-isnet \(error.localizedDescription)")
        }
        return try visionFallback(cgImage: cgImage)
    }

    private static func visionFallback(cgImage: CGImage) throws -> RefinedMask? {
        var rejected: [String] = []
        do {
            let person = try self.confidence(cgImage: cgImage, request: personRequest())
            if let refined = refine(person, width: cgImage.width, height: cgImage.height, method: "vision-person", allowBackgroundLike: false) {
                return refined
            }
            rejected.append("vision-person")
        } catch {
            CrashCapture.step("VACE segmentation failed: vision-person \(error.localizedDescription)")
        }
        if #available(iOS 17.0, *) {
            do {
                if let foreground = try foregroundConfidence(cgImage: cgImage),
                   let refined = refine(foreground, width: cgImage.width, height: cgImage.height, method: "vision-foreground", allowBackgroundLike: false) {
                    return refined
                }
                rejected.append("vision-foreground")
            } catch {
                CrashCapture.step("VACE segmentation failed: vision-foreground \(error.localizedDescription)")
            }
        }
        if !rejected.isEmpty {
            CrashCapture.step("VACE rejected Vision mask as background-like: \(rejected.joined(separator: ", "))")
        }
        return nil
    }

    private static func refine(_ confidence: [UInt8], width: Int, height: Int, method: String, allowBackgroundLike: Bool = true) -> RefinedMask? {
        guard confidence.count == width * height else { return nil }
        var binary = orientForeground(threshold(confidence, above: 128), width: width, height: height)
        binary = largestComponents(binary, width: width, height: height)
        binary = fillHoles(binary, width: width, height: height)
        let fraction = coverage(of: binary)
        guard fraction >= minimumCoverage, fraction <= 0.98 else { return nil }
        if !allowBackgroundLike && isBackgroundLike(binary, width: width, height: height) {
            let inverted = orientForeground(binary.map { 255 &- $0 }, width: width, height: height)
            let invertedCoverage = coverage(of: inverted)
            if invertedCoverage >= minimumCoverage && invertedCoverage <= 0.98 && !isBackgroundLike(inverted, width: width, height: height) {
                binary = largestComponents(inverted, width: width, height: height)
            } else {
                return nil
            }
        }
        let kept = coverage(of: binary)
        guard kept >= minimumCoverage else { return nil }
        let radius = max(2, min(width, height) / 180)
        let soft = feather(dilate(binary, width: width, height: height, radius: radius), width: width, height: height, radius: max(2, radius / 2))
        return RefinedMask(pixels: soft, method: method, coverage: kept)
    }

    private static func refineComponent(_ confidence: [UInt8], width: Int, height: Int, x: Int, y: Int, method: String) -> RefinedMask? {
        guard confidence.count == width * height else { return nil }
        let binary = orientForeground(threshold(confidence, above: 128), width: width, height: height)
        guard var component = component(containing: x, y: y, in: binary, width: width, height: height) else { return nil }
        component = fillHoles(component, width: width, height: height)
        let fraction = coverage(of: component)
        guard fraction >= minimumCoverage, fraction <= 0.98 else { return nil }
        let radius = max(2, min(width, height) / 180)
        let soft = feather(dilate(component, width: width, height: height, radius: radius), width: width, height: height, radius: max(2, radius / 2))
        return RefinedMask(pixels: soft, method: method, coverage: fraction)
    }

    /// A mask that hugs the frame border and leaves the centre empty is background.
    /// A body close-up at 60–95% covers the centre, so it stays.
    static func isBackgroundLike(_ pixels: [UInt8], width: Int, height: Int) -> Bool {
        guard width > 8, height > 8, pixels.count == width * height else { return false }
        let marginX = max(1, width / 12)
        let marginY = max(1, height / 12)
        var border = 0
        var center = 0
        var total = 0
        for y in 0..<height {
            for x in 0..<width {
                guard pixels[y * width + x] > 127 else { continue }
                total += 1
                if x < marginX || y < marginY || x >= width - marginX || y >= height - marginY { border += 1 }
                if x > width / 4 && x < (width * 3) / 4 && y > height / 4 && y < (height * 3) / 4 { center += 1 }
            }
        }
        guard total > 0 else { return true }
        let coverage = Double(total) / Double(pixels.count)
        let centerShare = Double(center) / Double(total)
        if coverage >= 0.55 && centerShare >= 0.2 { return false }
        return centerShare < 0.2 && Double(border) / Double(total) > 0.5
    }

    /// Flips a mask whose border is lit and whose centre is empty.
    private static func orientForeground(_ binary: [UInt8], width: Int, height: Int) -> [UInt8] {
        guard isBackgroundLike(binary, width: width, height: height) else { return binary }
        let inverted = binary.map { 255 &- $0 }
        if isBackgroundLike(inverted, width: width, height: height) { return binary }
        CrashCapture.step("VACE inverted a border-heavy mask so the centre is the subject")
        return inverted
    }

    private static func component(containing x: Int, y: Int, in binary: [UInt8], width: Int, height: Int) -> [UInt8]? {
        let radius = max(8, min(width, height) / 25)
        var seedX = x
        var seedY = y
        var found = binary[y * width + x] > 127
        if !found {
            var best = Int.max
            for dy in -radius...radius {
                for dx in -radius...radius where dx * dx + dy * dy <= radius * radius {
                    let px = x + dx
                    let py = y + dy
                    guard px >= 0, py >= 0, px < width, py < height, binary[py * width + px] > 127 else { continue }
                    let distance = dx * dx + dy * dy
                    if distance < best {
                        best = distance
                        seedX = px
                        seedY = py
                        found = true
                    }
                }
            }
        }
        guard found else { return nil }
        var labels = [UInt8](repeating: 0, count: binary.count)
        var stack = [(seedX, seedY)]
        labels[seedY * width + seedX] = 255
        var head = 0
        while head < stack.count {
            let (px, py) = stack[head]
            head += 1
            for (nx, ny) in [(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)] {
                guard nx >= 0, ny >= 0, nx < width, ny < height else { continue }
                let index = ny * width + nx
                guard binary[index] > 127, labels[index] == 0 else { continue }
                labels[index] = 255
                stack.append((nx, ny))
            }
        }
        return labels
    }

    static func coverage(of pixels: [UInt8]) -> Double {
        guard !pixels.isEmpty else { return 0 }
        var active = 0
        for pixel in pixels where pixel > 127 { active += 1 }
        return Double(active) / Double(pixels.count)
    }

    private static func largestComponents(_ pixels: [UInt8], width: Int, height: Int) -> [UInt8] {
        let count = pixels.count
        var labels = [Int](repeating: 0, count: count)
        var parent: [Int] = [0]
        func find(_ label: Int) -> Int {
            var label = label
            while parent[label] != label {
                parent[label] = parent[parent[label]]
                label = parent[label]
            }
            return label
        }
        func unite(_ left: Int, _ right: Int) {
            let a = find(left)
            let b = find(right)
            if a != b { parent[b] = a }
        }
        var nextLabel = 1
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard pixels[index] > 127 else { continue }
                let left = x > 0 ? labels[index - 1] : 0
                let up = y > 0 ? labels[index - width] : 0
                if left == 0 && up == 0 {
                    parent.append(nextLabel)
                    labels[index] = nextLabel
                    nextLabel += 1
                } else if left != 0 && up != 0 {
                    labels[index] = left
                    unite(left, up)
                } else {
                    labels[index] = left != 0 ? left : up
                }
            }
        }
        var area = [Int](repeating: 0, count: nextLabel)
        for index in 0..<count where labels[index] != 0 {
            area[find(labels[index])] += 1
        }
        let largest = area.max() ?? 0
        guard largest > 0 else { return [UInt8](repeating: 0, count: count) }
        let minimum = max(1, largest / 5)
        var kept = [UInt8](repeating: 0, count: count)
        for index in 0..<count where labels[index] != 0 && area[find(labels[index])] >= minimum {
            kept[index] = 255
        }
        return kept
    }

    private static func fillHoles(_ pixels: [UInt8], width: Int, height: Int) -> [UInt8] {
        var outside = [UInt8](repeating: 0, count: pixels.count)
        var stack: [Int] = []
        func push(_ index: Int) {
            guard pixels[index] <= 127, outside[index] == 0 else { return }
            outside[index] = 1
            stack.append(index)
        }
        for x in 0..<width {
            push(x)
            push((height - 1) * width + x)
        }
        for y in 0..<height {
            push(y * width)
            push(y * width + width - 1)
        }
        while let index = stack.popLast() {
            let x = index % width
            let y = index / width
            if x > 0 { push(index - 1) }
            if x + 1 < width { push(index + 1) }
            if y > 0 { push(index - width) }
            if y + 1 < height { push(index + width) }
        }
        var filled = pixels
        for index in filled.indices where filled[index] <= 127 && outside[index] == 0 {
            filled[index] = 255
        }
        return filled
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

    static func cgImage(from frame: VACERGBAFrame) -> CGImage? {
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

/// Per-frame on-device mask. Tracking and temporal smoothing stay inside one
/// shot. A hard cut clears the previous mask and segments that frame on its own.
final class VACEAutoMaskProvider: VACEFrameMaskProvider {
    let mode: VACEMaskMode
    private var previous: VACEMaskPlane?
    private var previousCoverage = 0.0
    private var previousSignature: VACEFrameSignature?
    private(set) var fellBackToWholeFrame = false

    init(mode: VACEMaskMode) {
        self.mode = mode
    }

    func mask(for frameIndex: Int, presentationTime: CMTime, source: VACERGBAFrame) async throws -> VACEMaskPlane {
        if mode == .wholeFrame {
            return try wholeFrame(width: source.width, height: source.height)
        }
        let signature = VACEShotCut.signature(of: source)
        var newShot = previousSignature == nil
        if let previousSignature {
            let comparison = VACEShotCut.isCut(from: previousSignature, to: signature)
            if comparison.cut {
                newShot = true
                previous = nil
                previousCoverage = 0
                VACEShotCut.logCut(at: presentationTime, histogramDistance: comparison.histogramDistance, frameDifference: comparison.frameDifference)
            }
        }
        self.previousSignature = signature
        do {
            let result = try VACESubjectSegmenter.editMask(frame: source, mode: mode)
            if result.foundSubject {
                let dropped = !newShot && previous != nil && previousCoverage >= VACESubjectSegmenter.minimumCoverage
                    && result.coverage < previousCoverage * 0.5
                if dropped, let previous {
                    CrashCapture.step(String(format: "VACE mask method previous-track coverage %.1f%%", previousCoverage * 100))
                    return previous
                }
                let mask = try smoothed(result.mask, newShot: newShot)
                previous = mask
                previousCoverage = VACESubjectSegmenter.coverage(of: mask.pixels)
                return mask
            }
        } catch {
            CrashCapture.step("VACE segmentation failed: \(error.localizedDescription)")
        }
        if !newShot, let previous {
            CrashCapture.step(String(format: "VACE mask method previous-track coverage %.1f%%", previousCoverage * 100))
            return previous
        }
        fellBackToWholeFrame = true
        CrashCapture.step("VACE segmentation found nothing; using whole frame")
        let filled = try wholeFrame(width: source.width, height: source.height)
        previous = filled
        previousCoverage = 1
        return filled
    }

    private func smoothed(_ mask: VACEMaskPlane, newShot: Bool) throws -> VACEMaskPlane {
        guard !newShot, let previous, previous.width == mask.width, previous.height == mask.height else { return mask }
        var pixels = mask.pixels
        for index in pixels.indices {
            pixels[index] = UInt8((Int(pixels[index]) * 2 + Int(previous.pixels[index])) / 3)
        }
        return try VACEMaskPlane(width: mask.width, height: mask.height, pixels: pixels)
    }

    private func wholeFrame(width: Int, height: Int) throws -> VACEMaskPlane {
        try VACEMaskPlane(width: width, height: height, pixels: [UInt8](repeating: 255, count: width * height))
    }
}
