import Foundation

/// Half-open integer rectangle in source-frame pixel coordinates.
struct VACERegionRect: Equatable, Hashable {
    let x: Int
    let y: Int
    let width: Int
    let height: Int

    init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x; self.y = y; self.width = width; self.height = height
    }
    var minX: Int { x }
    var minY: Int { y }
    var maxX: Int { x + width }
    var maxY: Int { y + height }
    var area: Int { width * height }
    var isEmpty: Bool { width <= 0 || height <= 0 }

    func intersects(_ other: VACERegionRect) -> Bool {
        !isEmpty && !other.isEmpty && minX < other.maxX && other.minX < maxX && minY < other.maxY && other.minY < maxY
    }
    func union(_ other: VACERegionRect) -> VACERegionRect {
        let left = min(minX, other.minX), top = min(minY, other.minY)
        let right = max(maxX, other.maxX), bottom = max(maxY, other.maxY)
        return VACERegionRect(x: left, y: top, width: right - left, height: bottom - top)
    }
    func intersection(_ other: VACERegionRect) -> VACERegionRect? {
        let left = max(minX, other.minX), top = max(minY, other.minY)
        let right = min(maxX, other.maxX), bottom = min(maxY, other.maxY)
        guard right > left, bottom > top else { return nil }
        return VACERegionRect(x: left, y: top, width: right - left, height: bottom - top)
    }
}

enum VACERegionPlanningError: LocalizedError {
    case invalidMaskDimensions
    case maskBufferSizeMismatch(expected: Int, actual: Int)
    case imageTooLarge(pixelCount: Int)
    case invalidOptions(String)
    case regionLimitExceeded(limit: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .invalidMaskDimensions: return "Mask width and height must be positive."
        case .maskBufferSizeMismatch(let expected, let actual): return "Buffer has \(actual) values; expected \(expected)."
        case .imageTooLarge(let count): return "Image contains \(count) pixels, above the safe planning limit."
        case .invalidOptions(let message): return "Invalid VACE planning options: \(message)"
        case .regionLimitExceeded(let limit, let actual): return "VACE mask expands to \(actual) crops, above the configured limit of \(limit)."
        }
    }
}

/// 8-bit row-major replacement mask. Zero means preserve; 255 means replace;
/// intermediate values provide feathered blending.
struct VACEMaskPlane {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) throws {
        guard width > 0, height > 0 else { throw VACERegionPlanningError.invalidMaskDimensions }
        let product = width.multipliedReportingOverflow(by: height)
        guard !product.overflow else { throw VACERegionPlanningError.imageTooLarge(pixelCount: Int.max) }
        guard product.partialValue <= 16_777_216 else {
            throw VACERegionPlanningError.imageTooLarge(pixelCount: product.partialValue)
        }
        guard pixels.count == product.partialValue else {
            throw VACERegionPlanningError.maskBufferSizeMismatch(expected: product.partialValue, actual: pixels.count)
        }
        self.width = width; self.height = height; self.pixels = pixels
    }

    var pixelCount: Int { width * height }
    var activePixelCount: Int { pixels.reduce(into: 0) { if $1 > 0 { $0 += 1 } } }
    subscript(x: Int, y: Int) -> UInt8 { pixels[y * width + x] }

    func cropped(to rect: VACERegionRect) throws -> VACEMaskPlane {
        let bounds = VACERegionRect(x: 0, y: 0, width: width, height: height)
        guard let crop = rect.intersection(bounds) else {
            throw VACERegionPlanningError.invalidOptions("Cannot crop a mask outside the frame.")
        }
        var result = [UInt8](repeating: 0, count: crop.width * crop.height)
        for row in 0..<crop.height {
            let sourceStart = (crop.y + row) * width + crop.x
            let destinationStart = row * crop.width
            result.replaceSubrange(destinationStart..<(destinationStart + crop.width),
                                   with: pixels[sourceStart..<(sourceStart + crop.width)])
        }
        return try VACEMaskPlane(width: crop.width, height: crop.height, pixels: result)
    }
}

struct VACERegionPlanningOptions {
    var contextPaddingPixels: Int = 32
    var cropAlignment: Int = 16
    var minimumActivePixels: Int = 4
    /// Maximum per-call crop dimensions; independent of final video dimensions.
    var maximumCropWidth: Int = 832
    var maximumCropHeight: Int = 480
    var tileOverlapPixels: Int = 64
    var maximumCropCount: Int = 32
    var foregroundThreshold: UInt8 = 0
}

struct VACERegionCropPlan: Equatable {
    let regionID: Int
    let cropRect: VACERegionRect
    let activeBounds: VACERegionRect
    /// Total pixels in the connected/merged component; repeated for its tiles.
    let componentPixelCount: Int
    let inferenceWidth: Int
    let inferenceHeight: Int
    var inferencePixelCount: Int { inferenceWidth * inferenceHeight }
}

private struct VACERegionCandidate {
    var activeBounds: VACERegionRect
    var cropBounds: VACERegionRect
    var activePixelCount: Int
}

/// Builds small, context-padded crop requests from a mask. It does not pretend to
/// be a denoiser: output from an actual VACE model is required before compositing.
enum VACESelectiveRegionPlanner {
    static func plan(mask: VACEMaskPlane,
                     options: VACERegionPlanningOptions = VACERegionPlanningOptions()) throws -> [VACERegionCropPlan] {
        try validate(options: options)
        let components = connectedComponents(in: mask, threshold: options.foregroundThreshold)
            .filter { $0.activePixelCount >= options.minimumActivePixels }
        guard !components.isEmpty else { return [] }

        var merged: [VACERegionCandidate] = []
        for component in components {
            let padded = alignedAndPadded(component.activeBounds, frameWidth: mask.width, frameHeight: mask.height,
                                          padding: options.contextPaddingPixels, alignment: options.cropAlignment)
            var candidate = VACERegionCandidate(activeBounds: component.activeBounds, cropBounds: padded,
                                                activePixelCount: component.activePixelCount)
            var index = 0
            while index < merged.count {
                if candidate.cropBounds.intersects(merged[index].cropBounds) {
                    let previous = merged.remove(at: index)
                    candidate.activeBounds = candidate.activeBounds.union(previous.activeBounds)
                    candidate.cropBounds = candidate.cropBounds.union(previous.cropBounds)
                    candidate.activePixelCount += previous.activePixelCount
                    index = 0
                } else { index += 1 }
            }
            merged.append(candidate)
        }

        var plans: [VACERegionCropPlan] = []
        for candidate in merged {
            let tiles = tile(candidate.cropBounds, maximumWidth: options.maximumCropWidth,
                             maximumHeight: options.maximumCropHeight, overlap: options.tileOverlapPixels)
            for tileRect in tiles {
                let clippedActive = candidate.activeBounds.intersection(tileRect) ?? candidate.activeBounds
                let canvas = inferenceCanvasSize(sourceWidth: tileRect.width, sourceHeight: tileRect.height,
                                                  maximumWidth: options.maximumCropWidth,
                                                  maximumHeight: options.maximumCropHeight,
                                                  alignment: options.cropAlignment)
                plans.append(VACERegionCropPlan(regionID: plans.count, cropRect: tileRect,
                                                activeBounds: clippedActive,
                                                componentPixelCount: candidate.activePixelCount,
                                                inferenceWidth: canvas.width, inferenceHeight: canvas.height))
                guard plans.count <= options.maximumCropCount else {
                    throw VACERegionPlanningError.regionLimitExceeded(limit: options.maximumCropCount, actual: plans.count)
                }
            }
        }
        return plans
    }

    private static func validate(options: VACERegionPlanningOptions) throws {
        guard options.contextPaddingPixels >= 0 else { throw VACERegionPlanningError.invalidOptions("padding cannot be negative") }
        guard options.cropAlignment > 0 else { throw VACERegionPlanningError.invalidOptions("alignment must be positive") }
        guard options.minimumActivePixels > 0 else { throw VACERegionPlanningError.invalidOptions("minimum area must be positive") }
        guard options.maximumCropWidth >= options.cropAlignment, options.maximumCropHeight >= options.cropAlignment else {
            throw VACERegionPlanningError.invalidOptions("maximum crop sizes must be at least one alignment unit")
        }
        guard options.tileOverlapPixels >= 0,
              options.tileOverlapPixels < min(options.maximumCropWidth, options.maximumCropHeight) else {
            throw VACERegionPlanningError.invalidOptions("overlap must be nonnegative and smaller than crop width and height")
        }
        guard options.maximumCropCount > 0 else { throw VACERegionPlanningError.invalidOptions("crop limit must be positive") }
    }

    private static func connectedComponents(in mask: VACEMaskPlane, threshold: UInt8) -> [VACERegionCandidate] {
        let width = mask.width, height = mask.height, count = mask.pixelCount
        var visited = [UInt8](repeating: 0, count: count)
        var queue: [Int] = []
        queue.reserveCapacity(min(count, 65_536))
        var components: [VACERegionCandidate] = []

        for start in 0..<count {
            if visited[start] != 0 || mask.pixels[start] <= threshold { continue }
            queue.removeAll(keepingCapacity: true)
            queue.append(start); visited[start] = 1
            var head = 0
            var minX = start % width, maxX = minX
            var minY = start / width, maxY = minY
            var area = 0
            while head < queue.count {
                let current = queue[head]; head += 1
                let x = current % width, y = current / width
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
                area += 1
                if x > 0 { appendIfActive(current - 1, mask: mask, threshold: threshold, visited: &visited, queue: &queue) }
                if x + 1 < width { appendIfActive(current + 1, mask: mask, threshold: threshold, visited: &visited, queue: &queue) }
                if y > 0 { appendIfActive(current - width, mask: mask, threshold: threshold, visited: &visited, queue: &queue) }
                if y + 1 < height { appendIfActive(current + width, mask: mask, threshold: threshold, visited: &visited, queue: &queue) }
            }
            components.append(VACERegionCandidate(
                activeBounds: VACERegionRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1),
                cropBounds: VACERegionRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1),
                activePixelCount: area))
        }
        return components
    }

    private static func appendIfActive(_ index: Int, mask: VACEMaskPlane, threshold: UInt8,
                                       visited: inout [UInt8], queue: inout [Int]) {
        guard visited[index] == 0, mask.pixels[index] > threshold else { return }
        visited[index] = 1; queue.append(index)
    }

    private static func alignedAndPadded(_ rect: VACERegionRect, frameWidth: Int, frameHeight: Int,
                                         padding: Int, alignment: Int) -> VACERegionRect {
        let left0 = max(0, rect.minX - padding), top0 = max(0, rect.minY - padding)
        let right0 = min(frameWidth, rect.maxX + padding), bottom0 = min(frameHeight, rect.maxY + padding)
        let left = (left0 / alignment) * alignment, top = (top0 / alignment) * alignment
        let right = min(frameWidth, ((right0 + alignment - 1) / alignment) * alignment)
        let bottom = min(frameHeight, ((bottom0 + alignment - 1) / alignment) * alignment)
        return VACERegionRect(x: left, y: top, width: right - left, height: bottom - top)
    }

    private static func tile(_ rect: VACERegionRect, maximumWidth: Int, maximumHeight: Int,
                             overlap: Int) -> [VACERegionRect] {
        let xs = tileStarts(from: rect.minX, to: rect.maxX, maximum: maximumWidth, overlap: overlap)
        let ys = tileStarts(from: rect.minY, to: rect.maxY, maximum: maximumHeight, overlap: overlap)
        var result: [VACERegionRect] = []
        result.reserveCapacity(xs.count * ys.count)
        for y in ys {
            for x in xs {
                result.append(VACERegionRect(x: x, y: y, width: min(maximumWidth, rect.maxX - x),
                                             height: min(maximumHeight, rect.maxY - y)))
            }
        }
        return result
    }

    private static func tileStarts(from start: Int, to end: Int, maximum: Int, overlap: Int) -> [Int] {
        guard end - start > maximum else { return [start] }
        let step = maximum - overlap
        var starts = [start]
        while true {
            let previous = starts[starts.count - 1]
            if previous + maximum >= end { break }
            let next = previous + step
            if next + maximum >= end {
                let finalStart = max(start, end - maximum)
                if finalStart > previous { starts.append(finalStart) }
                break
            }
            starts.append(next)
        }
        return starts
    }

    private static func inferenceCanvasSize(sourceWidth: Int, sourceHeight: Int, maximumWidth: Int,
                                             maximumHeight: Int, alignment: Int) -> (width: Int, height: Int) {
        let scale = min(Double(maximumWidth) / Double(sourceWidth), Double(maximumHeight) / Double(sourceHeight))
        let rawWidth = max(alignment, Int((Double(sourceWidth) * scale).rounded(.down)))
        let rawHeight = max(alignment, Int((Double(sourceHeight) * scale).rounded(.down)))
        return (max(alignment, min(maximumWidth, rawWidth / alignment * alignment)),
                max(alignment, min(maximumHeight, rawHeight / alignment * alignment)))
    }
}

struct VACETemporalWindow: Equatable {
    let ordinal: Int
    /// Exactly windowLength entries. Short clips repeat the final frame index.
    let frameIndices: [Int]
    let validFrameCount: Int
    let overlapWithPrevious: Int
}

enum VACETemporalWindowPlanner {
    static func makeWindows(frameCount: Int, windowLength: Int = 5, overlap: Int = 2) throws -> [VACETemporalWindow] {
        guard frameCount >= 0 else { throw VACERegionPlanningError.invalidOptions("frame count cannot be negative") }
        guard windowLength > 0 else { throw VACERegionPlanningError.invalidOptions("window length must be positive") }
        guard overlap >= 0, overlap < windowLength else {
            throw VACERegionPlanningError.invalidOptions("overlap must be in 0..<windowLength")
        }
        guard frameCount > 0 else { return [] }
        let stride = windowLength - overlap
        var starts = [0]
        if frameCount > windowLength {
            let lastStart = frameCount - windowLength
            var next = stride
            while next <= lastStart { starts.append(next); next += stride }
            if starts.last! < lastStart { starts.append(lastStart) }
        }
        return starts.enumerated().map { ordinal, start in
            let validCount = min(windowLength, frameCount - start)
            var indices = Array(start..<(start + validCount))
            if indices.count < windowLength, let last = indices.last {
                indices.append(contentsOf: repeatElement(last, count: windowLength - indices.count))
            }
            let priorOverlap = ordinal == 0 ? 0 : max(0, starts[ordinal - 1] + windowLength - start)
            return VACETemporalWindow(ordinal: ordinal, frameIndices: indices,
                                      validFrameCount: validCount, overlapWithPrevious: priorOverlap)
        }
    }
}

/// Per-frame handoff buffer. RGBA values are packed four bytes per pixel.
/// The frame decoder/encoder must use the same documented byte ordering.
struct VACERGBAFrame {
    let width: Int
    let height: Int
    let bytes: [UInt8]

    init(width: Int, height: Int, bytes: [UInt8]) throws {
        guard width > 0, height > 0 else { throw VACERegionPlanningError.invalidMaskDimensions }
        let pixels = width.multipliedReportingOverflow(by: height)
        guard !pixels.overflow else { throw VACERegionPlanningError.imageTooLarge(pixelCount: Int.max) }
        let byteCount = pixels.partialValue.multipliedReportingOverflow(by: 4)
        guard !byteCount.overflow || pixels.partialValue > 16_777_216 else {
            throw VACERegionPlanningError.imageTooLarge(pixelCount: Int.max)
        }
        guard pixels.partialValue <= 16_777_216 else {
            throw VACERegionPlanningError.imageTooLarge(pixelCount: pixels.partialValue)
        }
        guard bytes.count == byteCount.partialValue else {
            throw VACERegionPlanningError.maskBufferSizeMismatch(expected: byteCount.partialValue, actual: bytes.count)
        }
        self.width = width; self.height = height; self.bytes = bytes
    }
}

struct VACEReconstructedPatch {
    let sourceRect: VACERegionRect
    /// Already resized by the denoiser/decoder to exactly sourceRect dimensions.
    let frame: VACERGBAFrame
}

enum VACESelectiveCompositor {
    /// The output starts as a byte-for-byte copy of the decoded source. Pixels
    /// under zero mask values are never touched; RGB alone is blended for soft
    /// mask edges, and source alpha is always retained.
    static func composite(original: VACERGBAFrame, mask: VACEMaskPlane,
                          patches: [VACEReconstructedPatch]) throws -> VACERGBAFrame {
        guard original.width == mask.width, original.height == mask.height else {
            throw VACERegionPlanningError.invalidOptions("source frame and mask dimensions differ")
        }
        var output = original.bytes
        let bounds = VACERegionRect(x: 0, y: 0, width: original.width, height: original.height)
        for patch in patches {
            let rect = patch.sourceRect
            guard rect.width == patch.frame.width, rect.height == patch.frame.height,
                  let clipped = rect.intersection(bounds), clipped == rect else {
                throw VACERegionPlanningError.invalidOptions("patch size or coordinates are invalid")
            }
            for localY in 0..<rect.height {
                for localX in 0..<rect.width {
                    let destinationPixel = (rect.y + localY) * original.width + rect.x + localX
                    let alpha = Int(mask.pixels[destinationPixel])
                    if alpha == 0 { continue }
                    let destinationOffset = destinationPixel * 4
                    let patchOffset = (localY * rect.width + localX) * 4
                    for channel in 0..<3 {
                        let sourceValue = Int(original.bytes[destinationOffset + channel])
                        let generatedValue = Int(patch.frame.bytes[patchOffset + channel])
                        output[destinationOffset + channel] = UInt8(
                            (sourceValue * (255 - alpha) + generatedValue * alpha + 127) / 255)
                    }
                    output[destinationOffset + 3] = original.bytes[destinationOffset + 3]
                }
            }
        }
        return try VACERGBAFrame(width: original.width, height: original.height, bytes: output)
    }
}
