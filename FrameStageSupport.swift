import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import Metal

/// Shared pixel-buffer pools, a Metal texture cache for Core Image, and a
/// two-frame decode lookahead. Pixel values are not modified here.
enum FrameStageResources {
    static func makeCIContext(colorSpace: CGColorSpace? = nil) -> CIContext {
        var options: [CIContextOption: Any] = [.cacheIntermediates: false]
        if let colorSpace {
            options[.workingColorSpace] = colorSpace
            options[.outputColorSpace] = colorSpace
        }
        if let device = MTLCreateSystemDefaultDevice() {
            var cache: CVMetalTextureCache?
            if CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess, let cache {
                options[CIContextOption(rawValue: "metalTextureCache")] = cache
            }
        }
        return CIContext(options: options)
    }

    static func poolAttributes(width: Int, height: Int, format: OSType, minimumCount: Int = 3) -> [String: Any] {
        [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferPoolMinimumBufferCountKey as String: minimumCount
        ]
    }
}

/// Decodes the next source frame while the caller infers the current one.
/// Sample order and presentation timestamps stay the same.
final class FrameAheadReader: @unchecked Sendable {
    struct Item {
        let buffer: CVPixelBuffer
        let time: CMTime
    }

    private let output: AVAssetReaderOutput
    private let depth: Int
    private let queue = DispatchQueue(label: "frame-ahead-reader")
    private let lock = NSCondition()
    private var pending: [Item] = []
    private var decodeSeconds = 0.0
    private var decodeCount = 0
    private var finished = false
    private var started = false
    /// Time the last `next()` spent blocked until a decoded frame was available.
    private(set) var lastWaitSeconds = 0.0

    init(output: AVAssetReaderOutput, depth: Int = 2) {
        self.output = output
        self.depth = max(1, depth)
    }

    var averageDecodeMilliseconds: Double {
        lock.lock()
        defer { lock.unlock() }
        guard decodeCount > 0 else { return 0 }
        return decodeSeconds * 1000 / Double(decodeCount)
    }

    func next() -> Item? {
        let waited = CFAbsoluteTimeGetCurrent()
        start()
        lock.lock()
        while pending.isEmpty && !finished {
            lock.wait()
        }
        if pending.isEmpty {
            lock.unlock()
            lastWaitSeconds = CFAbsoluteTimeGetCurrent() - waited
            return nil
        }
        let item = pending.removeFirst()
        lock.signal()
        lock.unlock()
        lastWaitSeconds = CFAbsoluteTimeGetCurrent() - waited
        return item
    }

    private func start() {
        lock.lock()
        let shouldStart = !started
        started = true
        lock.unlock()
        guard shouldStart else { return }
        queue.async { [weak self] in
            self?.produce()
        }
    }

    private func produce() {
        while true {
            lock.lock()
            while pending.count >= depth && !finished {
                lock.wait()
            }
            if finished {
                lock.unlock()
                return
            }
            lock.unlock()

            let startedAt = CFAbsoluteTimeGetCurrent()
            let sample = output.copyNextSampleBuffer()
            let elapsed = CFAbsoluteTimeGetCurrent() - startedAt

            lock.lock()
            decodeSeconds += elapsed
            decodeCount += 1
            if let sample, let buffer = CMSampleBufferGetImageBuffer(sample) {
                pending.append(Item(buffer: buffer, time: CMSampleBufferGetPresentationTimeStamp(sample)))
                lock.signal()
                lock.unlock()
            } else {
                finished = true
                lock.broadcast()
                lock.unlock()
                return
            }
        }
    }
}
