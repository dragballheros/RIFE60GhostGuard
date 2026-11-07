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

/// Creates an app-owned snapshot of a pixel buffer so temporal stages never depend
/// on AVFoundation/CoreVideo buffer reuse or a stateful model's output pool.
enum PixelBufferOwnership {
    static func copy(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let format = CVPixelBufferGetPixelFormatType(source)
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]

        var destination: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            format,
            attributes as CFDictionary,
            &destination
        )
        guard status == kCVReturnSuccess, let destination else {
            throw ProcessorError.conversionFailed("could not allocate owned pixel-buffer snapshot")
        }

        let sourcePlaneCount = CVPixelBufferGetPlaneCount(source)
        let destinationPlaneCount = CVPixelBufferGetPlaneCount(destination)
        guard sourcePlaneCount == destinationPlaneCount else {
            throw ProcessorError.conversionFailed("pixel-buffer plane layout changed during ownership copy")
        }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }

        if sourcePlaneCount == 0 {
            guard let sourceBase = CVPixelBufferGetBaseAddress(source),
                  let destinationBase = CVPixelBufferGetBaseAddress(destination) else {
                throw ProcessorError.conversionFailed("pixel-buffer base address unavailable")
            }
            let sourceRow = CVPixelBufferGetBytesPerRow(source)
            let destinationRow = CVPixelBufferGetBytesPerRow(destination)
            let rowBytes = min(sourceRow, destinationRow)
            for y in 0..<height {
                memcpy(
                    destinationBase.advanced(by: y * destinationRow),
                    sourceBase.advanced(by: y * sourceRow),
                    rowBytes
                )
            }
        } else {
            for plane in 0..<sourcePlaneCount {
                guard let sourceBase = CVPixelBufferGetBaseAddressOfPlane(source, plane),
                      let destinationBase = CVPixelBufferGetBaseAddressOfPlane(destination, plane) else {
                    throw ProcessorError.conversionFailed("planar pixel-buffer base address unavailable")
                }
                let planeHeight = CVPixelBufferGetHeightOfPlane(source, plane)
                let sourceRow = CVPixelBufferGetBytesPerRowOfPlane(source, plane)
                let destinationRow = CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
                let rowBytes = min(sourceRow, destinationRow)
                for y in 0..<planeHeight {
                    memcpy(
                        destinationBase.advanced(by: y * destinationRow),
                        sourceBase.advanced(by: y * sourceRow),
                        rowBytes
                    )
                }
            }
        }

        CVBufferPropagateAttachments(source, destination)
        return destination
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
                let ownedBuffer = try? PixelBufferOwnership.copy(buffer)
                if let ownedBuffer {
                    pending.append(Item(buffer: ownedBuffer, time: CMSampleBufferGetPresentationTimeStamp(sample)))
                } else {
                    finished = true
                    lock.broadcast()
                    lock.unlock()
                    return
                }
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
