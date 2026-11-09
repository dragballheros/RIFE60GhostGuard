import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import Metal
import os

/// Shared pixel-buffer pools, a Metal texture cache for Core Image, and a
/// Bounded multi-frame decode lookahead. Pixel values are not modified here.
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
/// Adaptive bounded frame lookahead. Starts at eight frames, backs off when
/// memory or thermal pressure rises, and slowly restores capacity after pressure
/// clears. The current target is deliberately never allowed above eight.
final class AdaptiveFrameCachePolicy: @unchecked Sendable {
    static let maxDepth = 8
    static let minDepth = 1

    private let lock = NSLock()
    private var targetDepth = maxDepth
    private var lastPressureAt = CFAbsoluteTimeGetCurrent()
    private var lastRecoveryAt = CFAbsoluteTimeGetCurrent()
    private var lastObservedFrameBytes: UInt64 = 8 * 1024 * 1024

    func observe(frameBytes: UInt64?) -> Int {
        if let frameBytes, frameBytes > 0 {
            lock.lock()
            lastObservedFrameBytes = max(lastObservedFrameBytes, frameBytes)
            lock.unlock()
        }

        let available: UInt64
        #if os(iOS) || os(tvOS) || os(watchOS)
        available = UInt64(os_proc_available_memory())
        #else
        // This API is unavailable in the macOS SDK used by the standalone
        // validation harness. The adaptive policy is active on iOS, while
        // host validation treats available memory as unconstrained.
        available = UInt64.max
        #endif
        let thermal = ProcessInfo.processInfo.thermalState
        let now = CFAbsoluteTimeGetCurrent()

        lock.lock()
        defer { lock.unlock() }

        let frameBytes = max(lastObservedFrameBytes, UInt64(4 * 1024 * 1024))
        let pressureReserve = max(UInt64(192 * 1024 * 1024), frameBytes * 2)
        let memoryPressure = available < pressureReserve
        let thermalPressure = thermal == .serious || thermal == .critical

        if memoryPressure || thermalPressure {
            let newDepth: Int
            if thermal == .critical || available < pressureReserve / 2 {
                newDepth = Self.minDepth
            } else {
                newDepth = max(Self.minDepth, targetDepth - 2)
            }
            if newDepth < targetDepth {
                targetDepth = newDepth
                lastPressureAt = now
                lastRecoveryAt = now
            }
            return targetDepth
        }

        // Do not immediately climb after a pressure event. Let the system settle,
        // then recover one slot at a time so a marginal device stays at its
        // highest sustainable depth rather than oscillating between limits.
        let recoveryDelay = 15.0
        let comfortableMemory = max(
            UInt64(512 * 1024 * 1024),
            frameBytes * UInt64(targetDepth + 3)
        )
        if targetDepth < Self.maxDepth,
           available >= comfortableMemory,
           now - lastPressureAt >= recoveryDelay,
           now - lastRecoveryAt >= recoveryDelay {
            targetDepth += 1
            lastRecoveryAt = now
        }

        return targetDepth
    }
}

final class FrameAheadReader: @unchecked Sendable {
    struct Item {
        let buffer: CVPixelBuffer
        let time: CMTime
    }

    private let output: AVAssetReaderOutput
    private let requestedDepth: Int
    private let adaptivePolicy: AdaptiveFrameCachePolicy?
    private let queue = DispatchQueue(label: "frame-ahead-reader")
    private let lock = NSCondition()
    private var pending: [Item] = []
    private var headIndex = 0
    private var decodeSeconds = 0.0
    private var decodeCount = 0
    private var finished = false
    private var started = false
    private(set) var lastWaitSeconds = 0.0

    init(output: AVAssetReaderOutput, depth: Int = 8, adaptive: Bool = true) {
        self.output = output
        self.requestedDepth = min(max(1, depth), AdaptiveFrameCachePolicy.maxDepth)
        self.adaptivePolicy = adaptive ? AdaptiveFrameCachePolicy() : nil
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
        let item = pending[headIndex]
        headIndex += 1
        if headIndex >= pending.count {
            pending.removeAll(keepingCapacity: true)
            headIndex = 0
        } else if headIndex >= 16 && headIndex * 2 >= pending.count {
            pending.removeFirst(headIndex)
            headIndex = 0
        }
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
        var adaptiveDepth = requestedDepth
        while true {
            lock.lock()
            while (pending.count - headIndex) >= adaptiveDepth && !finished {
                lock.wait()
                adaptiveDepth = min(
                    requestedDepth,
                    (adaptivePolicy?.observe(frameBytes: nil) ?? requestedDepth)
                )
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
                let bytes = UInt64(CVPixelBufferGetBytesPerRow(buffer)) * UInt64(CVPixelBufferGetHeight(buffer))
                let ownedBuffer = try? PixelBufferOwnership.copy(buffer)
                if let ownedBuffer {
                    pending.append(Item(buffer: ownedBuffer, time: CMSampleBufferGetPresentationTimeStamp(sample)))
                    adaptiveDepth = min(
                        requestedDepth,
                        (adaptivePolicy?.observe(frameBytes: bytes) ?? requestedDepth)
                    )
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
