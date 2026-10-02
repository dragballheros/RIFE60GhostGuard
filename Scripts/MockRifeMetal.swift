// Test-only module: exercises the real tiled wrapper's memory lifecycle and
// band mapping without claiming to test neural-model quality or GPU memory.
import Foundation
import CoreVideo
public final class RifeInterpolator {
    public static var liveStreams = 0
    public static var releases = 0
    public init() {}
    public func makeStream(width: Int, height: Int) throws -> RifeStream { RifeStream(width: width, height: height) }
    public func releaseIdleStreamGraph() {
        precondition(Self.liveStreams == 0, "release must happen after all streams are destroyed")
        Self.releases += 1
    }
}
public final class RifeStream {
    let width: Int, height: Int
    var previous: CVPixelBuffer?
    init(width: Int, height: Int) { self.width = width; self.height = height; RifeInterpolator.liveStreams += 1 }
    deinit { RifeInterpolator.liveStreams -= 1 }
    public func push(_ frame: CVPixelBuffer, timesteps: [Float]) throws -> [CVPixelBuffer] {
        defer { previous = frame }
        guard let previous, !timesteps.isEmpty else { return [] }
        var result: [CVPixelBuffer] = []
        for time in timesteps {
            var optional: CVPixelBuffer?
            guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess, let output = optional else { throw NSError(domain: "test", code: 1) }
            CVPixelBufferLockBaseAddress(previous, .readOnly); CVPixelBufferLockBaseAddress(frame, .readOnly); CVPixelBufferLockBaseAddress(output, [])
            guard let a = CVPixelBufferGetBaseAddress(previous)?.assumingMemoryBound(to: UInt8.self), let b = CVPixelBufferGetBaseAddress(frame)?.assumingMemoryBound(to: UInt8.self), let c = CVPixelBufferGetBaseAddress(output)?.assumingMemoryBound(to: UInt8.self) else { throw NSError(domain: "test", code: 2) }
            for y in 0..<height {
                for x in 0..<(width * 4) {
                    c[y * CVPixelBufferGetBytesPerRow(output) + x] = UInt8((Float(a[y * CVPixelBufferGetBytesPerRow(previous) + x]) * (1 - time) + Float(b[y * CVPixelBufferGetBytesPerRow(frame) + x]) * time).rounded())
                }
            }
            CVPixelBufferUnlockBaseAddress(output, []); CVPixelBufferUnlockBaseAddress(frame, .readOnly); CVPixelBufferUnlockBaseAddress(previous, .readOnly)
            result.append(output)
        }
        return result
    }
}
