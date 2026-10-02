import Foundation
import CoreVideo
import RifeMetal

enum ProcessorError: Error { case conversionFailed(String) }
final class DiagnosticsLogger { static let shared = DiagnosticsLogger(); func log(_ text: String) { print(text) } }
enum Tier { case performance, safe }
struct Snapshot { var availableMemoryMB: Double; var tier: Tier }
var headroom = 2945.0
var pressure = false
func currentRenderPerformanceSnapshot() -> Snapshot { Snapshot(availableMemoryMB: headroom, tier: pressure ? .safe : .performance) }
func automaticPerformanceModeEnabled() -> Bool { !pressure }
func currentThermalStateName() -> String { "test" }
func require(_ condition: Bool, _ message: String) { if !condition { fatalError(message) } }
func frame(base: Int) throws -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(nil, 8, 96, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess, let result = optional else { throw ProcessorError.conversionFailed("fixture allocation") }
    CVPixelBufferLockBaseAddress(result, [])
    guard let pixels = CVPixelBufferGetBaseAddress(result)?.assumingMemoryBound(to: UInt8.self) else { throw ProcessorError.conversionFailed("fixture pixels") }
    for y in 0..<96 { for x in 0..<8 {
        let offset = y * CVPixelBufferGetBytesPerRow(result) + x * 4
        for channel in 0..<3 { pixels[offset + channel] = UInt8(base + y) }
        pixels[offset + 3] = 255
    } }
    CVPixelBufferUnlockBaseAddress(result, [])
    return result
}
func check(_ outputs: [CVPixelBuffer], bases: [Int]) throws {
    require(outputs.count == bases.count, "output count")
    for (frame, base) in zip(outputs, bases) {
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        guard let pixels = CVPixelBufferGetBaseAddress(frame)?.assumingMemoryBound(to: UInt8.self) else { throw ProcessorError.conversionFailed("result pixels") }
        for y in 0..<96 { for x in 0..<8 {
            let offset = y * CVPixelBufferGetBytesPerRow(frame) + x * 4
            for channel in 0..<3 { require(pixels[offset + channel] == UInt8(base + y), "band mapping / cache rebase at row \(y)") }
            require(pixels[offset + 3] == 255, "alpha")
        } }
        CVPixelBufferUnlockBaseAddress(frame, .readOnly)
    }
}
let parent = RifeInterpolator()
let tiled = try TiledHQInterpolator(interpolator: parent, width: 8, height: 96, overlap: 64, memoryAdaptive: true)
try tiled.seed(frame(base: 20))
try check(tiled.interpolate(current: frame(base: 100), timesteps: [0.25, 0.5, 0.75]), bases: [40, 60, 80])
require(RifeInterpolator.liveStreams == 1, "adaptive mode retains one stream")
headroom = 1800
try check(tiled.interpolate(current: frame(base: 40), timesteps: [0.5]), bases: [70])
require(RifeInterpolator.releases == 1 && RifeInterpolator.liveStreams == 1, "reband releases old stream before graph")
pressure = true
try check(tiled.interpolate(current: frame(base: 60), timesteps: [0.5]), bases: [50])
require(RifeInterpolator.releases == 2, "pressure shrinks working bands")
require(try tiled.interpolate(current: frame(base: 80), timesteps: []).isEmpty, "empty timesteps")
try check(tiled.interpolate(current: frame(base: 100), timesteps: [0.5]), bases: [90])
headroom = 650
var refused = false
do { _ = try tiled.interpolate(current: frame(base: 20), timesteps: [0.5]) } catch { refused = true }
require(refused && RifeInterpolator.liveStreams == 0, "emergency floor releases stream before stopping")
print("ADAPTIVE_RIFE_WRAPPER_PASS: band mapping, rebase, one stream, shrink/reclaim, empty timesteps, emergency floor (test backend)")
