import CoreVideo
import Foundation

/// Bytes our own pools can drop without changing a finished frame.
/// The governor adds this to raw headroom when it chooses Performance or Balanced.
final class ReclaimableMemoryRegistry: @unchecked Sendable {
    static let shared = ReclaimableMemoryRegistry()

    private let lock = NSLock()
    private var bytes: [String: Int] = [:]
    private var flushers: [String: () -> Void] = [:]

    var totalBytes: Int {
        lock.lock()
        let swiftBytes = bytes.values.reduce(0, +)
        lock.unlock()
        return swiftBytes + Int(h3WeightPoolReclaimableBytes())
    }

    func track(name: String, bytes held: Int, flush: @escaping () -> Void) {
        lock.lock()
        bytes[name] = max(0, held)
        flushers[name] = flush
        lock.unlock()
    }

    func remove(name: String) {
        lock.lock()
        bytes[name] = nil
        flushers[name] = nil
        lock.unlock()
    }

    func trackPixelPool(_ pool: CVPixelBufferPool, name: String, width: Int, height: Int, slots: Int = 3, bytesPerPixel: Int = 4) {
        let held = max(0, width) * max(0, height) * max(1, slots) * max(1, bytesPerPixel)
        track(name: name, bytes: held) {
            CVPixelBufferPoolFlush(pool, .excessBuffers)
        }
    }

    /// Drops idle pool buffers. Called before the Safe floor is applied.
    func flushAll() {
        lock.lock()
        let actions = Array(flushers.values)
        lock.unlock()
        for action in actions { action() }
        h3WeightPoolTrimIdle()
        DiagnosticsLogger.shared.log("Memory governor flushed reclaimable pools • \(totalBytes / 1_048_576) MB still reserved")
    }
}

@_silgen_name("h3_weight_pool_reclaimable_bytes")
private func h3WeightPoolReclaimableBytes() -> UInt

@_silgen_name("h3_weight_pool_trim_idle")
private func h3WeightPoolTrimIdle()
