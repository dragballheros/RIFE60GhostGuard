import XCTest
import CoreVideo
@testable import RifeMetal

/// Exercises the real bundled RIFE model and Metal stream implementation.
/// These tests intentionally run before the app workflow patches the vendored
/// package manifest for the iOS application build.
final class ProductionSequenceParityTests: XCTestCase {
    private func makeInterpolator() throws -> RifeInterpolator {
        try RifeInterpolator(configuration: .bundled(qualityTier: .hq))
    }

    func testProductionSequence_multiTimestampPairsAndEmptyRebaseMatchStateless() throws {
        let width = 256
        let height = 256
        let frames = try [
            SyntheticFrames.barFrame(width: width, height: height, barXs: [20, 88]),
            SyntheticFrames.barFrame(width: width, height: height, barXs: [36, 104]),
            SyntheticFrames.barFrame(width: width, height: height, barXs: [52, 120]),
            SyntheticFrames.barFrame(width: width, height: height, barXs: [68, 136]),
            SyntheticFrames.barFrame(width: width, height: height, barXs: [84, 152])
        ]

        let cases: [(pair: Int, timesteps: [Float])] = [
            (0, [0.20, 0.40, 0.60, 0.80]),
            (1, [0.25, 0.50, 0.75]),
            // An empty push must encode and rebase the cached previous frame.
            (2, []),
            (3, [0.15, 0.35, 0.65, 0.85])
        ]

        let interpolator = try makeInterpolator()
        let stream = try interpolator.makeStream(width: width, height: height)
        let seeded = try stream.push(frames[0], timesteps: [])
        XCTAssertTrue(seeded.isEmpty, "initial stream seed must not emit frames")

        for testCase in cases {
            let previous = frames[testCase.pair]
            let current = frames[testCase.pair + 1]
            let actual = try stream.push(current, timesteps: testCase.timesteps)

            if testCase.timesteps.isEmpty {
                XCTAssertTrue(actual.isEmpty, "empty-timestep rebase emitted output")
                continue
            }

            let expected = try interpolator.interpolate(
                previous: previous,
                current: current,
                timesteps: testCase.timesteps
            )
            XCTAssertEqual(actual.count, testCase.timesteps.count,
                           "pair \(testCase.pair): stream output count mismatch")
            XCTAssertEqual(expected.count, actual.count,
                           "pair \(testCase.pair): stateless output count mismatch")

            for index in actual.indices {
                let psnr = PSNRHelper.compareInMemory(actual[index], expected[index])
                XCTAssertGreaterThan(
                    psnr, 50.0,
                    "pair \(testCase.pair), timestep \(testCase.timesteps[index]): stream/stateless PSNR \(psnr) dB"
                )

                let defect = try localizedDifference(actual[index], expected[index], threshold: 8)
                XCTAssertLessThanOrEqual(
                    defect.largestComponent, 64,
                    "pair \(testCase.pair), timestep \(testCase.timesteps[index]): localized divergence " +
                    "\(defect.largestComponent) pixels at \(defect.bounds)"
                )
                print(
                    "RIFE_STREAM_PARITY pair=\(testCase.pair) timestep=\(testCase.timesteps[index]) " +
                    "psnr=\(psnr) largest_local_component=\(defect.largestComponent) bounds=\(defect.bounds)"
                )
            }
        }
    }

    private func localizedDifference(
        _ lhs: CVPixelBuffer,
        _ rhs: CVPixelBuffer,
        threshold: Int
    ) throws -> (largestComponent: Int, bounds: String) {
        let width = CVPixelBufferGetWidth(lhs)
        let height = CVPixelBufferGetHeight(lhs)
        guard width == CVPixelBufferGetWidth(rhs),
              height == CVPixelBufferGetHeight(rhs) else {
            throw NSError(domain: "ProductionSequenceParityTests", code: 1)
        }

        CVPixelBufferLockBaseAddress(lhs, .readOnly)
        CVPixelBufferLockBaseAddress(rhs, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(lhs, .readOnly)
            CVPixelBufferUnlockBaseAddress(rhs, .readOnly)
        }

        guard let lhsBase = CVPixelBufferGetBaseAddress(lhs),
              let rhsBase = CVPixelBufferGetBaseAddress(rhs) else {
            throw NSError(domain: "ProductionSequenceParityTests", code: 2)
        }

        let lhsStride = CVPixelBufferGetBytesPerRow(lhs)
        let rhsStride = CVPixelBufferGetBytesPerRow(rhs)
        var mismatch = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            let a = lhsBase.advanced(by: y * lhsStride).assumingMemoryBound(to: UInt8.self)
            let b = rhsBase.advanced(by: y * rhsStride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let offset = x * 4
                if (0..<3).contains(where: { abs(Int(a[offset + $0]) - Int(b[offset + $0])) > threshold }) {
                    mismatch[y * width + x] = 1
                }
            }
        }

        var visited = [UInt8](repeating: 0, count: mismatch.count)
        var largest = 0
        var largestBounds = "none"
        for start in mismatch.indices where mismatch[start] == 1 && visited[start] == 0 {
            var queue = [start]
            visited[start] = 1
            var head = 0
            var count = 0
            var minX = width
            var minY = height
            var maxX = 0
            var maxY = 0

            while head < queue.count {
                let index = queue[head]
                head += 1
                count += 1
                let x = index % width
                let y = index / width
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)

                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        let nx = x + dx
                        let ny = y + dy
                        guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                        let next = ny * width + nx
                        if mismatch[next] == 1 && visited[next] == 0 {
                            visited[next] = 1
                            queue.append(next)
                        }
                    }
                }
            }

            if count > largest {
                largest = count
                largestBounds = "[\(minX),\(minY)]- [\(maxX),\(maxY)]"
            }
        }
        return (largest, largestBounds)
    }
}
