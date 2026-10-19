import Foundation
import CoreVideo

/// A second, deliberately narrow guard for RIFE failures that occur during
/// large/fast motion. It never drops an output timestamp. When a synthesized
/// frame is rejected, the timestamp is kept and the nearest source endpoint
/// is written instead, so the output cadence remains 60 FPS.
final class FastMotionGhostGuard {
    struct Result {
        let fastMotion: Bool
        let reject: Bool
        let recoveryRecommended: Bool
        let motionScore: Double
        let artifactScore: Double
    }

    private let sensitivity: Double
    private let sampleColumns = 64
    private let sampleRows = 36

    init(sensitivity: Double) {
        self.sensitivity = min(max(sensitivity.isFinite ? sensitivity : 1.35, 0.65), 1.35)
    }

    func inspect(previous: CVPixelBuffer,
                 generated: CVPixelBuffer,
                 current: CVPixelBuffer) -> Result {
        guard CVPixelBufferGetPixelFormatType(previous) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(generated) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(current) == kCVPixelFormatType_32BGRA else {
            return Result(fastMotion: false, reject: false, recoveryRecommended: false, motionScore: 0, artifactScore: 0)
        }

        let width = min(CVPixelBufferGetWidth(previous), min(CVPixelBufferGetWidth(generated), CVPixelBufferGetWidth(current)))
        let height = min(CVPixelBufferGetHeight(previous), min(CVPixelBufferGetHeight(generated), CVPixelBufferGetHeight(current)))
        guard width >= 64, height >= 64 else {
            return Result(fastMotion: false, reject: false, recoveryRecommended: false, motionScore: 0, artifactScore: 0)
        }

        CVPixelBufferLockBaseAddress(previous, .readOnly)
        CVPixelBufferLockBaseAddress(generated, .readOnly)
        CVPixelBufferLockBaseAddress(current, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(current, .readOnly)
            CVPixelBufferUnlockBaseAddress(generated, .readOnly)
            CVPixelBufferUnlockBaseAddress(previous, .readOnly)
        }

        guard let previousBase = CVPixelBufferGetBaseAddress(previous),
              let generatedBase = CVPixelBufferGetBaseAddress(generated),
              let currentBase = CVPixelBufferGetBaseAddress(current) else {
            return Result(fastMotion: false, reject: false, recoveryRecommended: false, motionScore: 0, artifactScore: 0)
        }

        let previousRow = CVPixelBufferGetBytesPerRow(previous)
        let generatedRow = CVPixelBufferGetBytesPerRow(generated)
        let currentRow = CVPixelBufferGetBytesPerRow(current)

        let motionGate = max(0.055, 0.073 - sensitivity * 0.012)
        let artifactGate = 0.76 - sensitivity * 0.10

        var motionSum = 0.0
        var highMotionSamples = 0
        var rangeViolations = 0
        var blurCollapses = 0
        var edgeSamples = 0
        var sampleCount = 0

        // Regional counters prevent a small, severely damaged part of an
        // otherwise acceptable frame from disappearing in whole-frame averages.
        let regionColumns = 8
        let regionRows = 6
        let regionCount = regionColumns * regionRows
        var regionalMotion = [Double](repeating: 0, count: regionCount)
        var regionalHighMotion = [Int](repeating: 0, count: regionCount)
        var regionalRangeViolations = [Int](repeating: 0, count: regionCount)
        var regionalBlurCollapses = [Int](repeating: 0, count: regionCount)
        var regionalEdgeSamples = [Int](repeating: 0, count: regionCount)
        var regionalSamples = [Int](repeating: 0, count: regionCount)

        let xStep = max(1, width / sampleColumns)
        let yStep = max(1, height / sampleRows)
        let xOffset = xStep / 2
        let yOffset = yStep / 2

        for row in 0..<sampleRows {
            let y = min(height - 2, yOffset + row * yStep)
            let py = min(height - 2, y + 1)
            let ny = max(1, y - 1)

            let pRow = previousBase.advanced(by: y * previousRow).assumingMemoryBound(to: UInt8.self)
            let gRow = generatedBase.advanced(by: y * generatedRow).assumingMemoryBound(to: UInt8.self)
            let cRow = currentBase.advanced(by: y * currentRow).assumingMemoryBound(to: UInt8.self)
            let pRowUp = previousBase.advanced(by: ny * previousRow).assumingMemoryBound(to: UInt8.self)
            let pRowDown = previousBase.advanced(by: py * previousRow).assumingMemoryBound(to: UInt8.self)
            let gRowUp = generatedBase.advanced(by: ny * generatedRow).assumingMemoryBound(to: UInt8.self)
            let gRowDown = generatedBase.advanced(by: py * generatedRow).assumingMemoryBound(to: UInt8.self)
            let cRowUp = currentBase.advanced(by: ny * currentRow).assumingMemoryBound(to: UInt8.self)
            let cRowDown = currentBase.advanced(by: py * currentRow).assumingMemoryBound(to: UInt8.self)

            for column in 0..<sampleColumns {
                let x = min(width - 2, xOffset + column * xStep)
                let regionColumn = column / (sampleColumns / regionColumns)
                let regionRow = row / (sampleRows / regionRows)
                let regionIndex = regionRow * regionColumns + regionColumn
                regionalSamples[regionIndex] += 1
                let offset = x * 4
                let leftOffset = max(0, x - 1) * 4
                let rightOffset = min(width - 1, x + 1) * 4

                let pl = luminance(pRow, offset)
                let gl = luminance(gRow, offset)
                let cl = luminance(cRow, offset)

                let endpointMotion = abs(cl - pl)
                motionSum += endpointMotion
                regionalMotion[regionIndex] += endpointMotion
                if endpointMotion > 0.12 {
                    highMotionSamples += 1
                    regionalHighMotion[regionIndex] += 1
                }

                let low = min(pl, cl)
                let high = max(pl, cl)
                let outside = max(low - gl, gl - high)
                if outside > 0.10 {
                    rangeViolations += 1
                    regionalRangeViolations[regionIndex] += 1
                }

                let pg = gradient(pRow, pRowUp, pRowDown, leftOffset, rightOffset, offset)
                let gg = gradient(gRow, gRowUp, gRowDown, leftOffset, rightOffset, offset)
                let cg = gradient(cRow, cRowUp, cRowDown, leftOffset, rightOffset, offset)
                let endpointEdge = max(pg, cg)
                if endpointEdge > 0.20 {
                    edgeSamples += 1
                    regionalEdgeSamples[regionIndex] += 1
                    if gg < endpointEdge * 0.45 {
                        blurCollapses += 1
                        regionalBlurCollapses[regionIndex] += 1
                    }
                }
                sampleCount += 1
            }
        }

        guard sampleCount > 0 else {
            return Result(fastMotion: false, reject: false, recoveryRecommended: false, motionScore: 0, artifactScore: 0)
        }

        let motionScore = motionSum / Double(sampleCount)
        let highMotionFraction = Double(highMotionSamples) / Double(sampleCount)
        let rangeViolationFraction = Double(rangeViolations) / Double(sampleCount)
        let blurCollapseFraction = edgeSamples > 0 ? Double(blurCollapses) / Double(edgeSamples) : 0

        let fastMotion = motionScore >= motionGate && highMotionFraction >= 0.10
        guard fastMotion else {
            return Result(fastMotion: false, reject: false, recoveryRecommended: false, motionScore: motionScore, artifactScore: 0)
        }

        // Keep the existing global signal, then inspect each spatial region
        // independently. A frame with one good half and one badly warped half
        // must not pass simply because the unaffected pixels dilute the average.
        let rangeSignal = min(rangeViolationFraction / 0.20, 1.0)
        let blurSignal = min(blurCollapseFraction / 0.28, 1.0)
        let globalArtifactScore = 0.58 * rangeSignal + 0.42 * blurSignal

        var peakRegionalScore = 0.0
        var regionalRecoveryRecommended = false
        var regionalReject = false
        for region in 0..<regionCount {
            let count = Double(max(regionalSamples[region], 1))
            let localRange = Double(regionalRangeViolations[region]) / count
            let localEdgeCount = regionalEdgeSamples[region]
            let localBlur = localEdgeCount > 0
                ? Double(regionalBlurCollapses[region]) / Double(localEdgeCount)
                : 0.0
            let localRangeSignal = min(localRange / 0.20, 1.0)
            let localBlurSignal = min(localBlur / 0.28, 1.0)
            let localScore = 0.58 * localRangeSignal + 0.42 * localBlurSignal
            peakRegionalScore = max(peakRegionalScore, localScore)

            if localRange >= 0.06 || localBlur >= 0.16 {
                regionalRecoveryRecommended = true
            }

            // Hard-reject only a concentrated, high-severity defect. This avoids
            // flattening normal motion while catching a locally broken patch even
            // when it affects just one of the 48 analysis regions.
            let severeRangePatch = localRange >= 0.20 && localScore >= max(0.54, artifactGate - 0.08)
            // A uniform loss of edge energy can be valid at a temporal midpoint
            // when opposing endpoint detail cancels. Only classify blur as a
            // localized warp when this region is materially worse than the
            // frame-wide blur rate; concentrated damaged patches still qualify.
            let severeWarpPatch = localBlur >= max(0.60, blurCollapseFraction + 0.18) &&
                localScore >= 0.40 &&
                (regionalHighMotion[region] > 0 || regionalMotion[region] / count >= motionGate * 0.25)
            if severeRangePatch || severeWarpPatch {
                regionalReject = true
            }
        }

        let artifactScore = max(globalArtifactScore, peakRegionalScore)
        // A frame can be visibly difficult without crossing the hard rejection
        // threshold. Extra RIFE work is spent when either global or local evidence
        // suggests a partial-frame failure.
        let recoveryRecommended =
            artifactScore >= max(0.42, artifactGate - 0.18) ||
            rangeViolationFraction >= 0.08 ||
            blurCollapseFraction >= 0.12 ||
            regionalRecoveryRecommended

        let reject = regionalReject || (
            globalArtifactScore >= artifactGate &&
            (rangeViolationFraction >= 0.14 || blurCollapseFraction >= 0.20)
        )

        return Result(
            fastMotion: true,
            reject: reject,
            recoveryRecommended: recoveryRecommended,
            motionScore: motionScore,
            artifactScore: artifactScore
        )
    }

    private func luminance(_ row: UnsafePointer<UInt8>, _ offset: Int) -> Double {
        let b = Double(row[offset])
        let g = Double(row[offset + 1])
        let r = Double(row[offset + 2])
        return (0.0722 * b + 0.7152 * g + 0.2126 * r) / 255.0
    }

    private func gradient(_ row: UnsafePointer<UInt8>,
                          _ rowUp: UnsafePointer<UInt8>,
                          _ rowDown: UnsafePointer<UInt8>,
                          _ leftOffset: Int,
                          _ rightOffset: Int,
                          _ offset: Int) -> Double {
        let horizontal = luminance(row, rightOffset) - luminance(row, leftOffset)
        let vertical = luminance(rowDown, offset) - luminance(rowUp, offset)
        return sqrt(horizontal * horizontal + vertical * vertical)
    }
}
