import Foundation
import CoreVideo

/// Finds small, temporally isolated chroma corruption that luminance-only
/// detectors miss. Analysis is performed on a 96x54 grid split into local
/// regions. Repairs alter chroma only and preserve the candidate frame's luma.
final class LocalizedColorArtifactGuard {
    struct Region: Hashable {
        let column: Int
        let row: Int
    }

    struct Inspection {
        let regions: [Region]
        let score: Double
        let reason: String

        var isAnomaly: Bool { !regions.isEmpty }
    }

    private let sampleColumns = 96
    private let sampleRows = 54
    private let tileColumns = 12
    private let tileRows = 9

    func inspect(previous: CVPixelBuffer,
                 candidate: CVPixelBuffer,
                 next: CVPixelBuffer,
                 progress: Double = 0.5) -> Inspection {
        guard let a = Sample(buffer: previous, columns: sampleColumns, rows: sampleRows),
              let b = Sample(buffer: candidate, columns: sampleColumns, rows: sampleRows),
              let c = Sample(buffer: next, columns: sampleColumns, rows: sampleRows),
              a.width == b.width, b.width == c.width,
              a.height == b.height, b.height == c.height else {
            return Inspection(regions: [], score: 0, reason: "color analysis unavailable")
        }

        let t = min(max(progress.isFinite ? progress : 0.5, 0), 1)
        let tileWidth = sampleColumns / tileColumns
        let tileHeight = sampleRows / tileRows
        var suspicious: [Region] = []
        var maxScore = 0.0

        for tileRow in 0..<tileRows {
            for tileColumn in 0..<tileColumns {
                var chromaAB = 0.0
                var chromaBC = 0.0
                var chromaAC = 0.0
                var chromaMidpointError = 0.0
                var lumaAC = 0.0
                var lumaMidpointError = 0.0
                var count = 0.0

                let yStart = tileRow * tileHeight
                let xStart = tileColumn * tileWidth
                for y in yStart..<(yStart + tileHeight) {
                    for x in xStart..<(xStart + tileWidth) {
                        let i = y * sampleColumns + x
                        chromaAB += chromaDistance(a, b, i)
                        chromaBC += chromaDistance(b, c, i)
                        chromaAC += chromaDistance(a, c, i)

                        let expectedU = a.u[i] * (1 - t) + c.u[i] * t
                        let expectedV = a.v[i] * (1 - t) + c.v[i] * t
                        let du = b.u[i] - expectedU
                        let dv = b.v[i] - expectedV
                        chromaMidpointError += sqrt(Double(du * du + dv * dv))

                        lumaAC += Double(abs(a.y[i] - c.y[i]))
                        let expectedY = a.y[i] * (1 - t) + c.y[i] * t
                        lumaMidpointError += Double(abs(b.y[i] - expectedY))
                        count += 1
                    }
                }

                guard count > 0 else { continue }
                let ab = chromaAB / count
                let bc = chromaBC / count
                let ac = chromaAC / count
                let midColor = chromaMidpointError / count
                let endpointLumaChange = lumaAC / count
                let midLuma = lumaMidpointError / count

                // A localized chroma flash usually diverges from both neighbors
                // while those neighbors agree with each other. Luma consistency
                // prevents normal moving edges from being treated as color damage.
                let endpointsAgreeInColor = ac < 0.055 || ac < min(ab, bc) * 0.52
                let stableLuma = endpointLumaChange < 0.075 && midLuma < 0.060
                let isolatedChroma = ab > 0.045 && bc > 0.045 && midColor > 0.045
                if endpointsAgreeInColor && stableLuma && isolatedChroma {
                    suspicious.append(Region(column: tileColumn, row: tileRow))
                    let strength = min(midColor / 0.20, 1.0)
                    let agreement = max(0, 1.0 - min(ac / max(min(ab, bc), 0.001), 1.0))
                    maxScore = max(maxScore, 0.65 * strength + 0.35 * agreement)
                }
            }
        }

        // This detector is deliberately for localized defects, not wholesale
        // scene changes or broad intentional palette changes.
        let affectedFraction = Double(suspicious.count) / Double(tileColumns * tileRows)
        guard !suspicious.isEmpty, affectedFraction <= 0.20 else {
            return Inspection(regions: [], score: maxScore, reason: suspicious.isEmpty
                ? "no localized chroma outlier"
                : "chroma change is widespread")
        }
        return Inspection(regions: suspicious, score: maxScore, reason: "localized temporal chroma outlier")
    }

    /// Returns nil when no region needs repair or safe repair is unavailable.
    /// Within flagged regions, only chroma is pulled toward the temporally
    /// interpolated neighbors. Candidate luma and alpha remain unchanged.
    func repair(previous: CVPixelBuffer,
                candidate: CVPixelBuffer,
                next: CVPixelBuffer,
                progress: Double = 0.5,
                inspection: Inspection) -> CVPixelBuffer? {
        guard inspection.isAnomaly,
              CVPixelBufferGetPixelFormatType(previous) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(candidate) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetPixelFormatType(next) == kCVPixelFormatType_32BGRA else { return nil }

        let width = CVPixelBufferGetWidth(candidate)
        let height = CVPixelBufferGetHeight(candidate)
        guard width == CVPixelBufferGetWidth(previous),
              width == CVPixelBufferGetWidth(next),
              height == CVPixelBufferGetHeight(previous),
              height == CVPixelBufferGetHeight(next) else { return nil }

        var output: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                  attributes as CFDictionary, &output) == kCVReturnSuccess,
              let output else { return nil }

        CVPixelBufferLockBaseAddress(previous, .readOnly)
        CVPixelBufferLockBaseAddress(candidate, .readOnly)
        CVPixelBufferLockBaseAddress(next, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(output, [])
            CVPixelBufferUnlockBaseAddress(next, .readOnly)
            CVPixelBufferUnlockBaseAddress(candidate, .readOnly)
            CVPixelBufferUnlockBaseAddress(previous, .readOnly)
        }

        guard let pBase = CVPixelBufferGetBaseAddress(previous),
              let cBase = CVPixelBufferGetBaseAddress(candidate),
              let nBase = CVPixelBufferGetBaseAddress(next),
              let oBase = CVPixelBufferGetBaseAddress(output) else { return nil }

        let pRowBytes = CVPixelBufferGetBytesPerRow(previous)
        let cRowBytes = CVPixelBufferGetBytesPerRow(candidate)
        let nRowBytes = CVPixelBufferGetBytesPerRow(next)
        let oRowBytes = CVPixelBufferGetBytesPerRow(output)

        // Copy the full image first. Normal regions are bit-identical to input.
        for y in 0..<height {
            let source = cBase.advanced(by: y * cRowBytes).assumingMemoryBound(to: UInt8.self)
            let destination = oBase.advanced(by: y * oRowBytes).assumingMemoryBound(to: UInt8.self)
            for x in 0..<(width * 4) { destination[x] = source[x] }
        }

        let t = Float(min(max(progress.isFinite ? progress : 0.5, 0), 1))
        func unit(_ value: Float) -> Float { min(max(value, 0), 1) }
        func byte(_ value: Float) -> UInt8 { UInt8((unit(value) * 255).rounded()) }
        func luminance(_ b: Float, _ g: Float, _ r: Float) -> Float {
            0.0722 * b + 0.7152 * g + 0.2126 * r
        }

        for region in inspection.regions {
            let x0 = region.column * width / tileColumns
            let x1 = min(width, ((region.column + 1) * width + tileColumns - 1) / tileColumns)
            let y0 = region.row * height / tileRows
            let y1 = min(height, ((region.row + 1) * height + tileRows - 1) / tileRows)

            for y in y0..<y1 {
                let p = pBase.advanced(by: y * pRowBytes).assumingMemoryBound(to: UInt8.self)
                let c = cBase.advanced(by: y * cRowBytes).assumingMemoryBound(to: UInt8.self)
                let n = nBase.advanced(by: y * nRowBytes).assumingMemoryBound(to: UInt8.self)
                let o = oBase.advanced(by: y * oRowBytes).assumingMemoryBound(to: UInt8.self)

                for x in x0..<x1 {
                    let i = x * 4
                    let pb = Float(p[i]) / 255, pg = Float(p[i + 1]) / 255, pr = Float(p[i + 2]) / 255
                    let cb = Float(c[i]) / 255, cg = Float(c[i + 1]) / 255, cr = Float(c[i + 2]) / 255
                    let nb = Float(n[i]) / 255, ng = Float(n[i + 1]) / 255, nr = Float(n[i + 2]) / 255

                    let candidateY = luminance(cb, cg, cr)
                    let previousY = luminance(pb, pg, pr)
                    let nextY = luminance(nb, ng, nr)
                    let expectedY = previousY * (1 - t) + nextY * t
                    guard abs(candidateY - expectedY) < 0.085 else { continue }

                    let candidateU = cb - candidateY, candidateV = cr - candidateY
                    let previousU = pb - previousY, previousV = pr - previousY
                    let nextU = nb - nextY, nextV = nr - nextY
                    let expectedU = previousU * (1 - t) + nextU * t
                    let expectedV = previousV * (1 - t) + nextV * t
                    let du = candidateU - expectedU, dv = candidateV - expectedV
                    let deviation = sqrt(du * du + dv * dv)
                    guard deviation > 0.030 else { continue }

                    let amount = min(max(Double(deviation) * 3.0, 0.60), 0.90)
                    let repairedU = candidateU * Float(1 - amount) + expectedU * Float(amount)
                    let repairedV = candidateV * Float(1 - amount) + expectedV * Float(amount)
                    let red = candidateY + repairedV
                    let blue = candidateY + repairedU
                    let green = (candidateY - 0.2126 * red - 0.0722 * blue) / 0.7152
                    o[i] = byte(blue)
                    o[i + 1] = byte(green)
                    o[i + 2] = byte(red)
                    // Alpha is preserved by the initial copy.
                }
            }
        }

        CVBufferPropagateAttachments(candidate, output)
        return output
    }

    private func chromaDistance(_ a: Sample, _ b: Sample, _ index: Int) -> Double {
        let du = a.u[index] - b.u[index]
        let dv = a.v[index] - b.v[index]
        return sqrt(Double(du * du + dv * dv))
    }

    private struct Sample {
        let width: Int
        let height: Int
        let y: [Float]
        let u: [Float]
        let v: [Float]

        init?(buffer: CVPixelBuffer, columns: Int, rows: Int) {
            guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { return nil }
            let width = CVPixelBufferGetWidth(buffer)
            let height = CVPixelBufferGetHeight(buffer)
            guard width >= columns, height >= rows else { return nil }

            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            let pixels = base.assumingMemoryBound(to: UInt8.self)
            var lumas: [Float] = []
            var us: [Float] = []
            var vs: [Float] = []
            lumas.reserveCapacity(columns * rows)
            us.reserveCapacity(columns * rows)
            vs.reserveCapacity(columns * rows)

            for row in 0..<rows {
                let py = min(height - 1, ((2 * row + 1) * height) / (2 * rows))
                let source = pixels.advanced(by: py * rowBytes)
                for column in 0..<columns {
                    let px = min(width - 1, ((2 * column + 1) * width) / (2 * columns))
                    let i = px * 4
                    let b = Float(source[i]) / 255
                    let g = Float(source[i + 1]) / 255
                    let r = Float(source[i + 2]) / 255
                    let luma = 0.0722 * b + 0.7152 * g + 0.2126 * r
                    lumas.append(luma)
                    us.append(b - luma)
                    vs.append(r - luma)
                }
            }
            self.width = columns
            self.height = rows
            self.y = lumas
            self.u = us
            self.v = vs
        }
    }
}
