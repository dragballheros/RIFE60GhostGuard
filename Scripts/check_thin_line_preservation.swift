import Foundation
import CoreVideo

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fatalError("THIN_LINE_CHECK_FAILED: \(message)") }
}

func makeFrame(width: Int, height: Int, fill: UInt8, column: Int? = nil, columnValue: UInt8 = 20, block: CGRect? = nil, blockValue: UInt8 = 15) throws -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess,
          let buffer = optional else {
        throw NSError(domain: "ThinLineFixture", code: 1)
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw NSError(domain: "ThinLineFixture", code: 2) }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
        let row = pixels.advanced(by: y * rowBytes)
        for x in 0..<width {
            var value = fill
            if let column, x == column, y >= 8, y < height - 8 { value = columnValue }
            if let block, block.contains(CGPoint(x: x, y: y)) { value = blockValue }
            let i = x * 4
            row[i] = value
            row[i + 1] = value
            row[i + 2] = value
            row[i + 3] = 255
        }
    }
    return buffer
}

func gray(_ buffer: CVPixelBuffer, x: Int, y: Int) -> UInt8 {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(buffer))
    return row[x * 4]
}

let pass = ThinLinePreservationPass(sensitivity: 1.35)
let width = 96
let height = 80

let previous = try makeFrame(width: width, height: height, fill: 210, column: 40)
let next = try makeFrame(width: width, height: height, fill: 210, column: 40)
let faded = try makeFrame(width: width, height: height, fill: 210, column: 40, columnValue: 170)
let repaired = pass.preserve(previous: previous, interpolated: faded, next: next, progress: 0.5)
require(repaired != nil, "a static hair line missing from the in-between should be restored")
require(gray(repaired!, x: 40, y: 40) < 40, "the restored stroke should be dark again")
require(gray(repaired!, x: 40, y: 40) <= gray(previous, x: 40, y: 40) + 2, "the stroke should come back from a real source")
let feather = Int(gray(repaired!, x: 41, y: 40))
require(feather < 210 && feather > 40, "the 1px neighbor should be a soft mix, not a hard seam or the original background")
require(gray(repaired!, x: 48, y: 40) == 210, "pixels outside the 2px feather must stay untouched")
require(gray(repaired!, x: 10, y: 10) == gray(faded, x: 10, y: 10), "the rest of the frame must not be replaced")

let soft = try makeFrame(width: width, height: height, fill: 210, column: 40, columnValue: 30)
require(pass.preserve(previous: previous, interpolated: soft, next: next, progress: 0.5) == nil,
        "a stroke that is only slightly softer than the sources must be left alone")

let shiftedNext = try makeFrame(width: width, height: height, fill: 210, column: 41)
let shiftedRepair = pass.preserve(previous: previous, interpolated: faded, next: shiftedNext, progress: 0.25)
require(shiftedRepair != nil && gray(shiftedRepair!, x: 40, y: 40) < 40,
        "a 1px tremble should still count as the same static stroke")

let movedNext = try makeFrame(width: width, height: height, fill: 210, column: 46)
require(pass.preserve(previous: previous, interpolated: faded, next: movedNext, progress: 0.5) == nil,
        "a line that moved several pixels must stay under RIFE, not be pinned in place")

let block = CGRect(x: 24, y: 16, width: 40, height: 40)
let darkPrevious = try makeFrame(width: width, height: height, fill: 210, block: block, blockValue: 12)
let darkNext = try makeFrame(width: width, height: height, fill: 210, block: block, blockValue: 12)
let liftedBlock = try makeFrame(width: width, height: height, fill: 210, block: block, blockValue: 160)
let blockResult = pass.preserve(previous: darkPrevious, interpolated: liftedBlock, next: darkNext, progress: 0.5)
let blockCenter = blockResult.map { gray($0, x: 44, y: 36) } ?? gray(liftedBlock, x: 44, y: 36)
require(blockCenter == 160, "the inside of a large dark region is not a thin stroke and must not be repainted")

let cutPrevious = try makeFrame(width: width, height: height, fill: 220, column: 40)
let cutNext = try makeFrame(width: width, height: height, fill: 25, column: 40, columnValue: 5)
require(pass.preserve(previous: cutPrevious, interpolated: faded, next: cutNext, progress: 0.5) == nil,
        "a scene cut must skip the pass")

let tinyPrevious = try makeFrame(width: width, height: height, fill: 210)
let tinyNext = try makeFrame(width: width, height: height, fill: 210)
let tinyMid = try makeFrame(width: width, height: height, fill: 210)
func paintSpecks(_ buffer: CVPixelBuffer, value: UInt8) {
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 30..<36 {
        let row = base.advanced(by: y * rowBytes)
        let i = 40 * 4
        row[i] = value
        row[i + 1] = value
        row[i + 2] = value
    }
}
paintSpecks(tinyPrevious, value: 20)
paintSpecks(tinyNext, value: 20)
paintSpecks(tinyMid, value: 180)
require(pass.preserve(previous: tinyPrevious, interpolated: tinyMid, next: tinyNext, progress: 0.5) == nil,
        "a handful of speck pixels is below the agreement floor")

let borderline = try makeFrame(width: width, height: height, fill: 210, column: 40, columnValue: 48)
let strict = ThinLinePreservationPass(sensitivity: 1.35)
let loose = ThinLinePreservationPass(sensitivity: 0.65)
require(strict.preserve(previous: previous, interpolated: borderline, next: next, progress: 0.5) != nil,
        "default sensitivity should restore a clearly faded stroke")
require(loose.preserve(previous: previous, interpolated: borderline, next: next, progress: 0.5) == nil,
        "lower sensitivity should leave a milder fade alone")

func shapeFrame(width: Int = 160, height: Int = 90, rect: CGRect) throws -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess,
          let buffer = optional else {
        throw NSError(domain: "ThinLineFixture", code: 3)
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height {
        let row = base.advanced(by: y * rowBytes)
        for x in 0..<width {
            let inside = rect.contains(CGPoint(x: x, y: y))
            let i = x * 4
            // Same pink/beige on a dark ground. Only the shape layout changes.
            row[i] = inside ? 140 : 42
            row[i + 1] = inside ? 150 : 36
            row[i + 2] = inside ? 180 : 40
            row[i + 3] = 255
        }
    }
    return buffer
}

func blendFrames(_ a: CVPixelBuffer, _ b: CVPixelBuffer) throws -> CVPixelBuffer {
    let width = CVPixelBufferGetWidth(a)
    let height = CVPixelBufferGetHeight(a)
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess,
          let buffer = optional else {
        throw NSError(domain: "ThinLineFixture", code: 4)
    }
    CVPixelBufferLockBaseAddress(a, .readOnly)
    CVPixelBufferLockBaseAddress(b, .readOnly)
    CVPixelBufferLockBaseAddress(buffer, [])
    defer {
        CVPixelBufferUnlockBaseAddress(buffer, [])
        CVPixelBufferUnlockBaseAddress(b, .readOnly)
        CVPixelBufferUnlockBaseAddress(a, .readOnly)
    }
    let aBase = CVPixelBufferGetBaseAddress(a)!.assumingMemoryBound(to: UInt8.self)
    let bBase = CVPixelBufferGetBaseAddress(b)!.assumingMemoryBound(to: UInt8.self)
    let out = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let aRow = CVPixelBufferGetBytesPerRow(a)
    let bRow = CVPixelBufferGetBytesPerRow(b)
    let oRow = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height {
        let ar = aBase.advanced(by: y * aRow)
        let br = bBase.advanced(by: y * bRow)
        let orow = out.advanced(by: y * oRow)
        for x in 0..<(width * 4) {
            orow[x] = UInt8((Int(ar[x]) + Int(br[x])) / 2)
        }
    }
    return buffer
}

let leftShape = try shapeFrame(rect: CGRect(x: 16, y: 20, width: 32, height: 32))
let rightShape = try shapeFrame(rect: CGRect(x: 108, y: 24, width: 36, height: 28))
let shiftedShape = try shapeFrame(rect: CGRect(x: 21, y: 20, width: 32, height: 32))
let differentCut = StructuralSceneCut.assess(previous: leftShape, next: rightShape, sensitivity: 1.35)
require(differentCut?.structuralCut == true, "different shapes in the same palette should be a structural cut")
require(differentCut?.classicCut == false, "that cut should stay under the luma/RGB MAD gate")
require((differentCut?.appearance ?? 0) > 0.05 && (differentCut?.appearance ?? 1) < 0.14,
        "appearance change should match the measured similar-palette range")
require((differentCut?.overlap ?? 1) < 0.30, "1px dilation should put a different layout under the 0.30 Dice cut")
let onePixelShift = try shapeFrame(rect: CGRect(x: 17, y: 20, width: 32, height: 32))
let motionOverlap = StructuralSceneCut.assess(previous: leftShape, next: onePixelShift, sensitivity: 1.35)
require((motionOverlap?.overlap ?? 0) > 0.90 && motionOverlap?.isCut == false,
        "a 1px shift should keep Dice near 1, like ordinary motion")
let shiftedCut = StructuralSceneCut.assess(previous: leftShape, next: shiftedShape, sensitivity: 1.35)
require(shiftedCut?.isCut == false, "the same shape shifted a few pixels is motion, not a cut")

let blendedCut = try blendFrames(leftShape, rightShape)
let ghost = GhostGuard(sensitivity: 1.35, enableSceneCuts: true)
let ghostCut = ghost.inspect(previous: leftShape, generated: blendedCut, current: rightShape)
require(ghostCut.reject && ghostCut.reason == "scene cut / transition discontinuity",
        "GhostGuard should reject the blended midframe of a same-palette cut")
let ghostMotion = ghost.inspect(previous: leftShape, generated: leftShape, current: shiftedShape)
require(!ghostMotion.reject, "a few pixels of motion should not be rejected as a cut")

let sourceGuard = SourceFrameOutlierGuard()
let strayBlend = sourceGuard.inspect(previous: leftShape, candidate: blendedCut, next: rightShape)
require(strayBlend.isOutlier && strayBlend.reason == "stray frame at scene cut",
        "a cross-fade parked on a structural cut should not be bridged")
let ownedEndpoint = sourceGuard.inspect(previous: leftShape, candidate: leftShape, next: rightShape)
require(!ownedEndpoint.isOutlier, "a frame that belongs to the outgoing shot must stay")

require(pass.preserve(previous: leftShape, interpolated: blendedCut, next: rightShape, progress: 0.5) == nil,
        "thin-line repair must not run across the structural cut")

func lineFrame(width: Int = 160, height: Int = 90, column: Int, extraColumn: Int? = nil) throws -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess,
          let buffer = optional else {
        throw NSError(domain: "ThinLineFixture", code: 5)
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height {
        let row = base.advanced(by: y * rowBytes)
        for x in 0..<width {
            var value: UInt8 = 200
            if y >= 8, y < height - 8, abs(x - column) <= 1 { value = 16 }
            if let extraColumn, y >= 8, y < height - 8, abs(x - extraColumn) <= 1 { value = 16 }
            let i = x * 4
            row[i] = value
            row[i + 1] = value
            row[i + 2] = value
            row[i + 3] = 255
        }
    }
    return buffer
}

let stableLine = try lineFrame(column: 40)
let stableGuard = ghost.inspect(previous: stableLine, generated: stableLine, current: stableLine)
require(!stableGuard.reject, "an outline that is the same in all three frames is not warped")
let novelLine = try lineFrame(column: 110)
let novelGuard = ghost.inspect(previous: stableLine, generated: novelLine, current: stableLine)
require(novelGuard.reject && novelGuard.reason == "warped/doubled outline",
        "an outline present in neither source should be rejected")
let startLine = try lineFrame(column: 24)
let endLine = try lineFrame(column: 40)
let betweenLine = try lineFrame(column: 32)
let movingGuard = ghost.inspect(previous: startLine, generated: betweenLine, current: endLine)
require(!movingGuard.reject, "a line halfway between the two source positions is legitimate motion")
let doubledLine = try lineFrame(column: 40, extraColumn: 58)
let doubledGuard = ghost.inspect(previous: stableLine, generated: doubledLine, current: stableLine)
require(doubledGuard.reject && doubledGuard.reason == "warped/doubled outline",
        "a second outline next to the real one should be rejected")

func paintSquare(_ buffer: CVPixelBuffer, rect: CGRect, value: UInt8) {
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    for y in 0..<height where rect.minY <= CGFloat(y) && CGFloat(y) < rect.maxY {
        let row = base.advanced(by: y * rowBytes)
        for x in 0..<width where rect.minX <= CGFloat(x) && CGFloat(x) < rect.maxX {
            let i = x * 4
            row[i] = value
            row[i + 1] = value
            row[i + 2] = value
        }
    }
}

let staticA = try lineFrame(column: 80)
let staticC = try lineFrame(column: 80)
paintSquare(staticA, rect: CGRect(x: 8, y: 12, width: 16, height: 16), value: 30)
paintSquare(staticC, rect: CGRect(x: 128, y: 12, width: 16, height: 16), value: 30)
let dragged = try lineFrame(column: 96)
paintSquare(dragged, rect: CGRect(x: 60, y: 12, width: 16, height: 16), value: 30)
let lock = StaticRegionLock(sensitivity: 1.35)
let locked = lock.lock(previous: staticA, interpolated: dragged, next: staticC, progress: 0.25)
require(locked != nil, "static floor lines should be locked back to the sources")
require(gray(locked!, x: 80, y: 40) < 40, "the real static line should come back from the nearer source")
require(gray(locked!, x: 96, y: 40) > 150, "the dragged copy of that line should be removed")
require(gray(locked!, x: 68, y: 20) == 30, "a solid moving body between the two poses must stay")
require(lock.lock(previous: leftShape, interpolated: blendedCut, next: rightShape, progress: 0.5) == nil,
        "static lock must not run across a detected cut")

func panFrame(width: Int = 128, height: Int = 72, shift: Int, soft: Bool) throws -> CVPixelBuffer {
    var optional: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &optional) == kCVReturnSuccess,
          let buffer = optional else {
        throw NSError(domain: "ThinLineFixture", code: 6)
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    let columns = [12, 28, 44, 60]
    for y in 0..<height {
        let row = base.advanced(by: y * rowBytes)
        for x in 0..<width {
            var value: UInt8 = 210
            if y >= 4, y < height - 4, columns.contains(where: { abs(x - ($0 + shift)) == 0 }) {
                value = soft ? 120 : 18
            }
            let i = x * 4
            row[i] = value
            row[i + 1] = value
            row[i + 2] = value
            row[i + 3] = 255
        }
    }
    return buffer
}

let panA = try panFrame(shift: 0, soft: false)
let panC = try panFrame(shift: 3, soft: false)
let panM = try panFrame(shift: 0, soft: true)
let stabilizer = SmallMotionLineStabilizer(sensitivity: 1.35)
let panned = stabilizer.stabilize(previous: panA, interpolated: panM, next: panC, progress: 1.0 / 3.0)
require(panned != nil, "a 3px pan of thin lines should be stabilized")
require(gray(panned!, x: 13, y: 30) < 40, "the line should be crisp at the 1px in-between position")
require(gray(panned!, x: 12, y: 30) > 160, "the soft unshifted line should not remain")
let farC = try panFrame(shift: 24, soft: false)
require(stabilizer.stabilize(previous: panA, interpolated: panM, next: farC, progress: 0.5) == nil,
        "a large pan must be left to RIFE")

print("THIN_LINE_PRESERVATION_PASS: static restore, soft feather, slight-softness skip, 1px tolerance, moving-line skip, large-region skip, scene-cut skip, tiny-mask skip, sensitivity, structural same-palette cut, 1px dilation, warped outline, doubled outline, static-region lock, small-motion line pan")
