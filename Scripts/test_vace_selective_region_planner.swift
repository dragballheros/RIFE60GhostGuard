import Foundation

@main
enum VACESelectiveRegionPlannerTests {
    static func main() throws {
        try testEmptyMaskProducesNoCrops()
        try testMinimumComponentThresholdFiltersNoise()
        try testDisconnectedRegionsRemainSeparate()
        try testMaskUnionCoversMotionAcrossWindow()
        try testCropExtraction()
        try testLargeRegionIsTiledWithBounds()
        try testTemporalWindowsCoverEveryFrameAndPadShortClips()
        try testCompositorPreservesEveryUnmaskedByte()
        print("PASS: VACE selective-region planner/compositor checks")
    }

    static func check(_ condition: Bool, _ message: String) {
        if !condition { fatalError("FAIL: \(message)") }
    }

    static func makeMask(width: Int, height: Int, active: [(Int, Int, UInt8)]) throws -> VACEMaskPlane {
        var pixels = [UInt8](repeating: 0, count: width * height)
        for (x, y, value) in active { pixels[y * width + x] = value }
        return try VACEMaskPlane(width: width, height: height, pixels: pixels)
    }

    static func testEmptyMaskProducesNoCrops() throws {
        let mask = try makeMask(width: 64, height: 64, active: [])
        check(try VACESelectiveRegionPlanner.plan(mask: mask).isEmpty, "empty mask should produce no crop")
    }

    static func testMinimumComponentThresholdFiltersNoise() throws {
        let mask = try makeMask(width: 64, height: 64, active: [(10, 10, 255), (11, 10, 255), (40, 40, 255)])
        var options = VACERegionPlanningOptions()
        options.contextPaddingPixels = 4; options.minimumActivePixels = 2
        let plans = try VACESelectiveRegionPlanner.plan(mask: mask, options: options)
        check(plans.count == 1, "isolated single-pixel noise should be filtered")
        check(plans[0].componentPixelCount == 2, "component pixel count is incorrect")
    }

    static func testDisconnectedRegionsRemainSeparate() throws {
        let mask = try makeMask(width: 160, height: 96,
                                active: [(10, 10, 255), (11, 10, 255), (120, 70, 255), (121, 70, 255)])
        var options = VACERegionPlanningOptions()
        options.contextPaddingPixels = 8; options.cropAlignment = 4
        options.maximumCropWidth = 64; options.maximumCropHeight = 48; options.tileOverlapPixels = 8
        let plans = try VACESelectiveRegionPlanner.plan(mask: mask, options: options)
        check(plans.count == 2, "distant repair areas should become separate crops")
        check(plans.allSatisfy { $0.cropRect.width <= 64 && $0.cropRect.height <= 48 }, "crop limits must hold")
        check(plans.allSatisfy { $0.cropRect.x >= 0 && $0.cropRect.y >= 0 && $0.cropRect.maxX <= 160 && $0.cropRect.maxY <= 96 },
              "crop must stay in source bounds")
    }

    static func testMaskUnionCoversMotionAcrossWindow() throws {
        let first = try makeMask(width: 128, height: 64, active: [(8, 30, 255)])
        let second = try makeMask(width: 128, height: 64, active: [(110, 30, 255)])
        var options = VACERegionPlanningOptions()
        options.contextPaddingPixels = 4; options.cropAlignment = 4; options.minimumActivePixels = 1
        options.maximumCropWidth = 64; options.maximumCropHeight = 48; options.tileOverlapPixels = 8
        let plans = try VACESelectiveRegionPlanner.plan(masks: [first, second], options: options)
        check(plans.count == 2, "separated positions over time should retain separate bounded crops")
        check(plans.contains { $0.cropRect.minX <= 8 && $0.cropRect.maxX > 8 }, "first-frame motion position must be covered")
        check(plans.contains { $0.cropRect.minX <= 110 && $0.cropRect.maxX > 110 }, "later-frame motion position must be covered")
    }

    static func testCropExtraction() throws {
        let frame = try VACERGBAFrame(width: 2, height: 2,
            bytes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        let crop = try frame.cropped(to: VACERegionRect(x: 1, y: 0, width: 1, height: 2))
        check(crop.width == 1 && crop.height == 2, "cropped dimensions are incorrect")
        check(crop.bytes == [4, 5, 6, 7, 12, 13, 14, 15], "crop must copy the requested RGBA rows")
    }

    static func testLargeRegionIsTiledWithBounds() throws {
        var active: [(Int, Int, UInt8)] = []
        for y in 12..<588 { for x in 12..<1012 { active.append((x, y, 255)) } }
        let mask = try makeMask(width: 1024, height: 600, active: active)
        var options = VACERegionPlanningOptions()
        options.contextPaddingPixels = 0; options.cropAlignment = 16; options.minimumActivePixels = 1
        options.maximumCropWidth = 256; options.maximumCropHeight = 160; options.tileOverlapPixels = 32
        options.maximumCropCount = 64
        let plans = try VACESelectiveRegionPlanner.plan(mask: mask, options: options)
        check(plans.count > 1, "large masked regions must be tiled")
        check(plans.allSatisfy { $0.cropRect.width <= 256 && $0.cropRect.height <= 160 }, "source crops must obey bounds")
        check(plans.allSatisfy { $0.inferenceWidth <= 256 && $0.inferenceHeight <= 160 }, "model canvases must obey bounds")
        check(plans.map(\.cropRect).map(\.width).reduce(0, +) > 1000, "tile windows must cover the broad region")
    }

    static func testTemporalWindowsCoverEveryFrameAndPadShortClips() throws {
        let windows = try VACETemporalWindowPlanner.makeWindows(frameCount: 9, windowLength: 5, overlap: 2)
        check(windows.count == 3, "a final fixed-size window must cover the last frame")
        check(windows.allSatisfy { $0.frameIndices.count == 5 }, "all temporal windows need fixed length")
        let valid = Set(windows.flatMap { Array($0.frameIndices.prefix($0.validFrameCount)) })
        check(valid == Set(0..<9), "each source frame must appear in a valid position")
        check(windows[1].overlapWithPrevious >= 2, "successive windows must overlap")
        let short = try VACETemporalWindowPlanner.makeWindows(frameCount: 2, windowLength: 5, overlap: 2)
        check(short.count == 1 && short[0].validFrameCount == 2, "short clips must be padded")
        check(short[0].frameIndices == [0, 1, 1, 1, 1], "edge padding must repeat the final available frame")
        check(try VACETemporalWindowPlanner.makeWindows(frameCount: 0).isEmpty, "empty clip should produce no window")
    }

    static func testCompositorPreservesEveryUnmaskedByte() throws {
        let original = try VACERGBAFrame(width: 3, height: 1, bytes: [10, 20, 30, 77, 100, 120, 140, 88, 200, 210, 220, 99])
        let mask = try VACEMaskPlane(width: 3, height: 1, pixels: [0, 128, 255])
        let patch = try VACERGBAFrame(width: 3, height: 1, bytes: [250, 240, 230, 1, 0, 0, 0, 2, 1, 2, 3, 3])
        let result = try VACESelectiveCompositor.composite(
            original: original, mask: mask,
            patches: [VACEReconstructedPatch(sourceRect: VACERegionRect(x: 0, y: 0, width: 3, height: 1), frame: patch)])
        check(Array(result.bytes[0..<4]) == Array(original.bytes[0..<4]), "mask=0 must preserve original RGBA exactly")
        check(result.bytes[3] == 77 && result.bytes[7] == 88 && result.bytes[11] == 99, "source alpha must not change")
        check(result.bytes[4] == 50 && result.bytes[5] == 60 && result.bytes[6] == 70, "soft mask blend must use integer rounding")
        check(Array(result.bytes[8..<11]) == [1, 2, 3], "mask=255 must copy reconstructed RGB")
    }
}
