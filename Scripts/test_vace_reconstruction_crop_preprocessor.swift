import Foundation

@main
enum VACEReconstructionCropPreprocessorTests {
    static func main() throws {
        let width = 64, height = 32
        let frame = try VACERGBAFrame(width: width, height: height, bytes: [UInt8](repeating: 127, count: width * height * 4))
        var values = [UInt8](repeating: 0, count: width * height)
        for y in 8..<24 { for x in 16..<48 { values[y * width + x] = 255 } }
        let mask = try VACEMaskPlane(width: width, height: height, pixels: values)
        let prepared = try VACEReconstructionCropPreprocessor.prepare(image: frame, mask: mask, targetWidth: 32, targetHeight: 16)
        guard prepared.image.width == 32, prepared.image.height == 16,
              prepared.mask.width == 32, prepared.mask.height == 16,
              prepared.mask.pixels.allSatisfy({ $0 == 0 || $0 == 255 }) else {
            fatalError("FAIL: bounded VACE preprocessor")
        }
        let restored = try VACEReconstructionCropPreprocessor.restoreGeneratedCrop(prepared.image, toSourceWidth: width, height: height)
        guard restored.width == width, restored.height == height else { fatalError("FAIL: source-size restoration") }
        print("PASS: VACE bounded crop preprocessing")
    }
}
