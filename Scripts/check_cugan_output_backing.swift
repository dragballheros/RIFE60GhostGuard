import CoreML
import Foundation

enum BackingCheckError: Error, CustomStringConvertible {
    case message(String)
    var description: String { switch self { case .message(let text): return text } }
}

func expect(_ condition: Bool, _ message: String) throws {
    if !condition { throw BackingCheckError.message(message) }
}

let directory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let names = ["RealCUGAN2xNoise3_Tile512", "RealCUGAN2xNoise3_Tile704x608"]
for name in names {
    let url = directory.appendingPathComponent("\(name).mlmodelc")
    let model = try MLModel(contentsOf: url)
    let backing = try RealCUGANOutputBacking.make(for: model)
    try RealCUGANOutputBacking.assertDirectBacking(backing)
    guard let outputName = model.modelDescription.outputDescriptionsByName.keys.first,
          let value = backing[outputName] else {
        throw BackingCheckError.message("\(name) backing is missing its output feature")
    }
    if let image = model.modelDescription.outputDescriptionsByName[outputName]?.imageConstraint {
        guard let buffer = value as? CVPixelBuffer else {
            throw BackingCheckError.message("\(name) image output was not backed by a CVPixelBuffer")
        }
        try expect(CVPixelBufferGetWidth(buffer) == image.pixelsWide, "\(name) backing width")
        try expect(CVPixelBufferGetHeight(buffer) == image.pixelsHigh, "\(name) backing height")
        try expect(CVPixelBufferGetPixelFormatType(buffer) == image.pixelFormatType, "\(name) backing pixel format")
    } else if let array = model.modelDescription.outputDescriptionsByName[outputName]?.multiArrayConstraint {
        guard let multi = value as? MLMultiArray else {
            throw BackingCheckError.message("\(name) multiarray output was not backed by an MLMultiArray")
        }
        try expect(multi.dataType == array.dataType, "\(name) backing data type")
        try expect(multi.shape == array.shape, "\(name) backing shape")
    }

    var inputs: [String: MLFeatureValue] = [:]
    for (inputName, description) in model.modelDescription.inputDescriptionsByName {
        if let image = description.imageConstraint {
            inputs[inputName] = MLFeatureValue(pixelBuffer: try RealCUGANOutputBacking.pixelBuffer(matching: image))
        } else if let array = description.multiArrayConstraint {
            let multi = try MLMultiArray(shape: array.shape, dataType: array.dataType)
            inputs[inputName] = MLFeatureValue(multiArray: multi)
        } else {
            throw BackingCheckError.message("\(name) has an unsupported input \(inputName)")
        }
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
    let options = MLPredictionOptions()
    options.outputBackings = backing
    do {
        _ = try model.prediction(from: provider, options: options)
    } catch {
        throw BackingCheckError.message("\(name) rejected its output backing: \(error.localizedDescription)")
    }
    print("\(name) accepted a direct \(type(of: value)) backing for \(outputName)")
}
print("Real-CUGAN output backings accepted")
