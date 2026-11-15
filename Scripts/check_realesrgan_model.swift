import Foundation
import CoreML

func fail(_ message: String) -> Never {
    fputs("REALESRGAN_COREML_CHECK_FAILED: \(message)\n", stderr)
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: check_realesrgan_model <compiled-model-path>")
}
let url = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuOnly
do {
    let model = try MLModel(contentsOf: url, configuration: configuration)
    guard let inputDescription = model.modelDescription.inputDescriptionsByName["input"]?.multiArrayConstraint,
          let outputDescription = model.modelDescription.outputDescriptionsByName["output"]?.multiArrayConstraint else {
        fail("model must expose input and output multi-array features")
    }
    let expectedInput = [1, 3, 384, 384]
    let expectedOutput = [1, 3, 768, 768]
    guard inputDescription.shape.map(\.intValue) == expectedInput,
          inputDescription.dataType == .float32 else {
        fail("unexpected input tensor contract: \(inputDescription.shape) / \(inputDescription.dataType)")
    }
    guard outputDescription.shape.map(\.intValue) == expectedOutput,
          outputDescription.dataType == .float32 else {
        fail("unexpected output tensor contract: \(outputDescription.shape) / \(outputDescription.dataType)")
    }

    let input = try MLMultiArray(shape: inputDescription.shape, dataType: .float32)
    let inputValues = input.dataPointer.assumingMemoryBound(to: Float.self)
    for index in 0..<input.count {
        inputValues[index] = 0.5
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: [
        "input": MLFeatureValue(multiArray: input)
    ])
    let prediction = try model.prediction(from: provider)
    guard let output = prediction.featureValue(for: "output")?.multiArrayValue,
          output.shape.map(\.intValue) == expectedOutput else {
        fail("prediction output shape is invalid")
    }
    let outputValues = output.dataPointer.assumingMemoryBound(to: Float.self)
    var minimum = Float.greatestFiniteMagnitude
    var maximum = -Float.greatestFiniteMagnitude
    for index in 0..<output.count {
        let value = outputValues[index]
        guard value.isFinite else { fail("prediction contains NaN or infinity at index \(index)") }
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    }
    guard maximum > minimum else {
        fail("prediction is constant and did not produce a meaningful output")
    }
    print("REALESRGAN_COREML_CHECK_PASS: input=384x384 RGB • output=768x768 RGB • range=\(minimum)...\(maximum)")
} catch {
    fail(error.localizedDescription)
}
