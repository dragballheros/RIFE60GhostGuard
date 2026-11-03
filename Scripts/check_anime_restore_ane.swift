import CoreML
import Foundation

let root = URL(fileURLWithPath: CommandLine.arguments[1])
for name in ["APISR2xRRDB.mlpackage", "SPAN2xHFA2k.mlpackage"] {
    let package = root.appendingPathComponent(name)
    let compiled = try MLModel.compileModel(at: package)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    let plan = try MLComputePlan(modelAt: compiled, configuration: configuration)
    var neural = 0
    var other = 0
    guard case .program(let program) = plan.modelStructure else {
        fputs("ANE_PLAN \(name) model is not an ML program\n", stderr)
        exit(1)
    }
    for operation in program.operations {
        let description = String(describing: plan.deviceUsage(for: operation))
        if description.localizedCaseInsensitiveContains("neural") {
            neural += 1
        } else {
            other += 1
            fputs("ANE_OP \(name) \(operation.operatorName) \(description)\n", stderr)
        }
    }
    let input = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
    let model = try MLModel(contentsOf: compiled, configuration: configuration)
    let provider = try MLDictionaryFeatureProvider(dictionary: ["input": input])
    _ = try model.prediction(from: provider)
    let started = Date()
    _ = try model.prediction(from: provider)
    let milliseconds = Date().timeIntervalSince(started) * 1000
    print(String(format: "ANE_PLAN %@ neural=%d other=%d predict=%.1fms", name, neural, other, milliseconds))
    if neural == 0 {
        fputs("ANE_PLAN \(name) compiled without Neural Engine operations\n", stderr)
        exit(1)
    }
}
print("ANIME_RESTORE_ANE_PASS")
