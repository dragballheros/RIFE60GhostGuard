import CoreML
import Foundation

let root = URL(fileURLWithPath: CommandLine.arguments[1])
for name in ["APISR2xRRDB.mlpackage", "SPAN2xHFA2k.mlpackage"] {
    let package = root.appendingPathComponent(name)
    let compiled = try await MLModel.compileModel(at: package)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    let model = try MLModel(contentsOf: compiled, configuration: configuration)
    print("ANE_LOAD \(name) compiled and loaded with cpuAndNeuralEngine")
    do {
        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
        var neural = 0
        var other = 0
        if case .program(let program) = plan.modelStructure, let function = program.functions["main"] {
            for operation in function.block.operations {
                let description = String(describing: plan.deviceUsage(for: operation)?.preferred)
                if description.localizedCaseInsensitiveContains("neural") {
                    neural += 1
                } else {
                    other += 1
                }
            }
        }
        let input = try MLMultiArray(shape: [1, 3, 256, 256], dataType: .float32)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": input])
        let started = Date()
        _ = try await model.prediction(from: provider)
        let milliseconds = Date().timeIntervalSince(started) * 1000
        print(String(format: "ANE_PLAN %@ neural=%d other=%d predict=%.1fms", name, neural, other, milliseconds))
    } catch {
        print("ANE_PLAN \(name) logged without failing the build: \(error)")
    }
}
print("ANIME_RESTORE_ANE_PASS")
