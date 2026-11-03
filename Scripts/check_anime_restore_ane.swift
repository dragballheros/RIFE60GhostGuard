import CoreML
import Foundation

@main
struct AnimeRestoreANECheck {
    static func main() async {
        do {
            try await run()
        } catch {
            fputs("Anime Restore ANE check failed: \(error)\n", stderr)
            exit(1)
        }
    }

    static func run() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        for name in ["APISR2xRRDB.mlpackage", "SPAN2xHFA2k.mlpackage"] {
            let package = root.appendingPathComponent(name)
            let compiled = try MLModel.compileModel(at: package)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
            guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else {
                fputs("ANE_PLAN \(name) has no ML program main\n", stderr)
                exit(1)
            }
            var neural = 0
            var other = 0
            for operation in main.block.operations {
                let description = String(describing: plan.deviceUsage(for: operation)?.preferred)
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
    }
}
