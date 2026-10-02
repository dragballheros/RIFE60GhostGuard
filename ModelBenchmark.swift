import Foundation
import CoreML
import CoreVideo

/// Serial model loads and predictions; called only from a detached background task.
/// Synthetic single-tile timings compare compute policies, not end-to-end video speed.
enum ModelBenchmark {
    static func run(progress: @escaping @Sendable (Double, String) -> Void) throws -> String {
        let candidates = ["RealCUGAN2xNoise3_Tile704x608", "RealCUGAN2xNoise3_Tile512", "AnimeSharpieOutlineSubtle"]
        var rows = ["Model benchmark • 2 warm-ups + 5 timed predictions • median ms", "Model | Auto (.all) | GPU (CPU+GPU) | Neural Engine (CPU+ANE)"]
        DiagnosticsLogger.shared.log("Benchmark started • \(ProcessInfo.processInfo.operatingSystemVersionString) • thermal=\(currentThermalStateName())")
        var completed = 0
        for name in candidates {
            var values: [String] = []
            for preference in ModelComputePreference.allCases {
                try Task.checkCancellation()
                progress(Double(completed) / 9, "\(name) • \(preference.title)")
                do {
                    let median = try autoreleasepool { () throws -> Double in
                        let state = currentRenderPerformanceSnapshot()
                        guard state.tier != .safe, name != candidates[0] || state.allowsWideCUGANTiles else {
                            throw ProcessorError.conversionFailed("insufficient memory/thermal headroom; cool device and retry")
                        }
                        let url: URL
                        if name == "AnimeSharpieOutlineSubtle" { url = try OutlineEnhancer.compiledModelURL() }
                        else {
                            guard let bundled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { throw ProcessorError.conversionFailed("model missing") }
                            url = bundled
                        }
                        let options = MLModelConfiguration()
                        options.computeUnits = preference.units
                        options.allowLowPrecisionAccumulationOnGPU = true
                        let model = try MLModel(contentsOf: url, configuration: options)
                        let provider = try syntheticInput(for: model)
                        for _ in 0..<2 {
                            try Task.checkCancellation()
                            try autoreleasepool { _ = try model.prediction(from: provider) }
                        }
                        var samples: [Double] = []
                        for _ in 0..<5 {
                            try Task.checkCancellation()
                            guard currentRenderPerformanceSnapshot().tier != .safe else { throw ProcessorError.conversionFailed("memory/thermal safety stop") }
                            let elapsed = try autoreleasepool { () throws -> Double in
                                let start = CFAbsoluteTimeGetCurrent()
                                _ = try model.prediction(from: provider)
                                return (CFAbsoluteTimeGetCurrent() - start) * 1000
                            }
                            samples.append(elapsed)
                        }
                        return samples.sorted()[2]
                    }
                    values.append(String(format: "%.1f", median))
                    DiagnosticsLogger.shared.log("Benchmark sample • \(name) • \(preference.title) • median=\(String(format: "%.1f", median))ms • \(currentThermalStateName())")
                } catch is CancellationError { throw CancellationError() }
                catch {
                    values.append("FAILED")
                    DiagnosticsLogger.shared.log("Benchmark failure • \(name) • \(preference.title) • \(error.localizedDescription)")
                }
                completed += 1
                progress(Double(completed) / 9, "\(completed)/9 configurations complete")
            }
            rows.append(([name] + values).joined(separator: " | "))
        }
        rows.append("Policies permit CPU fallback; these labels do not prove every operation used the named accelerator. Sharpie production also uses parallel tiles. No setting changed automatically.")
        let report = rows.joined(separator: "\n")
        DiagnosticsLogger.shared.log(report)
        return report
    }

    private static func syntheticInput(for model: MLModel) throws -> MLFeatureProvider {
        var values: [String: MLFeatureValue] = [:]
        for (name, description) in model.modelDescription.inputDescriptionsByName {
            if let constraint = description.imageConstraint {
                var optional: CVPixelBuffer?
                let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferMetalCompatibilityKey as String: true]
                guard CVPixelBufferCreate(kCFAllocatorDefault, constraint.pixelsWide, constraint.pixelsHigh, constraint.pixelFormatType, attrs as CFDictionary, &optional) == kCVReturnSuccess,
                      let buffer = optional else { throw ProcessorError.conversionFailed("benchmark image allocation failed") }
                CVPixelBufferLockBaseAddress(buffer, [])
                guard let base = CVPixelBufferGetBaseAddress(buffer) else { CVPixelBufferUnlockBaseAddress(buffer, []); throw ProcessorError.conversionFailed("benchmark image address unavailable") }
                base.initializeMemory(as: UInt8.self, repeating: 127, count: CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer))
                if constraint.pixelFormatType == kCVPixelFormatType_32BGRA {
                    let bytes = base.assumingMemoryBound(to: UInt8.self)
                    for row in 0..<constraint.pixelsHigh {
                        for col in 0..<constraint.pixelsWide { bytes[row * CVPixelBufferGetBytesPerRow(buffer) + col * 4 + 3] = 255 }
                    }
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                values[name] = MLFeatureValue(pixelBuffer: buffer)
            } else if let constraint = description.multiArrayConstraint {
                let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
                let value = name == "alpha" ? 1.0 / 1.30 : 0.5
                for index in 0..<array.count { array[index] = NSNumber(value: value) }
                values[name] = MLFeatureValue(multiArray: array)
            } else { throw ProcessorError.conversionFailed("unsupported benchmark input \(name)") }
        }
        return try MLDictionaryFeatureProvider(dictionary: values)
    }
}
