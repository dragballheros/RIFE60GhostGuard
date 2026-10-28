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
        rows.append(contentsOf: try benchmarkStrategies(progress: progress))
        rows.append(contentsOf: try benchmarkSharpieTiles(progress: progress))
        rows.append("Policies permit CPU fallback; these labels do not prove every operation used the named accelerator. Sharpie production also uses parallel tiles. The fastest Real-CUGAN strategy that stays inside memory headroom is saved for this model, tile, device, iOS version, and app build. The next render uses it. Before a benchmark, renders use sequential backed predictions.")
        let report = rows.joined(separator: "\n")
        DiagnosticsLogger.shared.log(report)
        return report
    }

    private static func benchmarkStrategies(progress: @escaping @Sendable (Double, String) -> Void) throws -> [String] {
        let models = ["RealCUGAN2xNoise3_Tile704x608", "RealCUGAN2xNoise3_Tile512"]
        var rows = ["Real-CUGAN tile strategy | model | ms/tile | extra MB | result"]
        for name in models {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { continue }
            let tileWidth = name.contains("704") ? 704 : 512
            let tileHeight = name.contains("608") ? 608 : 512
            let available = currentRenderPerformanceSnapshot().availableMemoryMB
            var best: (RealCUGANTileStrategy, Int, Double, Double)?
            for strategy in RealCUGANTileStrategy.candidates(modelName: name, availableMB: available) {
                try Task.checkCancellation()
                progress(1, "Real-CUGAN \(strategy.title) • \(name)")
                let slots = strategy == .concurrentBacked ? RealCUGANStrategyStore.allowedConcurrentSlots(availableMB: available) : (strategy.fixedBatchCount ?? (strategy == .batchedUnbacked ? 2 : 1))
                do {
                    let sample = try timeStrategy(url: url, strategy: strategy, slots: slots, tileWidth: tileWidth, tileHeight: tileHeight)
                    rows.append("\(strategy.title) | \(name) | \(String(format: "%.1f", sample.millisecondsPerTile)) | \(String(format: "%.0f", sample.extraMB)) | kept")
                    DiagnosticsLogger.shared.log("Real-CUGAN strategy sample • \(strategy.rawValue) • \(name) • \(String(format: "%.1f", sample.millisecondsPerTile))ms/tile • extra=\(String(format: "%.0f", sample.extraMB))MB")
                    if best == nil || sample.millisecondsPerTile < best!.2 {
                        best = (strategy, slots, sample.millisecondsPerTile, sample.extraMB)
                    }
                } catch {
                    rows.append("\(strategy.title) | \(name) | - | - | \(error.localizedDescription)")
                    DiagnosticsLogger.shared.log("Real-CUGAN strategy discarded • \(strategy.rawValue) • \(name) • \(error.localizedDescription)")
                }
            }
            if let best {
                RealCUGANStrategyStore.save(RealCUGANStrategyRecord(
                    strategy: best.0.rawValue,
                    slots: best.1,
                    medianMilliseconds: best.2,
                    peakExtraMB: best.3,
                    modelName: name,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    deviceModel: RealCUGANStrategyStore.deviceModel(),
                    systemVersion: RealCUGANStrategyStore.systemVersion(),
                    buildNumber: RealCUGANStrategyStore.buildNumber(),
                    needsRebenchmark: false
                ))
                rows.append("winner | \(name) | \(best.0.title) | \(String(format: "%.1f", best.2)) ms/tile | slots \(best.1)")
            }
        }
        return rows
    }

    private static func benchmarkSharpieTiles(progress: @escaping @Sendable (Double, String) -> Void) throws -> [String] {
        var rows = ["Sharpie tile strategy | ms/frame | ms/tile | result"]
        var best: (SharpieTileStrategy, Double, Double)?
        let frameWidth = 3072
        let frameHeight = 1728
        for strategy in SharpieTileStrategy.candidates() {
            try Task.checkCancellation()
            progress(1, strategy.title)
            do {
                let url = try OutlineEnhancer.compiledModelURL(resource: strategy.resourceName)
                let options = MLModelConfiguration()
                options.computeUnits = ModelComputePreference.current == .auto ? .cpuAndNeuralEngine : ModelComputePreference.current.units
                let model = try MLModel(contentsOf: url, configuration: options)
                let second = strategy.usesUnbackedBatch ? try MLModel(contentsOf: url, configuration: options) : model
                let loadedModels = [model, second]
                let provider = try syntheticInput(for: model)
                let core = max(strategy.tile - 40, 1)
                let tiles = ((frameWidth + core - 1) / core) * ((frameHeight + core - 1) / core)
                // Production batched path keeps two model copies in flight, each
                // issuing a batch of four. Sequential production overlaps two workers.
                let copies = strategy.usesUnbackedBatch ? 2 : 1
                let batch = strategy.usesUnbackedBatch ? 4 : 1
                let workers = strategy.usesUnbackedBatch ? copies : 2
                var samples: [Double] = []
                for _ in 0..<3 {
                    try Task.checkCancellation()
                    let elapsed = try autoreleasepool { () throws -> Double in
                        let start = CFAbsoluteTimeGetCurrent()
                        if strategy.usesUnbackedBatch {
                            let group = DispatchGroup()
                            let errorLock = NSLock()
                            var firstError: Error?
                            for copy in 0..<copies {
                                group.enter()
                                let copyModel = loadedModels[copy]
                                DispatchQueue.global(qos: .userInitiated).async {
                                    defer { group.leave() }
                                    do {
                                        let waves = (tiles + copies * batch - 1) / (copies * batch)
                                        for _ in 0..<max(waves, 1) {
                                            let batchProvider = MLArrayBatchProvider(array: Array(repeating: provider, count: batch))
                                            _ = try copyModel.predictions(from: batchProvider, options: MLPredictionOptions())
                                        }
                                    } catch {
                                        errorLock.lock()
                                        if firstError == nil { firstError = error }
                                        errorLock.unlock()
                                    }
                                }
                            }
                            group.wait()
                            if let firstError { throw firstError }
                        } else {
                            let waves = (tiles + workers - 1) / workers
                            for _ in 0..<max(waves, 1) {
                                _ = try model.prediction(from: provider)
                            }
                        }
                        return (CFAbsoluteTimeGetCurrent() - start) * 1000
                    }
                    samples.append(elapsed)
                }
                let median = samples.sorted()[samples.count / 2]
                let perTile = median / Double(max(tiles, 1))
                rows.append("\(strategy.title) | \(String(format: "%.1f", median)) | \(String(format: "%.1f", perTile)) | kept")
                if best == nil || median < best!.1 { best = (strategy, median, perTile) }
            } catch {
                rows.append("\(strategy.title) | - | - | \(error.localizedDescription)")
            }
        }
        if let best {
            SharpieTileStore.save(best.0, medianMilliseconds: best.1)
            rows.append("winner | \(best.0.title) | \(String(format: "%.1f", best.1)) ms/frame | \(String(format: "%.1f", best.2)) ms/tile")
        }
        return rows
    }

    private static func timeStrategy(url: URL, strategy: RealCUGANTileStrategy, slots: Int, tileWidth: Int, tileHeight: Int) throws -> (millisecondsPerTile: Double, extraMB: Double) {
        let before = currentRenderPerformanceSnapshot().availableMemoryMB
        guard before >= 500 else { throw ProcessorError.conversionFailed("insufficient headroom") }
        let options = MLModelConfiguration()
        options.computeUnits = ModelComputePreference.current == .auto ? .cpuAndNeuralEngine : ModelComputePreference.current.units
        let modelURL: URL
        if let count = strategy.fixedBatchCount {
            guard let fixed = Bundle.main.url(forResource: "\(url.deletingPathExtension().lastPathComponent)_Batch\(count)", withExtension: "mlmodelc") else {
                throw ProcessorError.conversionFailed("fixed-batch model missing")
            }
            modelURL = fixed
        } else {
            modelURL = url
        }
        let model = try MLModel(contentsOf: modelURL, configuration: options)
        let provider = try syntheticInput(for: model)
        let repeats = 3
        // 1080p uses 6 wide tiles or 15 of the 512 tiles. Score the whole frame
        // so a serial concurrent launch is not divided into a fake per-tile win.
        let frameTiles = tileWidth >= 700 ? 6 : 15
        var samples: [Double] = []
        var lowest = before
        for _ in 0..<repeats {
            try Task.checkCancellation()
            let now = currentRenderPerformanceSnapshot()
            if now.tier == .safe || now.availableMemoryMB < 500 { throw ProcessorError.conversionFailed("exceeded memory headroom") }
            lowest = min(lowest, now.availableMemoryMB)
            let elapsed = try autoreleasepool { () throws -> Double in
                let start = CFAbsoluteTimeGetCurrent()
                var remaining = frameTiles
                while remaining > 0 {
                    if strategy == .batchedUnbacked {
                        let count = min(max(2, slots), remaining)
                        try predictUnbackedBatch(model: model, provider: provider, count: count)
                        remaining -= count
                    } else if strategy == .concurrentBacked {
                        let count = min(max(2, slots), remaining)
                        try predictConcurrent(model: model, provider: provider, slots: count, tileWidth: tileWidth, tileHeight: tileHeight)
                        remaining -= count
                    } else if strategy.fixedBatchCount != nil {
                        try predictBacked(model: model, provider: provider, repeats: 1, tileWidth: tileWidth, tileHeight: tileHeight)
                        remaining -= min(strategy.fixedBatchCount ?? 1, remaining)
                    } else {
                        try predictBacked(model: model, provider: provider, repeats: 1, tileWidth: tileWidth, tileHeight: tileHeight)
                        remaining -= 1
                    }
                }
                return (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(frameTiles)
            }
            samples.append(elapsed)
            lowest = min(lowest, currentRenderPerformanceSnapshot().availableMemoryMB)
        }
        if lowest < 500 { throw ProcessorError.conversionFailed("exceeded memory headroom") }
        let extra = max(0, before - lowest)
        return (samples.sorted()[samples.count / 2], extra)
    }

    private static func predictBacked(model: MLModel, provider: MLFeatureProvider, repeats: Int, tileWidth: Int, tileHeight: Int) throws {
        let backing = try RealCUGANOutputBacking.make(for: model)
        let prediction = MLPredictionOptions()
        prediction.outputBackings = backing
        for _ in 0..<repeats { _ = try model.prediction(from: provider, options: prediction) }
    }

    private static func predictUnbackedBatch(model: MLModel, provider: MLFeatureProvider, count: Int) throws {
        let batch = MLArrayBatchProvider(array: Array(repeating: provider, count: count))
        _ = try model.predictions(from: batch, options: MLPredictionOptions())
    }

    private static func predictConcurrent(model: MLModel, provider: MLFeatureProvider, slots: Int, tileWidth: Int, tileHeight: Int) throws {
        let group = DispatchGroup()
        let errorLock = NSLock()
        var firstError: Error?
        for _ in 0..<slots {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { group.leave() }
                do {
                    try predictBacked(model: model, provider: provider, repeats: 1, tileWidth: tileWidth, tileHeight: tileHeight)
                } catch {
                    errorLock.lock()
                    if firstError == nil { firstError = error }
                    errorLock.unlock()
                }
            }
        }
        group.wait()
        if let firstError { throw firstError }
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
