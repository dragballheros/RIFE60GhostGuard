import CoreML
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Reference to one separately exported Wan VACE transformer component.
struct VACEBlockModelReference: Sendable, Hashable {
    let identifier: String
    let relativePackagePath: String
    /// Must change whenever weights, quantization, or the Core ML graph changes.
    let cacheVersion: String

    init(identifier: String, relativePackagePath: String, cacheVersion: String) {
        self.identifier = identifier
        self.relativePackagePath = relativePackagePath
        self.cacheVersion = cacheVersion
    }
}

struct VACEBlockRuntimeTiming: Sendable {
    let identifier: String
    let modelLoadSeconds: TimeInterval
    let predictionSeconds: TimeInterval
    let wasModelReused: Bool
}

enum VACEBlockRuntimeError: LocalizedError {
    case invalidIdentifier(String)
    case pathEscapesModelDirectory(String)
    case packageMissing(URL)
    case compiledModelMissing(URL)
    case nonMultiArrayInput(String)
    case missingInput(String)
    case unexpectedInput(String)
    case nonMultiArrayOutput(String)

    var errorDescription: String? {
        switch self {
        case .invalidIdentifier(let value):
            return "Invalid VACE block identifier: \(value)"
        case .pathEscapesModelDirectory(let path):
            return "VACE package path escapes the configured local model directory: \(path)"
        case .packageMissing(let url):
            return "VACE model package is missing: \(url.lastPathComponent)"
        case .compiledModelMissing(let url):
            return "Compiled VACE model cache is missing or invalid: \(url.lastPathComponent)"
        case .nonMultiArrayInput(let name):
            return "VACE block input '\(name)' is not a tensor input."
        case .missingInput(let name):
            return "Required VACE block input '\(name)' was not provided."
        case .unexpectedInput(let name):
            return "Unexpected VACE block input '\(name)' was provided."
        case .nonMultiArrayOutput(let name):
            return "VACE block output '\(name)' is not a tensor output."
        }
    }
}

/// Serial inference host retaining at most one MLModel object at a time.
///
/// This is a runtime primitive, not the complete Wan VACE model. The denoising
/// coordinator must provide architecture-correct conditioning, modulation, RoPE,
/// masks/context and latent state. An on-device test must still measure actual
/// resident and Metal memory because Core ML may retain internal allocations after
/// this actor releases its MLModel reference.
actor VACEBlockStreamingRuntime {
    private let modelDirectoryURL: URL
    private let compiledCacheDirectoryURL: URL
    private let computeUnits: MLComputeUnits

    private var activeIdentifier: String?
    private var activeModel: MLModel?
    private var activeCompiledURL: URL?

    init(
        modelDirectoryURL: URL,
        compiledCacheDirectoryURL: URL,
        computeUnits: MLComputeUnits = .all
    ) {
        self.modelDirectoryURL = modelDirectoryURL.standardizedFileURL
        self.compiledCacheDirectoryURL = compiledCacheDirectoryURL.standardizedFileURL
        self.computeUnits = computeUnits
    }

    /// Loads one local model, predicts once, and returns all tensor outputs.
    /// The active model remains resident until another block is requested or
    /// unloadActiveModel() is called. Inference is deliberately serial.
    func predict(
        using reference: VACEBlockModelReference,
        inputs: [String: MLMultiArray]
    ) async throws -> ([String: MLMultiArray], VACEBlockRuntimeTiming) {
        try Task.checkCancellation()
        let allowedIdentifierCharacters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        guard !reference.identifier.isEmpty,
              reference.identifier != ".",
              reference.identifier != "..",
              reference.identifier.unicodeScalars.allSatisfy({ allowedIdentifierCharacters.contains($0) })
        else {
            throw VACEBlockRuntimeError.invalidIdentifier(reference.identifier)
        }
        guard !reference.cacheVersion.isEmpty,
              reference.cacheVersion != ".",
              reference.cacheVersion != "..",
              reference.cacheVersion.unicodeScalars.allSatisfy({ allowedIdentifierCharacters.contains($0) })
        else {
            throw VACEBlockRuntimeError.invalidIdentifier(reference.cacheVersion)
        }

        let packageURL = try resolvePackageURL(reference)
        let modelCacheKey = reference.identifier + "-" + reference.cacheVersion
        let wasModelReused = activeIdentifier == modelCacheKey && activeModel != nil
        var loadSeconds: TimeInterval = 0

        if !wasModelReused {
            activeModel = nil
            activeCompiledURL = nil
            activeIdentifier = nil
            await Task.yield()
            try Task.checkCancellation()

            let loadStart = Date()
            let compiledURL = try compileOrFindCachedModel(at: packageURL, reference: reference)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
            activeModel = model
            activeCompiledURL = compiledURL
            activeIdentifier = modelCacheKey
            loadSeconds = Date().timeIntervalSince(loadStart)
        }

        guard let model = activeModel else {
            throw VACEBlockRuntimeError.compiledModelMissing(packageURL)
        }

        let requiredInputs = model.modelDescription.inputDescriptionsByName
        for name in requiredInputs.keys {
            guard let description = requiredInputs[name] else { continue }
            guard description.type == .multiArray else {
                throw VACEBlockRuntimeError.nonMultiArrayInput(name)
            }
            guard inputs[name] != nil else {
                throw VACEBlockRuntimeError.missingInput(name)
            }
        }
        for name in inputs.keys where requiredInputs[name] == nil {
            throw VACEBlockRuntimeError.unexpectedInput(name)
        }

        var featureValues: [String: MLFeatureValue] = [:]
        featureValues.reserveCapacity(inputs.count)
        for (name, array) in inputs {
            featureValues[name] = MLFeatureValue(multiArray: array)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: featureValues)

        try Task.checkCancellation()
        let predictionStart = Date()
        let prediction = try await model.prediction(from: provider)
        let predictionSeconds = Date().timeIntervalSince(predictionStart)

        var outputs: [String: MLMultiArray] = [:]
        for name in prediction.featureNames {
            guard let feature = prediction.featureValue(for: name),
                  let array = feature.multiArrayValue else {
                throw VACEBlockRuntimeError.nonMultiArrayOutput(name)
            }
            outputs[name] = array
        }
        guard !outputs.isEmpty else {
            throw VACEBlockRuntimeError.nonMultiArrayOutput("<none>")
        }

        let timing = VACEBlockRuntimeTiming(
            identifier: reference.identifier,
            modelLoadSeconds: loadSeconds,
            predictionSeconds: predictionSeconds,
            wasModelReused: wasModelReused
        )
        return (outputs, timing)
    }

    /// Evicts the active model reference while preserving compiled model files.
    func unloadActiveModel() {
        activeModel = nil
        activeCompiledURL = nil
        activeIdentifier = nil
    }

    /// Clears compiled cache files only. Source model packages are never removed.
    func clearCompiledModelCache() throws {
        unloadActiveModel()
        if FileManager.default.fileExists(atPath: compiledCacheDirectoryURL.path) {
            try FileManager.default.removeItem(at: compiledCacheDirectoryURL)
        }
    }

    private func resolvePackageURL(_ reference: VACEBlockModelReference) throws -> URL {
        let relativePath = reference.relativePackagePath
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard !relativePath.isEmpty,
              !relativePath.hasPrefix("/"),
              !components.contains(".."),
              !relativePath.contains("\\") else {
            throw VACEBlockRuntimeError.pathEscapesModelDirectory(relativePath)
        }
        let resolved = modelDirectoryURL
            .appendingPathComponent(relativePath)
            .standardizedFileURL
        let base = modelDirectoryURL.path.hasSuffix("/")
            ? modelDirectoryURL.path
            : modelDirectoryURL.path + "/"
        guard resolved.path.hasPrefix(base) else {
            throw VACEBlockRuntimeError.pathEscapesModelDirectory(relativePath)
        }
        var isDirectory: ObjCBool = false
        let isSupportedPackage = ["mlpackage", "mlmodelc"].contains(resolved.pathExtension)
        guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDirectory),
              isDirectory.boolValue,
              isSupportedPackage else {
            throw VACEBlockRuntimeError.packageMissing(resolved)
        }
        return resolved
    }

    private func compileOrFindCachedModel(
        at packageURL: URL,
        reference: VACEBlockModelReference
    ) throws -> URL {
        if packageURL.pathExtension == "mlmodelc" {
            guard FileManager.default.fileExists(atPath: packageURL.path) else {
                throw VACEBlockRuntimeError.compiledModelMissing(packageURL)
            }
            return packageURL
        }

        try FileManager.default.createDirectory(
            at: compiledCacheDirectoryURL,
            withIntermediateDirectories: true
        )
        let cacheURL = compiledCacheDirectoryURL
            .appendingPathComponent(reference.identifier + "-" + reference.cacheVersion)
            .appendingPathExtension("mlmodelc")
        if FileManager.default.fileExists(atPath: cacheURL.path) {
            return cacheURL
        }

        let temporaryCompiledURL = try MLModel.compileModel(at: packageURL)
        do {
            try FileManager.default.copyItem(at: temporaryCompiledURL, to: cacheURL)
        } catch {
            if !FileManager.default.fileExists(atPath: cacheURL.path) {
                throw error
            }
        }
        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            throw VACEBlockRuntimeError.compiledModelMissing(cacheURL)
        }
        return cacheURL
    }
}

#if canImport(UIKit)
/// Optional owner for the memory-warning hook. Keep this object alive for as long
/// as a VACE editing coordinator may be running. The actor remains the sole model
/// owner; this observer only asks it to evict the active model reference.
@MainActor
final class VACEBlockMemoryPressureObserver {
    private let runtime: VACEBlockStreamingRuntime
    private var notificationToken: NSObjectProtocol?

    init(runtime: VACEBlockStreamingRuntime) {
        self.runtime = runtime
        notificationToken = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak runtime] _ in
            guard let runtime else { return }
            Task {
                await runtime.unloadActiveModel()
            }
        }
    }

    deinit {
        if let notificationToken {
            NotificationCenter.default.removeObserver(notificationToken)
        }
    }
}
#endif
