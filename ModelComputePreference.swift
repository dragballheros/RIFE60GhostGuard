import Foundation
import CoreML

enum ModelComputePreference: String, CaseIterable, Identifiable, Sendable {
    case auto, gpu, neuralEngine
    var id: String { rawValue }
    var title: String {
        switch self { case .auto: return "Auto"; case .gpu: return "GPU"; case .neuralEngine: return "Neural Engine" }
    }
    var units: MLComputeUnits {
        switch self { case .auto: return .all; case .gpu: return .cpuAndGPU; case .neuralEngine: return .cpuAndNeuralEngine }
    }
    static let defaultsKey = "RIFE60.ModelComputePreference.v1"
    static var current: ModelComputePreference {
        ModelComputePreference(rawValue: UserDefaults.standard.string(forKey: defaultsKey) ?? "auto") ?? .auto
    }
    func persist() { UserDefaults.standard.set(rawValue, forKey: Self.defaultsKey) }
}
