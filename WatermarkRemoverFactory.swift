import Foundation

enum WatermarkRemoverFactory {
    static func make(configuration: WatermarkConfiguration) throws -> any WatermarkRemovalApplying {
        switch configuration.model {
        case .lama:
            return try AnimeWatermarkRemover(configuration: configuration)
        case .deepCreamBar, .deepCreamMosaic:
            return try DeepCreamWatermarkRemover(configuration: configuration)
        }
    }
}
