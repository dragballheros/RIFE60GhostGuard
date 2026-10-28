import CoreML
import CoreVideo
import Foundation

/// Builds `MLPredictionOptions.outputBackings` from a model description.
/// Core ML accepts only a `CVPixelBuffer` or an `MLMultiArray` as the value.
/// An `MLFeatureValue` wrapper is rejected even when it holds one of those.
enum RealCUGANOutputBacking {
    enum BackingError: LocalizedError {
        case message(String)
        var errorDescription: String? {
            switch self {
            case .message(let text): return text
            }
        }
    }

    static func make(for model: MLModel) throws -> [String: Any] {
        var backing: [String: Any] = [:]
        for (name, description) in model.modelDescription.outputDescriptionsByName {
            if let image = description.imageConstraint {
                backing[name] = try pixelBuffer(matching: image)
            } else if let array = description.multiArrayConstraint {
                backing[name] = try MLMultiArray(shape: array.shape, dataType: array.dataType)
            } else {
                throw BackingError.message("Real-CUGAN output \(name) is neither an image nor a multi array")
            }
        }
        try assertDirectBacking(backing)
        return backing
    }

    /// Use a pooled image when it already matches the model's image output.
    /// Otherwise allocate a buffer from the constraint. The dictionary value is
    /// the buffer or array itself.
    static func imageBacking(named name: String, model: MLModel, reusing pooled: CVPixelBuffer?) throws -> [String: Any] {
        guard let description = model.modelDescription.outputDescriptionsByName[name] else {
            throw BackingError.message("Real-CUGAN model has no output named \(name)")
        }
        if let image = description.imageConstraint {
            if let pooled, buffer(pooled, matches: image) {
                let backing = [name: pooled as Any]
                try assertDirectBacking(backing)
                return backing
            }
            let backing = [name: try pixelBuffer(matching: image) as Any]
            try assertDirectBacking(backing)
            return backing
        }
        if let array = description.multiArrayConstraint {
            let backing: [String: Any] = [name: try MLMultiArray(shape: array.shape, dataType: array.dataType)]
            try assertDirectBacking(backing)
            return backing
        }
        throw BackingError.message("Real-CUGAN output \(name) cannot be backed")
    }

    static func assertDirectBacking(_ backing: [String: Any]) throws {
        if backing.isEmpty { throw BackingError.message("Real-CUGAN output backing is empty") }
        for (name, value) in backing {
            if value is MLFeatureValue {
                throw BackingError.message("output backing \(name) is an MLFeatureValue; Core ML requires a CVPixelBuffer or MLMultiArray")
            }
            if value is CVPixelBuffer || value is MLMultiArray { continue }
            throw BackingError.message("output backing \(name) is \(type(of: value)); Core ML requires a CVPixelBuffer or MLMultiArray")
        }
    }

    static func pixelBuffer(matching image: MLImageConstraint) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: image.pixelFormatType,
            kCVPixelBufferWidthKey as String: image.pixelsWide,
            kCVPixelBufferHeightKey as String: image.pixelsHigh,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, image.pixelsWide, image.pixelsHigh, image.pixelFormatType, attrs as CFDictionary, &buffer) == kCVReturnSuccess,
              let buffer else {
            throw BackingError.message("could not allocate a \(image.pixelsWide)x\(image.pixelsHigh) output backing")
        }
        return buffer
    }

    private static func buffer(_ buffer: CVPixelBuffer, matches image: MLImageConstraint) -> Bool {
        CVPixelBufferGetWidth(buffer) == image.pixelsWide
            && CVPixelBufferGetHeight(buffer) == image.pixelsHigh
            && CVPixelBufferGetPixelFormatType(buffer) == image.pixelFormatType
    }
}
