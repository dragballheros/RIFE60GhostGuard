#!/usr/bin/env python3
"""Patch the pinned Draw Things checkout with explicit local temporal Wan VACE inputs."""
from pathlib import Path

ROOT = Path("Vendor/draw-things-community")
PIPELINE = ROOT / "Libraries/MediaGenerationKit/Sources/MediaGenerationPipeline.swift"
SUPPORT = ROOT / "Libraries/MediaGenerationKit/Sources/MediaGenerationExecutionSupport.swift"
UTILITIES = ROOT / "Libraries/MediaGenerationKit/Sources/MediaGenerationExecutionUtilities.swift"

def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one patch anchor, found {count}: {old[:100]!r}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")

def replace_n(path: Path, old: str, new: str, expected: int) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} patch anchors, found {count}: {old[:100]!r}")
    path.write_text(source.replace(old, new), encoding="utf-8")

replace_once(SUPPORT, """struct MediaGenerationExecutionInputs {
  let image: Data?
  let mask: Data?
  let hints: [MediaGenerationExecutionHint]
}""", """struct MediaGenerationExecutionInputs {
  let image: Data?
  let mask: Data?
  let hints: [MediaGenerationExecutionHint]
  let videoFrames: [Data]?
  let videoMasks: [Data]?
}""")

replace_once(PIPELINE, """  public enum InputRole: Sendable, Equatable {
    case image
    case mask
    case moodboard
    case depth
  }""", """  public enum InputRole: Sendable, Equatable {
    case image
    case mask
    case moodboard
    case depth
    case videoSequence
  }""")

replace_once(PIPELINE, """  /// Wraps an image input as an inpainting mask.
  public struct MaskInput: MediaGenerationInput {""", """  /// A temporal source-video sequence with a matching per-frame edit mask.
  /// Wan VAE temporal sequences use 4n+1 frames; this is local-only input.
  public struct VideoSequenceInput: MediaGenerationInput {
    public let frames: [Data]
    public let masks: [Data]

    public init(frames: [Data], masks: [Data]) {
      self.frames = frames
      self.masks = masks
    }

    public var role: InputRole {
      .videoSequence
    }
  }

  /// Wraps an image input as an inpainting mask.
  public struct MaskInput: MediaGenerationInput {""")

replace_once(PIPELINE, """    public var height: Int {
      MediaGenerationImageCodec.height(for: tensor)
    }

    internal init(tensor: Tensor<FloatType>) {""", """    public var height: Int {
      MediaGenerationImageCodec.height(for: tensor)
    }

    /// Number of frames in a temporal result, or one for a still image.
    public var frameCount: Int {
      tensor.shape.count == 4 ? tensor.shape[0] : 1
    }

    /// Returns one generated frame, or nil when the index is outside the result.
    public func frame(at index: Int) -> Result? {
      let shape = tensor.shape
      if shape.count != 4 {
        return index == 0 ? self : nil
      }
      guard index >= 0, index < shape[0],
        shape[1] > 0, shape[2] > 0, shape[3] > 0
      else { return nil }
      return Result(
        tensor: tensor[
          index..<(index + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
      )
    }

    internal init(tensor: Tensor<FloatType>) {""")

replace_once(PIPELINE, """      let runtimeConfiguration = try configuration.runtimeConfiguration(
        template: recommendedTemplate)
      let previewModelVersion = ModelZoo.versionForModel(""", """      let runtimeConfiguration = try configuration.runtimeConfiguration(
        template: recommendedTemplate)
      if executionInputs.videoFrames != nil {
        guard remoteExecutor == nil else {
          throw MediaGenerationKitError.generationFailed(
            "Wan VACE temporal video editing is local-only; remote and cloud backends are disabled.")
        }
        let videoModelVersion = ModelZoo.versionForModel(
          runtimeConfiguration.model ?? resolvedModel
        )
        guard videoModelVersion == .wan21_1_3b || videoModelVersion == .wan21_14b else {
          throw MediaGenerationKitError.generationFailed(
            "A temporal video sequence can only be used with a Wan 2.1 video model.")
        }
      }
      let previewModelVersion = ModelZoo.versionForModel(""")

replace_once(PIPELINE, """                  image: executionInputs.image,
                  mask: executionInputs.mask,
                  hints: executionInputs.hints,
                  cancellationBridge: cancellationBridge,""", """                  image: executionInputs.image,
                  mask: executionInputs.mask,
                  hints: executionInputs.hints,
                  videoFrames: executionInputs.videoFrames,
                  videoMasks: executionInputs.videoMasks,
                  cancellationBridge: cancellationBridge,""")

replace_once(PIPELINE, """    var image: Data?
    var mask: Data?
    var moodboardImages: [MediaGenerationExecutionHintImage] = []
    var depthImages: [MediaGenerationExecutionHintImage] = []

    for input in inputs {
      switch input.role {
      case .image:
        guard image == nil else {""", """    var image: Data?
    var mask: Data?
    var videoFrames: [Data]?
    var videoMasks: [Data]?
    var moodboardImages: [MediaGenerationExecutionHintImage] = []
    var depthImages: [MediaGenerationExecutionHintImage] = []

    for input in inputs {
      switch input.role {
      case .videoSequence:
        guard videoFrames == nil, image == nil, mask == nil else {
          throw MediaGenerationKitError.generationFailed(
            "A temporal video sequence cannot be combined with still-image inputs.")
        }
        guard let sequence = input as? VideoSequenceInput else {
          throw MediaGenerationKitError.generationFailed("Invalid temporal video sequence input type.")
        }
        guard sequence.frames.count >= 5,
          (sequence.frames.count - 1).isMultiple(of: 4),
          sequence.frames.count == sequence.masks.count
        else {
          throw MediaGenerationKitError.generationFailed(
            "Wan VACE requires matching source frames and masks in a 4n+1 sequence (at least five frames).")
        }
        videoFrames = sequence.frames
        videoMasks = sequence.masks
      case .image:
        guard videoFrames == nil else {
          throw MediaGenerationKitError.generationFailed(
            "A temporal video sequence cannot be combined with still-image inputs.")
        }
        guard image == nil else {""")

replace_once(PIPELINE, """      case .mask:
        guard mask == nil else {""", """      case .mask:
        guard videoFrames == nil else {
          throw MediaGenerationKitError.generationFailed(
            "A temporal video sequence already contains per-frame masks.")
        }
        guard mask == nil else {""")

replace_once(PIPELINE, """      case .moodboard:
        moodboardImages.append(""", """      case .moodboard:
        guard videoFrames == nil else {
          throw MediaGenerationKitError.generationFailed(
            "Wan VACE sequence inputs cannot be mixed with still-image reference hints.")
        }
        moodboardImages.append(""")
replace_once(PIPELINE, """      case .depth:
        depthImages.append(""", """      case .depth:
        guard videoFrames == nil else {
          throw MediaGenerationKitError.generationFailed(
            "Wan VACE sequence inputs cannot be mixed with still-image depth hints.")
        }
        depthImages.append(""")
replace_once(PIPELINE, """    return MediaGenerationExecutionInputs(image: image, mask: mask, hints: hints)
  }

  fileprivate static func encodedImageData(from input: Input) throws -> Data {""", """    return MediaGenerationExecutionInputs(
      image: image, mask: mask, hints: hints,
      videoFrames: videoFrames, videoMasks: videoMasks
    )
  }

  fileprivate static func encodedImageData(from input: Input) throws -> Data {""")
replace_once(PIPELINE, """  public static func file(_ path: String) -> FileInput {
    FileInput(path)
  }

  /// Builds a pipeline from a model reference and backend.""", """  public static func file(_ path: String) -> FileInput {
    FileInput(path)
  }

  /// Downloads/checks model assets as local files; inference remains on-device.
  public func prepareLocalWanVACE13B(
    progress: (@Sendable (String, Int64, Int64) -> Void)? = nil
  ) async throws {
    guard case .local = backend else {
      throw MediaGenerationKitError.generationFailed(
        "Wan VACE video editing requires the local inference backend.")
    }
    guard ModelZoo.versionForModel(storage.resolvedModel) == .wan21_1_3b else {
      throw MediaGenerationKitError.generationFailed(
        "This explicitly selected experimental profile requires Wan 2.1 1.3B.")
    }
    try await storage.environment.storage.ensureModelReady(
      file: storage.resolvedModel,
      includeDependencies: true,
      progress: { item in progress?(item.file, item.bytesWritten, item.totalBytesExpected) }
    )
    try await storage.environment.storage.ensureModelReady(
      file: "wan_v2.1_1.3b_vace_480p_q8p.ckpt",
      includeDependencies: false,
      progress: { item in progress?(item.file, item.bytesWritten, item.totalBytesExpected) }
    )
  }

  /// Builds a pipeline from a model reference and backend.""")
replace_n(UTILITIES, """    configuration: GenerationConfiguration,
    image: Data? = nil,
    mask: Data? = nil,
    hints: [MediaGenerationExecutionHint] = [],
    cancellationBridge: MediaGenerationCancellationBridge?,""", """    configuration: GenerationConfiguration,
    image: Data? = nil,
    mask: Data? = nil,
    hints: [MediaGenerationExecutionHint] = [],
    videoFrames: [Data]? = nil,
    videoMasks: [Data]? = nil,
    cancellationBridge: MediaGenerationCancellationBridge?,""", expected=2)

replace_once(UTILITIES, """      configuration: configuration,
      image: image,
      mask: mask,
      hints: hints,
      cancellationBridge: cancellationBridge,
      feedback: feedback,""", """      configuration: configuration,
      image: image,
      mask: mask,
      hints: hints,
      videoFrames: videoFrames,
      videoMasks: videoMasks,
      cancellationBridge: cancellationBridge,
      feedback: feedback,""")

replace_once(UTILITIES, """      let targetWidth = Int(configuration.startWidth) * 64
      let targetHeight = Int(configuration.startHeight) * 64

      var inputTensor: Tensor<FloatType>? = nil
      var imageMask: Tensor<UInt8>? = nil
      if let imageData = image {
        let (tensor, autoMask) = imageDataToTensor(
          imageData, width: targetWidth, height: targetHeight)
        inputTensor = tensor
        imageMask = autoMask
      }

      let maskTensor: Tensor<UInt8>? =
        mask.flatMap {
          maskDataToTensor($0, width: targetWidth, height: targetHeight)
        } ?? imageMask""", """      let targetWidth = Int(configuration.startWidth) * 64
      let targetHeight = Int(configuration.startHeight) * 64

      var inputTensor: Tensor<FloatType>? = nil
      var imageMask: Tensor<UInt8>? = nil
      var maskTensor: Tensor<UInt8>? = nil

      if let videoFrames {
        guard let videoMasks, videoFrames.count >= 5,
          (videoFrames.count - 1).isMultiple(of: 4),
          videoFrames.count == videoMasks.count, image == nil, mask == nil
        else {
          completion(.failure(.generationFailed(
            "Invalid Wan VACE input: every temporal source frame must have a matching mask.")))
          return
        }
        var decodedFrames = [Tensor<FloatType>]()
        var decodedMasks = [Tensor<UInt8>]()
        decodedFrames.reserveCapacity(videoFrames.count)
        decodedMasks.reserveCapacity(videoMasks.count)
        for frameData in videoFrames {
          let (tensor, _) = imageDataToTensor(frameData, width: targetWidth, height: targetHeight)
          guard let tensor else {
            completion(.failure(.generationFailed(
              "A Wan VACE source frame could not be decoded at the requested inference size.")))
            return
          }
          decodedFrames.append(tensor)
        }
        for maskData in videoMasks {
          guard let tensor = maskDataToTensor(maskData, width: targetWidth, height: targetHeight) else {
            completion(.failure(.generationFailed(
              "A Wan VACE source mask could not be decoded at the requested inference size.")))
            return
          }
          decodedMasks.append(tensor)
        }
        guard let temporalImage = stackTemporalImages(decodedFrames, width: targetWidth, height: targetHeight),
          let temporalMask = stackTemporalMasks(decodedMasks, width: targetWidth, height: targetHeight)
        else {
          completion(.failure(.generationFailed("Wan VACE temporal tensors could not be assembled safely.")))
          return
        }
        inputTensor = temporalImage
        maskTensor = temporalMask
      } else {
        if let imageData = image {
          let (tensor, autoMask) = imageDataToTensor(
            imageData, width: targetWidth, height: targetHeight)
          inputTensor = tensor
          imageMask = autoMask
        }
        maskTensor = mask.flatMap {
          maskDataToTensor($0, width: targetWidth, height: targetHeight)
        } ?? imageMask
      }""")

replace_once(UTILITIES, """  private static func hintImageDataToTensor(_ data: Data) -> Tensor<FloatType>? {""", """  private static func stackTemporalImages(
    _ frames: [Tensor<FloatType>], width: Int, height: Int
  ) -> Tensor<FloatType>? {
    guard !frames.isEmpty else { return nil }
    var output = Tensor<FloatType>(.CPU, .NHWC(frames.count, height, width, 3))
    // Every output element is assigned by the validated frame loop below.
    for (index, frame) in frames.enumerated() {
      let shape = frame.shape
      guard shape.count == 4, shape[0] == 1, shape[1] == height,
        shape[2] == width, shape[3] >= 3
      else { return nil }
      output[index..<(index + 1), 0..<height, 0..<width, 0..<3] =
        frame[0..<1, 0..<height, 0..<width, 0..<3].copied()
    }
    return output
  }

  private static func stackTemporalMasks(
    _ masks: [Tensor<UInt8>], width: Int, height: Int
  ) -> Tensor<UInt8>? {
    guard !masks.isEmpty else { return nil }
    var output = Tensor<UInt8>(.CPU, .NHWC(masks.count, height, width, 1))
    // Every output element is assigned by the validated mask loop below.
    for (index, mask) in masks.enumerated() {
      let shape = mask.shape
      guard shape.count == 4, shape[0] == 1, shape[1] == height,
        shape[2] == width, shape[3] == 1
      else { return nil }
      output[index..<(index + 1), 0..<height, 0..<width, 0..<1] =
        mask[0..<1, 0..<height, 0..<width, 0..<1].copied()
    }
    return output
  }

  private static func hintImageDataToTensor(_ data: Data) -> Tensor<FloatType>? {""")

print("Applied the local temporal-sequence patch.")
