#!/usr/bin/env python3
"""Patch the pinned Draw Things engine to carry pixel masks through native Wan VACE conditioning.

The official VACE condition is not a per-frame still-image mask argument: it consists of
inactive video latents, reactive/masked video latents, and 64 packed mask channels. This
patch keeps the source video and mask sequence temporal from app input through VAE encoding.
"""
from pathlib import Path

ROOT = Path("Vendor/draw-things-community")
UTILITIES = ROOT / "Libraries/MediaGenerationKit/Sources/MediaGenerationExecutionUtilities.swift"
LOCAL_GENERATOR = ROOT / "Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift"
CONTROL_MODEL = ROOT / "Libraries/SwiftDiffusion/Sources/ControlModel.swift"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one mask-patch anchor, found {count}: {old[:100]!r}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def replace_n(path: Path, old: str, new: str, expected: int) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} mask-patch anchors, found {count}: {old[:100]!r}")
    path.write_text(source.replace(old, new), encoding="utf-8")


# The execution bridge must dispatch to the local concrete implementation that accepts videoMask.
replace_once(
    UTILITIES,
    "    generator: ImageGenerator,",
    "    generator: LocalImageGenerator,"
)

# Keep the mask out of the legacy single-frame inpainting argument. Convert the temporal masks
# to FloatType [N,H,W,1] and pass them through the explicit video-mask channel.
replace_once(
    UTILITIES,
    """      var inputTensor: Tensor<FloatType>? = nil
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
      }""",
    """      var inputTensor: Tensor<FloatType>? = nil
      var imageMask: Tensor<UInt8>? = nil
      var maskTensor: Tensor<UInt8>? = nil
      var videoMaskTensor: Tensor<FloatType>? = nil

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
        guard let temporalImage = stackTemporalImages(decodedFrames, width: targetWidth, height: targetHeight)
        else {
          completion(.failure(.generationFailed("Wan VACE source frames could not be stacked safely.")))
          return
        }
        var temporalVideoMask = Tensor<FloatType>(
          .CPU, .NHWC(decodedMasks.count, targetHeight, targetWidth, 1))
        temporalVideoMask.full(0)
        for (frameIndex, decodedMask) in decodedMasks.enumerated() {
          guard decodedMask.shape.count == 4, decodedMask.shape[0] == 1,
            decodedMask.shape[1] == targetHeight, decodedMask.shape[2] == targetWidth,
            decodedMask.shape[3] == 1
          else {
            completion(.failure(.generationFailed("A decoded Wan VACE mask had an invalid tensor shape.")))
            return
          }
          for y in 0..<targetHeight {
            for x in 0..<targetWidth {
              temporalVideoMask[frameIndex, y, x, 0] =
                decodedMask[0, y, x, 0] == 0 ? 0 : 1
            }
          }
        }
        inputTensor = temporalImage
        videoMaskTensor = temporalVideoMask
        // Do not pass an N-frame mask into the still-image mask argument.
        maskTensor = nil
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
      }"""
)

replace_once(
    UTILITIES,
    """        result = try generator.generate(
          trace: ImageGeneratorTrace(fromBridge: true),
          image: inputTensor,
          scaleFactor: 1,
          mask: maskTensor,
          hints: generatorHints,""",
    """        result = try generator.generateWithVideoMask(
          trace: ImageGeneratorTrace(fromBridge: true),
          image: inputTensor,
          scaleFactor: 1,
          mask: maskTensor,
          videoMask: videoMaskTensor,
          hints: generatorHints,"""
)

# Add the explicit entry point while preserving the ImageGenerator protocol's original signature.
replace_once(
    LOCAL_GENERATOR,
    """  // AVC shares the raw waveform and encoded conditioning across segment calls. Encoding happens
  // inside the generation branch, alongside image conditioning.
  func generate(""",
    """  /// On-device-only temporal VACE entry point. Ordinary still-image callers continue to use
  /// the unmodified ImageGenerator protocol method above.
  public func generateWithVideoMask(
    trace: ImageGeneratorTrace,
    image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    videoMask: Tensor<FloatType>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    fileMapping: [String: String], keywords: [String],
    cancellation: (@escaping () -> Void) -> Void,
    feedback: @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) -> Bool
  ) -> ([Tensor<FloatType>]?, [Tensor<Float>]?, Int) {
    generate(
      trace: trace, image: image, scaleFactor: scaleFactor, mask: mask,
      hints: hints, text: text, negativeText: negativeText, configuration: configuration,
      fileMapping: fileMapping, keywords: keywords, audioContext: nil, videoMask: videoMask,
      cancellation: cancellation, feedback: feedback
    )
  }

  // AVC shares the raw waveform and encoded conditioning across segment calls. Encoding happens
  // inside the generation branch, alongside image conditioning.
  func generate("""
)

replace_once(
    LOCAL_GENERATOR,
    """  func generate(
    trace: ImageGeneratorTrace,
    image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    fileMapping: [String: String], keywords: [String],
    audioContext: AudioConditioningContext?,
    cancellation: (@escaping () -> Void) -> Void,""",
    """  func generate(
    trace: ImageGeneratorTrace,
    image: Tensor<FloatType>?, scaleFactor: Int, mask: Tensor<UInt8>?,
    hints: [(ControlHintType, [(AnyTensor, Float)])],
    text: String, negativeText: String, configuration: GenerationConfiguration,
    fileMapping: [String: String], keywords: [String],
    audioContext: AudioConditioningContext?,
    videoMask: Tensor<FloatType>? = nil,
    cancellation: (@escaping () -> Void) -> Void,"""
)

# Make the per-frame mask available to the VACE ControlNet in every stage.
replace_once(
    LOCAL_GENERATOR,
    """    depth: DynamicGraph.Tensor<FloatType>?, hints: [ControlHintType: AnyTensor],
    custom: Tensor<FloatType>?, shuffles: [(Tensor<FloatType>, Float)], pose: Tensor<FloatType>?,
    mask: Tensor<FloatType>?,
    controls: [Control], version: ModelVersion, tiledDiffusion: TiledConfiguration,""",
    """    depth: DynamicGraph.Tensor<FloatType>?, hints: [ControlHintType: AnyTensor],
    custom: Tensor<FloatType>?, shuffles: [(Tensor<FloatType>, Float)], pose: Tensor<FloatType>?,
    mask: Tensor<FloatType>?, videoMask: Tensor<FloatType>? = nil,
    controls: [Control], version: ModelVersion, tiledDiffusion: TiledConfiguration,"""
)

# Keep all other generation modes source-compatible; only the main video inference
# call receives the mask. The optional parameter defaults to nil for all other callers.
replace_once(
    LOCAL_GENERATOR,
    """        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, pose: poses.last?.0,
        mask: nil, controls: configuration.controls, version: modelVersion,""",
    """        depth: depthImage, hints: hints, custom: custom, shuffles: shuffles, pose: poses.last?.0,
        mask: nil, videoMask: videoMask, controls: configuration.controls, version: modelVersion,"""
)

replace_once(
    LOCAL_GENERATOR,
    """      let type = ControlNetZoo.typeForModel(file)
      let isPreprocessorDownloaded = ControlNetZoo.preprocessorForModel(file).map {""",
    """      let type = ControlNetZoo.typeForModel(file)
      let vaceVideoMask = videoMask
      let isPreprocessorDownloaded = ControlNetZoo.preprocessorForModel(file).map {"""
)

replace_once(
    LOCAL_GENERATOR,
    """        case .shuffle:
          let isVACE = (version == .wan21_1_3b || version == .wan21_14b)
          guard !shuffles.isEmpty else {""",
    """        case .shuffle:
          let isVACE = (version == .wan21_1_3b || version == .wan21_14b)
          if isVACE, let vaceVideoMask {
            // For masked video-to-video, the full temporal condition belongs in VACE's
            // 96-channel latent path, not in a still-image Shuffle reference slot.
            let hints = controlModel.hint(
              batchSize: batchSize, startHeight: startHeight, startWidth: startWidth,
              image: image, graph: graph, inputs: [], videoMask: vaceVideoMask,
              cancellation: cancellation
            ).map { ($0, control.weight) }
            return (model: controlModel, hints: hints)
          }
          guard !shuffles.isEmpty else {"""
)

# Extend only the local ControlModel hint route. The default keeps all existing callers source-compatible.
replace_once(
    CONTROL_MODEL,
    """  public func hint(
    batchSize: Int, startHeight: Int, startWidth: Int, image: DynamicGraph.Tensor<FloatType>?,
    graph: DynamicGraph, inputs: [(hint: DynamicGraph.Tensor<FloatType>, weight: Float)],
    cancellation: (@escaping () -> Void) -> Void""",
    """  public func hint(
    batchSize: Int, startHeight: Int, startWidth: Int, image: DynamicGraph.Tensor<FloatType>?,
    graph: DynamicGraph, inputs: [(hint: DynamicGraph.Tensor<FloatType>, weight: Float)],
    videoMask: Tensor<FloatType>? = nil,
    cancellation: (@escaping () -> Void) -> Void"""
)

# Add the official VACE 8x8 pixel-mask packing transform (64 mask channels per latent frame).
replace_once(
    CONTROL_MODEL,
    """  public func hint(
    batchSize: Int, startHeight: Int, startWidth: Int, image: DynamicGraph.Tensor<FloatType>?,""",
    """  private func packVaceMaskTensor(
    _ mask: Tensor<FloatType>, latentDepth: Int, height: Int, width: Int
  ) -> Tensor<FloatType>? {
    let shape = mask.shape
    guard latentDepth > 0, shape.count == 4, shape[0] > 0, shape[3] == 1,
      shape[1] == height, shape[2] == width,
      height > 0, width > 0, height.isMultiple(of: 8), width.isMultiple(of: 8)
    else { return nil }
    var packed = Tensor<FloatType>(
      .CPU, .NHWC(latentDepth, height / 8, width / 8, 64))
    // Every packed element is assigned by the loops below; Tensor<FloatType> has no full() API.
    for latentFrame in 0..<latentDepth {
      // Matches the upstream VACE nearest-exact temporal resize.
      let sourceFrame = min(
        shape[0] - 1, ((2 * latentFrame + 1) * shape[0]) / (2 * latentDepth))
      for y in 0..<(height / 8) {
        for x in 0..<(width / 8) {
          for dy in 0..<8 {
            for dx in 0..<8 {
              let value = mask[sourceFrame, y * 8 + dy, x * 8 + dx, 0]
              packed[latentFrame, y, x, dy * 8 + dx] = value > 0.5 ? 1 : 0
            }
          }
        }
      }
    }
    return packed
  }

  public func hint(
    batchSize: Int, startHeight: Int, startWidth: Int, image: DynamicGraph.Tensor<FloatType>?,"""
)

replace_once(
    CONTROL_MODEL,
    """            var inactive: DynamicGraph.Tensor<FloatType>
            (inactive, encoder) = firstStage.encode(
              frames, encoder: encoder, cancellation: cancellation)
            guard !isCancelled.load(ordering: .acquiring) else {
              return [[]]
            }
            inactive = firstStage.scale(
              inactive[
                0..<(batchSize - refs.count), 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16
              ].copied())
            var reactive = inactive
            if image != nil {
              frames.full(0)
              (reactive, _) = firstStage.encode(
                frames, encoder: encoder, cancellation: cancellation)
              reactive = firstStage.scale(
                reactive[
                  0..<(batchSize - refs.count), 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16
                ].copied())
              guard !isCancelled.load(ordering: .acquiring) else {
                return [[]]
              }
            }
            encoder = nil
            var vaceLatents = graph.variable(
              .GPU(0), .NHWC(batchSize, startHeight / 8, startWidth / 8, 96), of: FloatType.self)
            vaceLatents.full(0)
            for (i, ref) in refs.enumerated() {
              let shape = ref.shape
              vaceLatents[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<16] = ref
              // Both mask and reactive are 0s.
            }
            vaceLatents[
              refs.count..<batchSize, 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16] =
              inactive
            vaceLatents[
              refs.count..<batchSize, 0..<(startHeight / 8), 0..<(startWidth / 8), 16..<32] =
              reactive
            // Inactive area set mask to 0. The rest are 1s.
            vaceLatents[
              (refs.count + (firstFrames > 0 ? (firstFrames - 1) / 4 + 1 : 0))..<batchSize,
              0..<(startHeight / 8), 0..<(startWidth / 8), 32..<96
            ].full(1)
            return [[vaceLatents]]""",
    """            var inactive: DynamicGraph.Tensor<FloatType>
            var reactive: DynamicGraph.Tensor<FloatType>
            var packedVideoMask: Tensor<FloatType>? = nil

            if let videoMask, firstFrames > 0 {
              let maskShape = videoMask.shape
              guard firstFrames <= frames.shape[0], maskShape.count == 4,
                maskShape[0] == firstFrames, maskShape[3] == 1
              else { return [[]] }

              var sourceMask = graph.variable(videoMask.toGPU(0))
              if maskShape[1] != startHeight || maskShape[2] != startWidth {
                sourceMask = Upsample(
                  .nearest,
                  widthScale: Float(startWidth) / Float(maskShape[2]),
                  heightScale: Float(startHeight) / Float(maskShape[1])
                )(sourceMask)
              }
              // Unknown/padded output frames are fully editable; the source interval uses its
              // actual per-pixel mask. White/1 means edit, black/0 means preserve.
              var maskFrames = graph.variable(
                .GPU(0), .NHWC(frames.shape[0], startHeight, startWidth, 1),
                of: FloatType.self)
              maskFrames.full(1)
              maskFrames[
                0..<firstFrames, 0..<startHeight, 0..<startWidth, 0..<1
              ] = sourceMask[
                0..<firstFrames, 0..<startHeight, 0..<startWidth, 0..<1
              ].copied()

              let inactiveFrames = frames .* (1 - maskFrames)
              let reactiveFrames = frames .* maskFrames
              (inactive, encoder) = firstStage.encode(
                inactiveFrames, encoder: encoder, cancellation: cancellation)
              guard !isCancelled.load(ordering: .acquiring) else {
                return [[]]
              }
              inactive = firstStage.scale(
                inactive[
                  0..<(batchSize - refs.count), 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16
                ].copied())
              (reactive, _) = firstStage.encode(
                reactiveFrames, encoder: encoder, cancellation: cancellation)
              reactive = firstStage.scale(
                reactive[
                  0..<(batchSize - refs.count), 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16
                ].copied())
              guard !isCancelled.load(ordering: .acquiring) else {
                return [[]]
              }
              packedVideoMask = packVaceMaskTensor(
                videoMask, latentDepth: batchSize - refs.count,
                height: startHeight, width: startWidth)
              guard packedVideoMask != nil else { return [[]] }
            } else {
              (inactive, encoder) = firstStage.encode(
                frames, encoder: encoder, cancellation: cancellation)
              guard !isCancelled.load(ordering: .acquiring) else {
                return [[]]
              }
              inactive = firstStage.scale(
                inactive[
                  0..<(batchSize - refs.count), 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16
                ].copied())
              reactive = inactive
              if image != nil {
                frames.full(0)
                (reactive, _) = firstStage.encode(
                  frames, encoder: encoder, cancellation: cancellation)
                reactive = firstStage.scale(
                  reactive[
                    0..<(batchSize - refs.count), 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16
                  ].copied())
                guard !isCancelled.load(ordering: .acquiring) else {
                  return [[]]
                }
              }
            }

            encoder = nil
            var vaceLatents = graph.variable(
              .GPU(0), .NHWC(batchSize, startHeight / 8, startWidth / 8, 96), of: FloatType.self)
            vaceLatents.full(0)
            for (i, ref) in refs.enumerated() {
              let shape = ref.shape
              vaceLatents[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<16] = ref
              // Reference rows are not editable video frames.
            }
            vaceLatents[
              refs.count..<batchSize, 0..<(startHeight / 8), 0..<(startWidth / 8), 0..<16] =
              inactive
            vaceLatents[
              refs.count..<batchSize, 0..<(startHeight / 8), 0..<(startWidth / 8), 16..<32] =
              reactive
            if let packedVideoMask {
              vaceLatents[
                refs.count..<batchSize, 0..<(startHeight / 8), 0..<(startWidth / 8), 32..<96
              ] = graph.variable(packedVideoMask.toGPU(0))
            } else {
              // Preserve Draw Things' established reference-to-video behavior when no edit mask
              // was provided by an MV2V sequence.
              vaceLatents[
                (refs.count + (firstFrames > 0 ? (firstFrames - 1) / 4 + 1 : 0))..<batchSize,
                0..<(startHeight / 8), 0..<(startWidth / 8), 32..<96
              ].full(1)
            }
            return [[vaceLatents]]"""
)

print("Applied the native temporal Wan VACE per-pixel mask conditioning patch.")
