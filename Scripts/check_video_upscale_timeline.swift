import Foundation

func require(_ condition: Bool, _ message: String) {
    if !condition { fatalError("UPSCALE_ROUTE_CHECK_FAILED: \(message)") }
}

func sample(_ time: Double, _ engine: VideoUpscaleEngine?, _ confidence: Double = 0.9, cut: Bool = false) -> VideoUpscaleSample {
    VideoUpscaleSample(timeSeconds: time, engine: engine, confidence: confidence, reason: "fixture", sceneCutBefore: cut)
}

let mixed = VideoUpscalePlan.stabilized(duration: 12, samples: [
    sample(0, .realCUGAN), sample(1, .realCUGAN), sample(2, .realCUGAN),
    sample(3, .realCUGAN), sample(4, .realESRGAN, 0.9, cut: true), sample(5, .realESRGAN),
    sample(6, .realESRGAN), sample(7, .realESRGAN), sample(8, .realESRGAN),
    sample(9, .realCUGAN, 0.9, cut: true), sample(10, .realCUGAN), sample(11, .realCUGAN)
])
require(mixed.segments.count == 3, "mixed source should form three stable runs")
require(mixed.engine(at: 1.5) == .realCUGAN, "opening anime scene should use CUGAN")
require(mixed.engine(at: 5.5) == .realESRGAN, "movie CGI scene should use ESRGAN")
require(mixed.engine(at: 10.5) == .realCUGAN, "return to anime scene should use CUGAN")
require(mixed.usesAnimeOutline(at: 1.5), "anime scene should retain anime outline reconstruction")
require(!mixed.usesAnimeOutline(at: 5.5), "CGI scene should bypass anime outline reconstruction")
require(mixed.usesAnimeOutline(at: 10.5), "return to anime scene should restore anime outline reconstruction")
require(mixed.segments[0].startSeconds == 0 && mixed.segments[0].endSeconds == 4, "first scene boundary")
require(mixed.segments[1].startSeconds == 4 && mixed.segments[1].endSeconds == 9, "second scene boundary")

let flicker = VideoUpscalePlan.stabilized(duration: 8, samples: [
    sample(0, .realCUGAN), sample(1, .realCUGAN), sample(2, .realESRGAN),
    sample(3, .realCUGAN), sample(4, .realCUGAN), sample(5, .realESRGAN),
    sample(6, .realCUGAN), sample(7, .realCUGAN)
])
require(flicker.segments.count == 1, "isolated classification spikes must not switch the model")
require(flicker.engine(at: 5.5) == .realCUGAN, "stable engine should survive a one-window ESRGAN error")

let ambiguous = VideoUpscalePlan.stabilized(duration: 5, samples: [
    sample(0, nil, 0.2), sample(1, .realESRGAN, 0.6), sample(2, nil, 0.1),
    sample(3, .realESRGAN, 0.6), sample(4, nil, 0.1)
])
require(ambiguous.engine(at: 0.5) == .realCUGAN, "ambiguous opening must default to CUGAN")
require(ambiguous.segments.count == 1, "unconfirmed ESRGAN samples must not cause a switch")

let noSceneCut = VideoUpscalePlan.stabilized(duration: 6, samples: [
    sample(0, .realCUGAN), sample(1, .realCUGAN), sample(2, .realESRGAN),
    sample(3, .realESRGAN), sample(4, .realESRGAN), sample(5, .realESRGAN)
])
require(noSceneCut.segments.count == 1, "classifier agreement without a scene cut must not switch models")

let ambiguousCut = VideoUpscalePlan.stabilized(duration: 7, samples: [
    sample(0, .realCUGAN), sample(1, .realCUGAN), sample(2, nil, 0.2, cut: true),
    sample(3, .realESRGAN), sample(4, .realESRGAN), sample(5, .realESRGAN),
    sample(6, .realESRGAN)
])
require(ambiguousCut.engine(at: 3.5) == .realESRGAN, "confident samples shortly after an ambiguous cut should still switch")
require(ambiguousCut.segments.first?.endSeconds == 3, "the confirmed switch should be backdated to first confident post-cut sample")

let manual = VideoUpscalePlan.fixed(duration: 9, engine: .realESRGAN, reason: "manual")
require(manual.segments.count == 1 && manual.engine(at: 4.5) == .realESRGAN, "manual selection must remain fixed")
require(!manual.usesAnimeOutline(at: 4.5), "manual Real-ESRGAN must bypass anime outline reconstruction")
let manualCUGAN = VideoUpscalePlan.fixed(duration: 9, engine: .realCUGAN, reason: "manual")
require(manualCUGAN.usesAnimeOutline(at: 4.5), "manual Real-CUGAN must retain anime outline reconstruction")

print("UPSCALE_ROUTE_PASS: scene routing, outline policy, switch hysteresis, ambiguous fallback, manual overrides")
