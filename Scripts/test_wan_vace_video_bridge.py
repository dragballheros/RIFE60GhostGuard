#!/usr/bin/env python3
"""Validate that local Wan VACE video conditioning reaches the native VACE latent builder."""
from pathlib import Path

patch = Path("Scripts/patch_draw_things_vace_video_sequence.py").read_text()
mask_patch = Path("Scripts/patch_draw_things_vace_video_masks.py").read_text()
engine = Path("DrawThingsLocalVACEReconstructionEngine.swift").read_text()
view = Path("VACEVideoEditView.swift").read_text()
pipeline = Path("VACESelectiveVideoPipeline.swift").read_text()
workflow = Path(".github/workflows/build-unsigned-ipa.yml").read_text()

for label, marker in {
    "local-only rejection": "remoteExecutor == nil",
    "4n+1 temporal validation": "(sequence.frames.count - 1).isMultiple(of: 4)",
    "temporal image stack": "stackTemporalImages",
    "frame extraction": "public func frame(at index: Int)",
    "local model preparation": "prepareLocalWanVACE13B",
}.items():
    if marker not in patch:
        raise SystemExit(f"missing temporal bridge contract: {label}")

for label, marker in {
    "Float temporal video masks": "var videoMaskTensor: Tensor<FloatType>? = nil",
    "dedicated generation API": "generateWithVideoMask",
    "white-edit pixel mask condition": "maskFrames.full(1)",
    "inactive source video latents": "frames .* (1 - maskFrames)",
    "reactive masked video latents": "frames .* maskFrames",
    "official 64-channel VACE mask packing": "packVaceMaskTensor",
    "video-mask path bypasses reference-image Shuffle": "inputs: [], videoMask: vaceVideoMask",
}.items():
    if marker not in mask_patch:
        raise SystemExit(f"missing native mask-conditioning contract: {label}")

for label, marker in {
    "native local inference": "localPipeline.generate(",
    "explicit VACE Q8 control": "wan_v2.1_1.3b_vace_480p_q8p.ckpt",
    "source frame sequence": "MediaGenerationPipeline.VideoSequenceInput",
    "strict generated frame validation": "missingGeneratedFrames",
    "official VACE mask semantics": "VACE inpaint mask: white=edit",
}.items():
    if marker not in engine:
        raise SystemExit(f"missing live adapter contract: {label}")

for label, marker in {
    "source video picker": "PhotosPicker(selection: $videoItem, matching: .videos)",
    "mask picker": "PhotosPicker(selection: $maskItem, matching: .images)",
    "real video export": "WanVACEExports",
    "no substitute output on failure": "Stopped without exporting a substitute video.",
}.items():
    if marker not in view:
        raise SystemExit(f"missing editor workflow: {label}")

if "negativePrompt: String = \"\"" not in pipeline:
    raise SystemExit("negative prompt is not propagated to each temporal request")
for script_name in [
    "Scripts/patch_draw_things_vace_video_sequence.py",
    "Scripts/patch_draw_things_vace_video_masks.py",
]:
    if script_name not in workflow:
        raise SystemExit(f"workflow does not apply {script_name}")
print("Native Wan VACE temporal mask conditioning contracts are present.")
