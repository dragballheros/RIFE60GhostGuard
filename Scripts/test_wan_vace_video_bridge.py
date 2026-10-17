#!/usr/bin/env python3
"""Static contract check for the actual temporal inference bridge in the IPA build."""
from pathlib import Path

patch = Path("Scripts/patch_draw_things_vace_video_sequence.py").read_text()
engine = Path("DrawThingsLocalVACEReconstructionEngine.swift").read_text()
view = Path("VACEVideoEditView.swift").read_text()
pipeline = Path("VACESelectiveVideoPipeline.swift").read_text()
workflow = Path(".github/workflows/build-unsigned-ipa.yml").read_text()

required = {
    "local-only rejection": "remoteExecutor == nil",
    "4n+1 sequence validation": "(sequence.frames.count - 1).isMultiple(of: 4)",
    "temporal image tensor stack": "stackTemporalImages",
    "temporal mask tensor stack": "stackTemporalMasks",
    "frame extraction": "public func frame(at index: Int)",
    "base model assets": "prepareLocalWanVACE13B",
}
for label, marker in required.items():
    if marker not in patch:
        raise SystemExit(f"missing temporal bridge contract: {label}")
for label, marker in {
    "actual local native generation": "localPipeline.generate(",
    "explicit VACE Q8 control": "wan_v2.1_1.3b_vace_480p_q8p.ckpt",
    "input source frame sequence": "MediaGenerationPipeline.VideoSequenceInput",
    "strict generated output validation": "missingGeneratedFrames",
    "mask inversion": "255 &- $0",
}.items():
    if marker not in engine:
        raise SystemExit(f"missing live adapter contract: {label}")
for label, marker in {
    "real video selection": "PhotosPicker(selection: $videoItem, matching: .videos)",
    "mask selection": "PhotosPicker(selection: $maskItem, matching: .images)",
    "exported result": "WanVACEExports",
    "no substitute output on failure": "Stopped without exporting a substitute video.",
}.items():
    if marker not in view:
        raise SystemExit(f"missing user live-test workflow: {label}")
if "negativePrompt: String = \"\"" not in pipeline:
    raise SystemExit("negative prompt is not propagated to each temporal request")
if "Scripts/patch_draw_things_vace_video_sequence.py" not in workflow:
    raise SystemExit("the pinned Draw Things checkout is not patched in the IPA workflow")
print("Wan VACE temporal generation bridge contracts are present.")
