# Wan 2.1 VACE → iOS feasibility work

Status: **investigation branch; no iOS conversion or device execution is claimed yet.**

## Objective

Determine whether the official 720p `Wan-AI/Wan2.1-VACE-14B` checkpoint can be converted into a fully local iPhone implementation for RIFE60GhostGuard, without any hosted inference endpoint. The 14B/720p path is the target—not the 1.3B/480p path. Do not add a selectable model to the production UI until the end-to-end backend is real and validated.

## Primary sources

- Official Wan 2.1 repository and checkpoint table: https://github.com/Wan-Video/Wan2.1
- Official VACE implementation and model table: https://github.com/ali-vilab/VACE
- Official 1.3B checkpoint: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
- Official 14B checkpoint (reference only): https://huggingface.co/Wan-AI/Wan2.1-VACE-14B

The official model documentation describes VACE as an all-in-one video creation/editing model. **User requirement: target a 1080p final video, not 480p.** The official VACE model table lists 1.3B at about 480p (832×480) and 14B at about 720p (1280×720); neither official VACE checkpoint is documented as a native 1080p editing model. The base Wan VAE can encode/decode 1080p video, but that alone does not mean the VACE denoiser performs 1080p editing. Sources: https://github.com/ali-vilab/VACE and https://github.com/Wan-Video/Wan2.1. The published reference inference stack uses PyTorch and CUDA-specific setup. That establishes the reference environment, not that iOS conversion is impossible.

## Verified checkpoint-size finding (2026-10-15)

The official Hugging Face model repository currently reports about **19 GB** total and lists these core artifacts:

- `diffusion_pytorch_model.safetensors`: **7.15 GB**
- `models_t5_umt5-xxl-enc-bf16.pth`: **11.4 GB**
- `Wan2.1_VAE.pth`: **508 MB**

Source: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/tree/main

The public model `config.json` identifies the denoiser as `VaceWanModel` with 30 layers, hidden dimension 1536, 12 attention heads, FFN dimension 8960, and 15 configured VACE layers. These dimensions mean the activation footprint—not only the 7.15 GB transformer checkpoint—must be measured before choosing a temporal chunk length. Source: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/blob/main/config.json

This changes the first engineering question: do not attempt a full download onto the iPhone. The official repository as published is far larger than the phone's physical memory and a practical on-device model-storage budget. Quantizing weights could reduce storage, but would not by itself solve peak activation memory, unsupported ops, or runtime/thermal limits. The 11.4 GB T5 encoder is a particular investigation target: establish whether a smaller compatible text encoder can be used without unacceptable quality/conditioning changes, or whether the actual VACE pipeline requires that exact encoder. Any such substitution must be treated as a model variant and validated—not silently assumed equivalent.

A realistic first pass should audit only the model config, safetensors header, VAE architecture/config, and inference source, without downloading all checkpoint files. Then estimate memory from actual tensor metadata and test export of the smallest isolated component. No full checkpoint download or phone-side download should be triggered by this branch.

## 1080p output strategy (required)

The target deliverable is a final **1920×1080** video. Treat this as a hard output requirement, but do not label 480p/720p inference as native 1080p. The investigation must compare these possibilities:

- **Native 1080p VACE inference:** not supported by the official VACE model table and likely much more expensive in activation memory; do not assume it is feasible on A17 Pro.
- **Primary target — 720p VACE 14B editing then 1080p upscale:** this is the requested model path. Its large checkpoint and activation footprint are major blockers on an iPhone-class device; investigate quantization, bounded temporal windows, and component-wise export without pretending these techniques guarantee feasibility. Upscaling after 720p editing may soften or invent detail and must be quality-tested.
- **480p VACE 1.3B:** do not silently substitute this as the shipped target. It may be used only as an isolated tooling/debugging experiment if needed, and any user-facing downgrade requires explicit approval.

Use the app's existing local Real-CUGAN path as a candidate post-upscaler only after VACE inference can run. Compare 1080p output against the original clip for temporal flicker, line/detail retention, geometry changes, and frame consistency. Keep output frame rate, duration, and audio handling explicit. A 1920×1080 file container alone is not proof of native 1080p detail.

## Additional source-code findings

The official inference source imports `torch.cuda.amp`, and its CLI initializes CUDA devices for distributed execution. Its published requirements include `flash_attn` and `onnxruntime-gpu`. These are concrete portability blockers for a direct iOS port, not proof that a clean-room implementation is impossible. The denoiser also uses 3D convolution for VACE conditioning and repeated temporal/spatial attention blocks; the attention implementation and sequence padding will need a supported Metal/Core ML equivalent with bounded activation memory. Source files: https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/modules/model.py and https://github.com/ali-vilab/VACE/blob/main/requirements/framework.txt.

**Current technical conclusion:** a direct "convert the whole PyTorch model to one Core ML package" attempt is not a credible first move. A first fixed-shape VAE-encoder export spike is now implemented in `Scripts/probe_wan_vae_coreml.py`, with a dedicated macOS GitHub Actions workflow at `.github/workflows/wan-vae-coreml-probe.yml`. The workflow fetches only the official VAE weights (not the entire 19 GB checkpoint), pins the upstream Wan source revision, attempts conversion, and compares converted output against PyTorch. Passing a VAE-only test would not mean VACE editing works end-to-end; failure should be used to identify the exact unsupported operation or conversion blocker.

## Why this must be a separate video pipeline

The existing `WatermarkRemovalApplying.apply(CVPixelBuffer)` interface performs single-frame inpainting. VACE video-to-video and masked video-to-video workflows use a temporal clip, prompt, and task-specific conditioning/preprocessing. Adding VACE to `WatermarkModel` would misrepresent the feature and discard temporal context.

Keep the existing LaMa, DeepCream bar, and DeepCream mosaic paths untouched. The eventual VACE UI should be a separate, explicitly local prompt-driven video-editing option with input video, prompt, copy/paste affordances, progress/cancel, and conservative resource controls.

## Conversion investigation gates

1. **Artifact audit:** record the exact official checkpoint revision, license, file names, formats, sizes, shards, tensor keys, and dtypes. Never substitute an unofficial or hosted-only endpoint.
2. **Graph map:** identify the VACE conditioning path, temporal DiT, text encoder/tokenizer, video VAE, scheduler, and preprocessing annotators needed for the desired edit mode. Distinguish model weights from auxiliary annotator weights.
3. **Export spike:** attempt an export of one component at a time. Record every unsupported operator, dynamic-shape issue, precision limitation, and conversion tool/version. Do not claim a full conversion from a successful component export.
4. **Numerical validation:** run reference and converted inference on a tiny fixed clip/conditioning sample; compare intermediate tensors where possible and compare output quality. Add deterministic repeatability and temporal-consistency checks.
5. **Device feasibility:** evaluate peak resident memory and Metal allocation on the target A17 Pro / approximately 6 GB physical-memory class, with the 720p/14B path as the required target. Start with a small temporal window and quantized/component-wise experiments. If the 14B model cannot fit and run within measured device limits, report that as a blocker; do not silently substitute the 480p model. Test FP16 first where possible; consider lower-bit weights only after measuring quality. Weight streaming alone does not bound activations, KV/context tensors, VAE buffers, or temporary allocations.
6. **1080p quality gate:** require a 1920×1080 final export and compare native source, edited intermediate, and upscaled output. Record the effective inference resolution separately from export resolution; no claims of native 1080p quality unless the denoiser truly ran at that resolution.
7. **iOS integration:** only after the export and memory gates pass, implement a separate clip-level backend. Keep model files optional/downloadable rather than bloating the IPA, and verify checksums, available storage, cancellation cleanup, and low-memory failure handling.
8. **CI:** run static checks and conversion smoke tests on GitHub Actions where supported. CI success is not evidence of acceptable A17 Pro speed or thermals; device performance must be explicitly marked unverified until measured on hardware.

## Acceptance criteria before exposing the feature

- No network inference calls or API credentials.
- Uses the official checkpoint and license, with reproducible artifact hashes.
- Complete inference path exists; no placeholder selector or mocked success.
- Reference-vs-converted output validation passes on a small sample.
- Peak memory stays under a measured device budget with headroom for iOS and video buffers.
- Short-clip processing completes without crash/watchdog termination; progress, cancellation, and cleanup work.
- Existing watermark models, RIFE, upscaling, image studio, and release workflow remain unchanged.
- Final export is 1920×1080; the inference resolution is separately disclosed and the upscaled result passes temporal/detail checks.
- App UI labels any remaining limitations honestly; 480p/short-clip output is not considered fulfillment of the user's preferred 1080p target.

## Current decision

Proceed with the component export spike and audit. Do **not** add VACE to the existing per-frame watermark picker or ship an unverified runtime. If a critical component cannot be exported to a supported iOS runtime, record the blocker and test whether a smaller compatible model family is a viable alternative rather than silently switching to cloud inference.
