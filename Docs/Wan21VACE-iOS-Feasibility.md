# Wan 2.1 VACE → iOS feasibility work

Status: **investigation branch; no iOS conversion or device execution is claimed yet.**

## Objective

Determine whether the official `Wan-AI/Wan2.1-VACE-1.3B` checkpoint can be converted into a fully local iPhone implementation for RIFE60GhostGuard, without any hosted inference endpoint. Do not add a selectable model to the production UI until the end-to-end backend is real and validated.

## Primary sources

- Official Wan 2.1 repository and checkpoint table: https://github.com/Wan-Video/Wan2.1
- Official VACE implementation and model table: https://github.com/ali-vilab/VACE
- Official 1.3B checkpoint: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
- Official 14B checkpoint (reference only): https://huggingface.co/Wan-AI/Wan2.1-VACE-14B

The official model documentation describes VACE as an all-in-one video creation/editing model. The 1.3B checkpoint is intended for approximately 480p output (the README notes 720p may be less stable). The published reference inference stack uses PyTorch and CUDA-specific setup. That establishes the reference environment, not that iOS conversion is impossible.

## Verified checkpoint-size finding (2026-10-15)

The official Hugging Face model repository currently reports about **19 GB** total and lists these core artifacts:

- `diffusion_pytorch_model.safetensors`: **7.15 GB**
- `models_t5_umt5-xxl-enc-bf16.pth`: **11.4 GB**
- `Wan2.1_VAE.pth`: **508 MB**

Source: https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/tree/main

This changes the first engineering question: do not attempt a full download onto the iPhone. The official repository as published is far larger than the phone's physical memory and a practical on-device model-storage budget. Quantizing weights could reduce storage, but would not by itself solve peak activation memory, unsupported ops, or runtime/thermal limits. The 11.4 GB T5 encoder is a particular investigation target: establish whether a smaller compatible text encoder can be used without unacceptable quality/conditioning changes, or whether the actual VACE pipeline requires that exact encoder. Any such substitution must be treated as a model variant and validated—not silently assumed equivalent.

A realistic first pass should audit only the model config, safetensors header, VAE architecture/config, and inference source, without downloading all checkpoint files. Then estimate memory from actual tensor metadata and test export of the smallest isolated component. No full checkpoint download or phone-side download should be triggered by this branch.

## Why this must be a separate video pipeline

The existing `WatermarkRemovalApplying.apply(CVPixelBuffer)` interface performs single-frame inpainting. VACE video-to-video and masked video-to-video workflows use a temporal clip, prompt, and task-specific conditioning/preprocessing. Adding VACE to `WatermarkModel` would misrepresent the feature and discard temporal context.

Keep the existing LaMa, DeepCream bar, and DeepCream mosaic paths untouched. The eventual VACE UI should be a separate, explicitly local prompt-driven video-editing option with input video, prompt, copy/paste affordances, progress/cancel, and conservative resource controls.

## Conversion investigation gates

1. **Artifact audit:** record the exact official checkpoint revision, license, file names, formats, sizes, shards, tensor keys, and dtypes. Never substitute an unofficial or hosted-only endpoint.
2. **Graph map:** identify the VACE conditioning path, temporal DiT, text encoder/tokenizer, video VAE, scheduler, and preprocessing annotators needed for the desired edit mode. Distinguish model weights from auxiliary annotator weights.
3. **Export spike:** attempt an export of one component at a time. Record every unsupported operator, dynamic-shape issue, precision limitation, and conversion tool/version. Do not claim a full conversion from a successful component export.
4. **Numerical validation:** run reference and converted inference on a tiny fixed clip/conditioning sample; compare intermediate tensors where possible and compare output quality. Add deterministic repeatability and temporal-consistency checks.
5. **Device feasibility:** evaluate peak resident memory and Metal allocation on the target A17 Pro / approximately 6 GB physical-memory class. Start with short clips and conservative 480p latent/frame sizes. Test FP16 first; consider lower-bit weights only after measuring quality. Weight streaming alone does not bound activations, KV/context tensors, VAE buffers, or temporary allocations.
6. **iOS integration:** only after the export and memory gates pass, implement a separate clip-level backend. Keep model files optional/downloadable rather than bloating the IPA, and verify checksums, available storage, cancellation cleanup, and low-memory failure handling.
7. **CI:** run static checks and conversion smoke tests on GitHub Actions where supported. CI success is not evidence of acceptable A17 Pro speed or thermals; device performance must be explicitly marked unverified until measured on hardware.

## Acceptance criteria before exposing the feature

- No network inference calls or API credentials.
- Uses the official checkpoint and license, with reproducible artifact hashes.
- Complete inference path exists; no placeholder selector or mocked success.
- Reference-vs-converted output validation passes on a small sample.
- Peak memory stays under a measured device budget with headroom for iOS and video buffers.
- Short-clip processing completes without crash/watchdog termination; progress, cancellation, and cleanup work.
- Existing watermark models, RIFE, upscaling, image studio, and release workflow remain unchanged.
- App UI labels limitations honestly (initially 480p/short clip if required).

## Current decision

Proceed with artifact and component audit. Do **not** add VACE to the existing per-frame watermark picker or ship an unverified runtime. If a critical component cannot be exported to a supported iOS runtime, record the blocker and test whether a smaller compatible model family is a viable alternative rather than silently switching to cloud inference.
