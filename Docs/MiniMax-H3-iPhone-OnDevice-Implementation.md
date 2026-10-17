# MiniMax H3 on iPhone: implementation investigation

Status: **not yet an iOS runtime; do not expose H3 as a working editor option until the runtime gate below passes.**

## Product requirement

Keep the existing local Wan VACE path. Add MiniMax H3 as a separate model choice in the video-edit screen. The model chooser must be a visible list with a checkbox-style selection indicator for each row; no dropdown. The chosen model must dispatch to its own real backend. Inference must remain on the iPhone: no hosted API, remote ComfyUI, RunPod, or paid cloud.

The target device is iPhone 14 Pro Max (A17 Pro, approximately 5.5 GiB physical RAM class). The user has substantial free storage and accepts slow local inference. Storage capacity is not a substitute for working memory or an executable Metal/Core ML graph.

## Verified repository baseline

At the inspected baseline commit `4d84938527e1e03627e147bc3afaaa6f8c22e56d`:

- `VACEVideoEditView.swift` is the current local video-edit UI. It directly constructs `DrawThingsLocalVACEReconstructionEngine`; model choice is not yet abstracted.
- `DrawThingsLocalVACEReconstructionEngine.swift` runs the Draw Things native Metal/NNC Wan 2.1 1.3B Q8 model with a 1.3B VACE control checkpoint, 512x512 crops, five-frame windows, and 20 steps.
- `VACESelectiveVideoPipeline.swift` owns local temporal-window processing and video export.
- `VACEBlockStreamingRuntime.swift` loads separately exported Core ML packages one at a time. It is a generic component loader, not a MiniMax H3 transformer, encoder, VAE, or end-to-end pipeline.
- The existing app project vendors Draw Things and does not currently include a MiniMax H3 runtime or an H3-compatible quantized checkpoint.
- CI/Swift compilation and existing Wan component probes are not proof of a complete H3 edit on an iPhone.

## Official H3 architecture and size findings

Official sources checked 2026-10-17:

- Model repository: https://huggingface.co/MiniMaxAI/MiniMax-H3
- Diffusers API: https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3
- Model license: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE

The official Hugging Face repository reports about **498 GB** for the current repository. This is a repository size, not the RAM requirement of one selected workflow. The Diffusers documentation says H3 is a modular joint video-and-audio model with separate workflows (`t2va`, `fl2va`, and `ref2va`) and transformer partitions. Selecting a workflow avoids loading both transformer partitions, but the documented transformer partitions are still approximately **61.7 GB each**. The model also includes a Qwen3-VL-32B text/vision encoder and video/audio VAEs. The pipeline uses a packed sequence with joint text, image/video, and audio latents and two schedulers; this is not interchangeable with Wan VACE's architecture.

The official usage example is Python/PyTorch Diffusers and demonstrates `device_map="cuda"`; the docs do not provide an iOS/Core ML/Metal runtime or a validated Apple Neural Engine export. A direct dependency on Diffusers/PyTorch cannot be embedded into this Swift iOS application as-is. Also, the documented `t2va`, `fl2va`, and `ref2va` workflows describe text-to-video, first/last-keyframe, and reference-conditioned generation. The official docs do not document a drop-in white-mask/black-preserve video-inpainting workflow equivalent to the current Wan VACE UI. Matching the user's existing masked-edit experience may therefore require new conditioning/training or a different H3-compatible workflow, and must not be assumed just because H3 accepts video references.

Even a theoretical 4-bit quantization of a 61.7 GB transformer partition is roughly 15.4 GB of raw weights before quantization scales, non-quantized layers, encoder, VAE, activations, attention buffers, and runtime overhead. This calculation is only a lower bound; it is not a claim that a compatible 4-bit checkpoint or layer-streaming implementation exists. Streaming from disk can reduce resident weights but cannot remove the peak working set for an active block, packed sequence, encoder, VAE, or attention operation. No H3 model has yet been converted, quantized, executed, or parity-tested in this repository.

## Technical decision

Do not pretend that adding an H3 row, downloading the original checkpoint, or reusing `VACEBlockStreamingRuntime` makes H3 functional. The runtime loader accepts Core ML tensor components but does not implement H3's model semantics.

The first implementation spike should test a **single real H3 transformer block** and the H3 conditioning path, not a toy tensor or generic attention operation:

1. Pin an immutable upstream H3 revision and record exact model config, tensor names/shapes/dtypes, license terms, and workflow partition.
2. Identify a supported conversion route from the actual H3 block to an iOS-compatible Metal/Core ML graph. Confirm the required operators, rotary/positional encoding, packed multimodal sequence layout, and joint audio/video scheduler behavior.
3. Produce a low-bit, block-wise checkpoint format only if it preserves required tensor semantics and can be loaded without mapping the full partition into RAM. Include checksums, quantization metadata, and a fail-closed manifest.
4. Build a macOS reference harness against the pinned official implementation and compare deterministic intermediate tensors, then validate the exported block on Apple Silicon/Core ML. Host parity is necessary, not sufficient.
5. Measure peak resident memory and per-step latency for the actual block on the target iPhone. A generic Core ML test on CI cannot certify A17 Pro thermals, memory pressure, or watchdog safety.
6. Add H3-specific image/video conditioning, mask or reference handling, temporal/audio scheduling as supported by the chosen official workflow, VAE encode/decode, cancellation, and bounded intermediate storage.
7. Run a complete offline edit of a short source clip through the app and export a playable result. Record model revision, workflow, inference dimensions, frame count/rate, output dimensions, peak memory, elapsed time, thermal state, and whether audio is preserved.
8. Only after that gate passes should the UI show H3 as selectable and dispatch to its backend. Keep Wan VACE selectable and unchanged.

## UI acceptance criteria

- Visible model list, not a dropdown.
- One selected editing backend at a time; use a checkbox-style checkmark in each list row as requested, while enforcing single-selection semantics.
- The selected model name, workflow, and actual inference resolution must be visible.
- Wan VACE continues to use its current backend.
- H3 must not silently route to Wan VACE or a remote service.
- If the H3 backend is missing or fails to load, fail with a specific local diagnostic and never export a fake/substitute video.
- Do not label H3 "ready" based on compilation, a metadata probe, a converted isolated layer, or a mock output.

## License and authorization

The current public H3 Community License text excludes the United States, European Union, United Kingdom, and South Korea from its default applicable territory and says interested parties in excluded territories may contact MiniMax for a separate license. The user reports receiving an email authorizing TEXAS JUMBO PLATTER RESTAURANT CORP to use MiniMax H3 and MiniMax H3 Works, conditioned on continued compliance with the request representations. Preserve that email and obtain clarification from MiniMax if the intended use, derivative conversion/quantization, private sideloaded app, or redistribution is not explicitly covered. This repository note is not legal advice and does not infer rights beyond the actual authorization.

## Acceptance status

- [x] Existing Wan VACE implementation and UI entry point identified.
- [x] Official H3 repository, modular workflow model, approximate repository/transformer partition sizes, and license territory terms inspected.
- [ ] H3 block export and reference parity.
- [ ] Quantized/streamed iOS-compatible H3 weights.
- [ ] H3-specific multimodal conditioning and inference runtime.
- [ ] Target-device memory, thermal, and latency validation.
- [ ] End-to-end offline edit and video export on iPhone 14 Pro Max.
- [ ] Visible two-model list wired to two working backends.

Until the unchecked runtime and device gates pass, H3 is a feasibility investigation, not a functional feature.
