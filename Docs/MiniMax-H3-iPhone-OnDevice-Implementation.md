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

## New implementation lead: native C/Metal `h3.c` (checked 2026-10-17)

A more relevant implementation candidate than MLX is https://github.com/antirez/h3.c. It is an MIT-licensed C/Objective-C Metal runtime with an explicit public C API (`h3_load_dir`, `h3_generate`, frame/progress callbacks) and SSD-streamed DiT weights. Its README reports about 2.0–2.1 GiB of tracked DiT tensor storage for low-memory runs on M5 Max; it explicitly warns that this is not total process memory and its tested/optimized systems are M3/M5 Macs. This is a stronger starting point than porting the Python/MLX stack, but it is **not evidence it works on iOS or A17 Pro**.

Useful source references:
- Runtime/API: https://github.com/antirez/h3.c/blob/main/h3.h
- Platform probe: https://github.com/antirez/h3.c/blob/main/h3_metal.m
- Runtime and SSD-streaming notes: https://github.com/antirez/h3.c
- MIT license: https://github.com/antirez/h3.c/blob/main/LICENSE

### Port feasibility blockers to solve in code

1. **iOS compilation and API availability:** the current native project uses Foundation, Metal, MetalPerformanceShaders, MetalPerformanceShadersGraph, Accelerate and ICU, and the platform probe contains macOS-specific availability assumptions. Build a separate iOS-targeted compatibility layer; do not assume a macOS Makefile binary can be linked into the app unchanged.
2. **Media I/O:** the runtime's current media path relies on FFmpeg/FFprobe executables and file paths. An iOS app cannot assume a shell, external FFmpeg binary, or unrestricted subprocess execution. The integration needs AVFoundation input/output adapters or a deliberate, license-compatible embedded media implementation. The public frame callback is a promising output seam, but video reference conditioning still needs a supported way to decode and present temporal references to H3.
3. **Memory and thermal gate:** SSD streaming reduces DiT residency, not all model/process memory. H3's Qwen3-VL text/vision encoder and both VAEs run in separate phases but still need to fit individually alongside the app and iOS system. Measure peak resident memory, Metal allocations, jetsam margin and latency on the target iPhone; Mac results cannot certify this.
4. **Editing semantics:** `h3.c` supports prompt-to-video, first/last-frame conditioning, and ordered image/video/audio references. It does not expose Wan VACE-style per-pixel inpainting masks. A first H3 integration can be a distinct reference-conditioned video-to-video workflow, but must not claim exact masked-region preservation unless a mask-conditioned H3 path is implemented and validated.
5. **Model distribution:** keep model weights out of the IPA. Implement resumable, checksum-verified, user-initiated downloads into app storage, with enough disk-space checks and cleanup controls. Select only the required FL2VA/Ref2VA components for the chosen workflow, not the entire 498 GB repository. Verify the user's separate MiniMax authorization covers the specific checkpoint variants, conversion/derivative weights, and private sideloaded distribution.

### Revised implementation sequence

- [ ] Pin the upstream `h3.c` commit and vendor/license notices in a dedicated third-party source directory.
- [ ] Add an iOS compile target/CI audit for its C and Objective-C runtime, then fix platform-specific compile errors without breaking its macOS build.
- [ ] Replace subprocess/FFmpeg media dependencies with AVFoundation adapters or an embedded, audited media layer.
- [ ] Port model-path access to the app's sandbox and add resumable, verified model downloads.
- [ ] Add an H3 runtime bridge with cancellation/progress/frame callbacks and fail-closed memory checks.
- [ ] Test model metadata and one real short generation on the iPhone before wiring H3 into the editor.
- [ ] Add a separate H3 reference-video workflow in the model list, with a checkbox-style single-selection indicator; retain Wan VACE's mask-based workflow.
- [ ] Only claim mask editing if H3's output can demonstrably respect the selected mask. Otherwise label the workflow accurately as reference-conditioned video regeneration.
- [ ] Validate an offline end-to-end clip, audio/video decode, memory peak, latency, thermal state, and export on the iPhone 14 Pro Max.

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
- [x] Native `antirez/h3.c` runtime identified and pinned for a compile-only iOS SDK audit.
- [x] Added `Scripts/audit_h3_ios_portability.sh` and `.github/workflows/minimax-h3-ios-port-audit.yml`; audit result has not yet been verified here.
- [ ] iOS SDK compile audit passes without source compatibility patches.
- [ ] H3 block export and reference parity.
- [ ] Quantized/streamed iOS-compatible H3 weights.
- [ ] H3-specific multimodal conditioning and inference runtime.
- [ ] Target-device memory, thermal, and latency validation.
- [ ] End-to-end offline edit and video export on iPhone 14 Pro Max.
- [ ] Visible two-model list wired to two working backends.

Until the unchecked runtime and device gates pass, H3 is a feasibility investigation, not a functional feature.


## Follow-up research: native AVFoundation integration candidate (2026-10-17)

A second project, [JunzoKamahara/H3cApp](https://github.com/JunzoKamahara/H3cApp), is a native macOS application built around an H3 C/Metal engine. Its published project description states that media I/O uses AVFoundation/ImageIO rather than requiring an external FFmpeg executable for the app's media path. This is useful evidence that the H3 engine can be separated from its command-line media shell, but the application is explicitly macOS-only and is not an iOS port. Its model notes also describe FL2VA alone as roughly 134 GiB and FL2VA plus Ref2VA as roughly 268 GiB; those figures are checkpoint storage, not a claim that the iPhone can hold or execute the model.

This changes the next engineering step: before writing app integration code, compare the fork's media boundary and upstream H3 engine changes against the pinned revision in `Scripts/audit_h3_ios_portability.sh`. Do not copy the whole macOS app into this iOS repository. Extract only portable inference code and the AVFoundation design where APIs are available on iOS, then compile and link an iOS-targeted static library with an explicit source list. The current compile audit is only a first probe; it does not yet validate linking, API availability at runtime, model compatibility, or device memory.

The GitHub connector did not return any PR-triggered workflow runs for head commit `d309b818ced3806158a4a9018695e4acf69bcd56`. Because that connector only returns PR-triggered runs, this result does **not** establish that the push-triggered audit did not run. The audit status remains **unverified** until the specific Actions run is inspected.


## Compile-audit result and compatibility findings (2026-10-17 16:20 UTC)

The pinned-source audit run `39048984560` completed successfully: all listed C and Objective-C translation units compiled against the iOS SDK. This confirms source-level compilation only. The log includes availability warnings for APIs introduced after the declared iOS 16.0 deployment target, including MPSGraph 3D convolution APIs introduced in iOS 16.3 and scaled-dot-product attention introduced in iOS 18.0. Therefore, a successful compile does **not** establish safe execution on the app's iOS 16 minimum. Before integration, either add correct runtime availability checks/fallback implementations for every newer API or define and enforce an H3-specific minimum OS while keeping the rest of the app's deployment target unchanged. Do not silence these warnings globally.

The audit currently compiles translation units independently; it does not create `libh3.a`, resolve all symbols, or link the framework dependencies into an iOS app. It also compiles the pinned upstream revision, not the newer H3cApp fork. The next native-engine gate is a reproducible iOS static-library link probe with explicit source and framework lists, followed by an availability audit and AVFoundation-only media adapter review. This is still not target-device inference validation.

The H3cApp fork is explicitly a macOS 15+ application. Its README confirms that its current attention path uses macOS 15 APIs and its media layer uses AVFoundation/ImageIO. This is a useful source of an AVFoundation adapter design, but its stated platform requirement and newer fork revision must be accounted for before reusing code. The H3cApp checkpoint sizes are large (approximately 134 GiB for FL2VA and 268 GiB for FL2VA+Ref2VA); these are model-storage figures and do not establish that those complete checkpoints are practical on the target iPhone.


## Audit refinement: AVFoundation engine fork (2026-10-17)

The portability script now targets pinned H3cApp commit `826fafffc84a15283cb7b3f4c234e7aedf07516c`, not the older upstream command-line engine. This fork replaces the FFmpeg media shell with native `h3_av_reader.m` and `h3_av_writer.m`, so it is a better candidate for the app's on-device media flow. The audit source list includes the fork's LoRA and attention-cache modules and its AVFoundation reader/writer, while intentionally excluding the macOS app UI and CLI.

This is a portability experiment, not yet a vendored runtime integration. The H3cApp project currently declares macOS 15 as its platform and uses APIs newer than this app's iOS 16 deployment target. The audit should be treated as failed if its link probe fails; if it passes, the next gates remain API availability guards/fallbacks, integration into the iOS XcodeGen target, model layout/downloading, and a device execution test. Do not add a selectable H3 model option until a real engine adapter is linked into the app and can run.

## Updated native-port audit result (2026-10-17 17:58 UTC)

The GitHub Actions run for commit `2a003c012c2c6739ec7e289ed4d288d7b51b0647` passed:
https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/39054454374

It ran both:
1. `Scripts/test_patch_h3_ios_availability.py` (guard insertion, idempotence, and fail-closed behavior for missing/ambiguous source signatures).
2. `Scripts/audit_h3_ios_portability.sh` (compiles the selected native C/Objective-C translation units for arm64 iOS and force-links the static archive).

This is a portability result only. The workflow compiles a temporary checkout of the pinned H3cApp fork with a CI-applied availability patch; it does not compile H3 into the app target, execute inference, or verify behavior on an iPhone.

## Memory gate before application integration

The pinned H3 API's cache documentation states that retaining the DiT cache holds approximately 1.5 GiB more between generations and retaining the video decoder cache holds approximately 2.7 GiB. Those are cache-retention costs, not total end-to-end peak memory. Together they are approximately 4.2 GiB before the app, encoder, audio/video conditioning, transient Metal buffers, allocator fragmentation, and iOS itself. Therefore the iPhone integration must leave interactive caches disabled by default and must not enable DIT/decoder caching merely to improve repeat-generation speed.

The native engine reports device physical memory, recommended working set, and Metal maximum buffer length via `h3_device_info`; its model inspection API reports component sizes. The next runtime milestone must collect these values before generation and refuse unsafe profiles, then measure peak process/Metal memory on an actual target device. A Mac compile/link result cannot substitute for that measurement.

## Next implementation boundary

Do not yet add H3 as an enabled backend in `VACEVideoEditView`. That screen currently requires a `VACERegionReconstructionEngine` which produces masked, source-sized five-frame patches. H3's `h3_generate` API instead produces a complete generated clip from its own text/reference inputs and does not accept the VACE per-pixel edit mask. Wiring H3 into the existing protocol would either be semantically incorrect or require a mask-conditioned H3 implementation that has not been built.

The next correct step is a separate H3 engine adapter and workflow, preserving the Wan VACE editor untouched: first integrate the native runtime and AVFoundation media adapters, then expose a distinct reference-conditioned generation path, with explicit disclosure that it does not provide Wan VACE's exact mask-preservation semantics. Only after real iPhone memory and inference tests should the H3 action become enabled in the UI.
