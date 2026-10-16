# Wan VACE iPhone memory-first implementation plan

Status: implementation branch; **not production-ready and not evidence of a complete on-device VACE pipeline**.

## Product decision

Do not try to ship the official Wan 2.1 VACE 14B checkpoint on iPhone 14 Pro Max. Its ideal uniform INT4 weight payload alone is about 7 GB decimal before scales, non-quantized tensors, activations, VAE, text conditioning, and iOS overhead. This is incompatible with a phone in the ~5.5 GB physical-memory class.

The implementation target is therefore a **separately identified, phone-oriented VACE variant**, not a claim that the official 14B model has been made to fit. The first candidate is the official VACE 1.3B architecture at its documented lower resolution, with quantized/streamed transformer blocks and a mobile-feasible text-conditioning path. This candidate is not presumed feasible: its published transformer and T5-XXL checkpoints also exceed the device's memory/storage budget and require component-level changes and validation.

User's export requirement remains **1920×1080 final video**. Lower-resolution inference followed by upscale must be labeled accurately and pass detail, geometry, and temporal-consistency tests. It must never be described as native 1080p VACE inference.

## Required implementation stages

### 1. Reproducible model and license audit

- Pin an immutable revision of the official VACE source and model repository.
- Audit checkpoint tensor shapes, dtypes, layer boundaries, non-transformer weights, and license.
- Do not download the full 14B model to the phone or bundle multi-gigabyte checkpoints into the IPA.
- Record exact hashes and model-variant identity in generated manifests.
- Fail closed if checkpoint keys, architecture configuration, or tensor counts do not match the pinned manifest.

### 2. Establish a device memory budget before conversion

Use a conservative app-level working-set target of **no more than 2.5 GiB** for the VACE job until actual device measurements justify a different ceiling. This is an engineering guardrail, not a guarantee that iOS will allow that allocation. Reserve additional headroom for iOS, Metal/Core ML allocations, decoded video frames, app UI, and thermal/runtime overhead.

Measure and report separately:
- resident/loaded quantized transformer block;
- peak attention and FFN activations;
- conditioning tensors and temporal context;
- text encoder and tokenizer;
- VAE encoder/decoder and frame buffers;
- Core ML / Metal allocations where measurable;
- total elapsed time, frames, resolution, and thermal state.

Do not infer runtime memory from checkpoint file size or theoretical quantized payload size.

### 3. Transformer execution prototype

- Do not convert the full transformer into one monolithic ML Program.
- Prototype a single real VACE block with the actual VACE conditioning/injection path, normalization, rotary embeddings, masks, and attention semantics.
- Validate the block numerically against the pinned official PyTorch reference on deterministic representative tensors.
- Test supported weight quantization formats on Apple silicon and iOS Core ML/Metal. Keep normalization and other numerically sensitive tensors at a validated precision.
- Only after parity passes, prototype block-wise loading/eviction. The runtime must prove that blocks can be loaded, executed, and released without retaining the full model or leaking memory.
- A generic SDPA kernel passing a micro-test is not sufficient evidence that VACE's attention path works.

### 4. Text conditioning

The official VACE 1.3B repository includes a T5-XXL encoder checkpoint that is itself far beyond the phone's available memory. Do not silently bundle or load it. Compare these options:
- a supported quantized/streamed T5 path, if measured peak memory and speed are viable;
- a smaller text encoder, treated as a distinct model variant and evaluated for prompt-conditioning quality;
- precomputed prompt embeddings only for a clearly constrained workflow, if it preserves the user's prompt-driven editing requirements.

No replacement is acceptable merely because it runs; record prompt adherence and visual-quality regressions. No hosted inference or paid cloud runtime is allowed.

### 5. VAE and temporal streaming

- Keep the official temporal VAE cache schedule, causal padding, and frame ordering intact.
- Validate both encoder and decoder against the official reference on representative short clips, not just one fixed tensor shape.
- Bound temporal chunk length and release intermediate tensors promptly.
- Add cancellation and cleanup paths for temporary files, compiled models, and partially decoded clips.
- Avoid holding the source clip, latent sequence, decoded output, and multiple full-resolution frame copies simultaneously.

### 6. End-to-end minimum proof

Before exposing a user-facing VACE editing control, run a complete local edit with:
- imported short source clip;
- prompt and required reference/mask conditioning;
- full denoising schedule;
- VAE decoding with temporal behavior;
- saved playable output;
- no network inference;
- memory, time, output-shape, and thermal diagnostics.

A component smoke test, successful model compilation, or mocked progress display does not satisfy this gate. A failed attempt must preserve the source and produce a useful diagnostic instead of crashing.

### 7. Resolution and final export

- Start with the smallest supported spatial resolution and shortest legal temporal window.
- Increase spatial and temporal dimensions only after end-to-end completion and memory headroom are measured.
- The likely first VACE 1.3B quality target is its documented 480p class, not 720p. Treat 720p as a later gate, not a promise.
- Use the existing local Real-CUGAN path for a 1920×1080 upscale only after VACE output quality and temporal consistency pass. Report both inference resolution and final export resolution.
- Validate line/detail retention, flicker, geometric drift, frame rate, duration, and audio preservation.

## CI and device gates

CI can validate scripts, model manifests, conversion, reference parity, and a host-side end-to-end fixture. It cannot certify iPhone memory, A17 Pro throughput, thermals, or watchdog safety. Keep those as explicit on-device gates.

The feature must remain marked experimental and must not be presented as a working video editor until:
1. actual VACE block conversion and reference parity pass;
2. temporal VAE parity passes;
3. a complete short edit runs locally;
4. peak memory stays below the measured safe device budget with headroom;
5. the app survives cancellation, low-memory conditions, and repeated runs;
6. the 1920×1080 export and quality checks pass;
7. the existing RIFE, Real-CUGAN, DeepCream, image, and recovery paths remain regression-free.

## Current truth

The existing VAE and attention probes are useful component experiments, but they do not implement the VACE denoiser, text conditioning, task conditioning, or end-to-end video editing. This branch starts the memory-first implementation path; it does not yet deliver a working VACE runtime or IPA. The next code milestone is the real VACE transformer-block conversion/parity harness, followed by the quantized block runtime and a full short-clip pipeline.
