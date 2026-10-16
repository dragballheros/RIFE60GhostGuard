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

## Implementation milestone 1: real VACE block conversion harness

Added `Scripts/probe_vace_block_coreml.py` and
`.github/workflows/wan-vace-block-coreml-parity.yml`.

The converter imports the pinned upstream `wan.modules.vace_model.VaceWanAttentionBlock`
and creates the published VACE 1.3B transformer width (1536), FFN width (8960),
and 12-head configuration without allocating all 30 transformer blocks. It
exercises the block's conditioning injection, normalization, self-attention,
text cross-attention, FFN, and skip-hint projection. It rewrites Wan's complex
rotary multiplication as real arithmetic and uses portable scaled-dot-product
attention in place of the upstream CUDA-only FlashAttention kernel.

The converter first checks its fixed-grid rotary rewrite against Wan's original
complex implementation, then traces/converts the real upstream block to Core ML,
runs a Core ML prediction, and records per-output max absolute error, mean
absolute error, RMSE, conversion time, package size, and parameter footprint.
The VACE before/after projections are explicitly given deterministic nonzero test
weights unless trained checkpoint tensors are supplied, so those operators are
exercised instead of passing trivially because upstream initializes them to zero.
With `--checkpoint`, the converter uses SafeTensors header metadata to materialize
only the selected `vace_blocks.0.*` tensors, verifies strict key/shape compatibility,
and loads no other checkpoint tensors. The checkpoint-backed lane is guarded by
an explicit `[vace-trained]` commit marker and verifies the pinned official
checkpoint's SHA-256 before loading it.

The CI workflow now covers an iOS 16-compatible INT8 model and a separate iOS 18
model for INT8/INT4 compression. INT4 requires iOS 18 Core ML support; the first
attempt on the iOS 16 target correctly reported that restriction rather than
silently claiming compatibility. Weight-compressed package size and numerical
error are measured against the same PyTorch reference. CI is pinned to the
upstream Wan2.1 source revision and uploads machine-readable validation reports.

Measured with the actual official 1.3B Diffusers checkpoint (SHA-256 verified),
the first VACE block loaded 31 learned tensors covering 51,162,368 parameters,
with no unmapped/ignored tensors. On the fixed four-token test, Float16 Core ML
conversion passed the configured PyTorch reference gate: conditioned-context
max absolute error 0.0521, RMSE 0.00523; skip-hint max absolute error 0.00388,
RMSE 0.00091. The FP16 package was 102,360,246 bytes.

On that same trained block, INT8 compression passed the configured float-reference
gate and reduced package size to 51,286,938 bytes (about 50% smaller). Its
conditioned-context max absolute error was 0.0593 / RMSE 0.00946; skip-hint
max absolute error was 0.00849 / RMSE 0.00189. Uniform INT4 reduced the package
to 25,723,286 bytes, but **failed numerical parity**: conditioned-context RMSE
was 0.1677 and skip-hint RMSE was 0.0413. Do not use uniform INT4 for this
trained block. This evidence supports continuing with INT8 as the current
candidate, and investigating mixed precision / selective quantization only
with a new error report. These are still four-token host tests, not realistic
sequence or iPhone accuracy measurements.

**Important scope boundary:** deterministic random weights remain the default
for the regular quick run, while the separate explicit trained-checkpoint CI
lane has now passed for the one-block Float16 and INT8 variants. Uniform INT4
failed the trained-weight numerical gate. The grid remains four latent tokens
with a short fixed text context. These results validate only one-block host
conversion and error under compression, not a production-ready VACE model,
an iPhone memory pass, or an end-to-end result.

## Experimental milestone: flexible sequence length and INT8 storage

On a separate experiment branch, the same VACE control-block graph was converted
with Core ML flexible token/context dimensions. With FP16 arithmetic, the graph
converted and executed across the tested shapes but failed a strict numerical
gate, so FP16 dynamic shape is not accepted. Changing the graph to FP32 passed
all 16 tested combinations (1 to 4 latent frames with a 4×4 spatial token grid,
and context lengths 1, 5, 8, and 16) with max absolute error about
`3.6e-7`. The FP32 block package was `204,639,660` bytes.

Post-training INT8 weight compression on that FP32 dynamic-shape package also
passed all 16 tests. It reduced package size to `51,363,477` bytes, with worst
maximum absolute errors about `0.00197` for conditioned context and
`0.00189` for skip hint; worst RMSE was about `3.78e-4` and `3.94e-4`.
This is promising component-level evidence for INT8 weights with FP32 activations,
but still uses deterministic random weights and a tiny 4×4 spatial grid. The
four-frame tests do not measure real 480p latent-grid attention memory. The
experiment workflow is separate from the main implementation branch and has not
yet been integrated into the app runtime.

## Implementation milestone 2: serial Core ML block runtime primitive

Added `VACEBlockStreamingRuntime.swift`, automatically included in the iOS target
by the existing XcodeGen `*.swift` source rule. The actor accepts locally stored
`.mlpackage` or `.mlmodelc` directories, compiles a package on first use, caches
the compiled artifact on local disk, and retains only one `MLModel` object at a
time. It validates requested tensor inputs against the model description, returns
tensor outputs with per-call load and prediction timings, and exposes explicit
eviction/cache-clear calls. Package paths and cache identifiers are validated
before file access.

A dedicated macOS CI job passes Swift parsing and Core ML typechecking for this
runtime. This checks the source API surface, not the full app build or iOS runtime
behavior. The actor cannot force Core ML/Metal to release every internal buffer,
and it is not yet connected to a VACE denoising coordinator. No on-device memory
or speed claim is made from this source check.

## Current truth

This branch contains executable VACE control-block and base-block converters,
strict learned-checkpoint mapping tests, quantization experiments, and a serial
Core ML runtime primitive that passes macOS and iOS 16 SDK typechecking. The
pinned trained-weight control block passed Float16 and INT8 parity; uniform INT4
failed and is rejected. A separate dynamic-shape experiment passed at FP32 and
with INT8 weight storage across a tiny test grid. A trained base-block pass is
currently being run. None of these component experiments is an end-to-end video
editor or an iPhone memory/performance result. The full model stack, mobile
text-conditioning solution, denoising schedule, temporal VAE, mask/reference
conditioning, end-to-end output, measured phone memory/thermal behavior and
existing-feature regression tests are still outstanding.

## Trained-checkpoint smoke test

The checkpoint-backed lane is intentionally opt-in because downloading the official 1.3B checkpoint is a multi-gigabyte CI task. A commit message containing `[vace-trained]` downloads the immutable checkpoint revision, verifies its SHA-256, and exercises only `vace_blocks.0`; it is a component test and not hosted inference.
