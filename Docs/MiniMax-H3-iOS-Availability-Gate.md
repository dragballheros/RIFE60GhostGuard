# MiniMax H3 iOS API availability gate

Date: 2026-10-17  
Pinned engine source: H3cApp `826fafffc84a15283cb7b3f4c234e7aedf07516c`

## Current finding

The CI audit passes an arm64 iOS compile and forced static-link probe, but compiler output still reports use of MPSGraph APIs newer than the app's iOS 16.0 deployment target. A successful link does not make these APIs safe to call on older operating systems.

Confirmed call sites in the pinned `h3_gpu.m`:

- Around line 1580, the SDPA cache path checks `respondsToSelector:` for the unmasked selector, but that selector check alone is not an OS availability contract. The masked SDPA selector is invoked in the same path.
- Around line 1602, the unmasked SDPA selector is invoked after the selector check.
- Around line 2105, MPSGraph 3D convolution is used; the SDK marks this API as introduced in iOS 16.3.
- Around line 4535, the BF16 GQA path calls masked scaled-dot-product attention; the SDK marks SDPA as introduced in iOS 18.0.

## Required runtime policy

1. Keep the app-wide deployment target at iOS 16.0 so existing editing features are not unnecessarily restricted.
2. Treat H3 as unavailable unless its runtime requirements are met. The current fork's SDPA-based paths require iOS 18.0 or later unless a real fallback implementation is added.
3. Before exposing H3 in the model chooser, add an H3-specific capability gate and verify every entry point into the native engine is unreachable on unsupported OS versions. Do not rely on a UI-only disabled state; enforce the gate in the native/Swift bridge as well.
4. For any path that can execute on iOS 16.0–17.x, either provide a tested alternative attention implementation and guard 3D convolution on iOS 16.0–16.2, or reject the H3 request with a specific unsupported-OS diagnostic before creating GPU graphs.
5. Do not globally suppress availability warnings. Keep them visible in CI until the source-level guards/fallbacks are reviewed. Add a simulator/device runtime test for each supported OS tier where practical.
6. Do not treat this gate as proof of memory feasibility. Model loading, encoder/VAE working sets, peak Metal allocation, thermal behavior, inference parity, and end-to-end video export remain separate release gates.

## Next engineering step

Implement the capability check in the app-facing H3 runtime bridge and a native-engine initialization guard, then extend CI to validate that the guard is present and that availability diagnostics are not hidden. This must precede adding H3 as a selectable editor backend. The guard is a compatibility boundary, not a substitute for device inference tests.

## Current acceptance state

- [x] Selected native sources compile and force-link for arm64 iOS.
- [x] Newer API call sites and OS availability risk identified from compiler diagnostics/source review.
- [ ] H3-specific OS capability gate implemented in the app.
- [ ] Runtime guards or tested fallbacks implemented in the native engine.
- [ ] Model/block execution validated on iPhone 14 Pro Max.
- [ ] End-to-end local reference-conditioned video generation and export validated.
