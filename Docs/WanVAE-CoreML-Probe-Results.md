# Wan VAE Core ML export probe — measured result

Date: 2026-10-15  
Branch: `research/wan21-vace-ios-feasibility`  
Initial encoder workflow: [Probe #8](https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/38781782053)  
Follow-up encoder + decoder workflow: [Probe #14](https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/38796988952)  
Latest decoder diagnostics: [Workflow artifact](https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/38796988952/artifacts/12002806289)

## Result

The fixed-shape **encoder-only** export completed successfully on the macOS GitHub Actions runner using the official Wan2.1 VAE source pinned to `9737cba9c1c3c4d04b33fcad41c111989865d315` and the official `Wan2.1_VAE.pth` weights. The test input was a deterministic random tensor with seed `20261015`, not an all-zero tensor.

| Measurement | Result |
|---|---:|
| Input | `[1, 3, 5, 64, 64]` |
| Encoder output | `[1, 32, 5, 8, 8]` |
| Maximum absolute error vs traced PyTorch reference | 0.0735290 |
| Mean absolute error vs traced PyTorch reference | 0.00804413 |
| Core ML package | `WanVAEEncoder-5x64.mlpackage` |
| Diagnostics artifact | [Download workflow artifact](https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/38781782053/artifacts/11996147482) |

## Follow-up result: fixed-shape decoder (Probe #14)

The same macOS CI workflow also exported the fixed-shape decoder after replacing the unsupported `nearest-exact` upsample with explicit index selection. The replacement passed an equivalence check against the official PyTorch upsample with **maximum absolute error 0.0** before conversion.

| Measurement | Result |
|---|---:|
| Decoder input | `[1, 16, 5, 8, 8]` |
| Decoder output | `[1, 3, 5, 64, 64]` |
| Max absolute error vs traced PyTorch reference | 0.0195475 |
| Mean absolute error vs traced PyTorch reference | 0.00212934 |
| RMSE | 0.00276781 |
| Reference output range | [-1.104354, 1.201722] |
| Core ML output range | [-1.097656, 1.201172] |

This is a meaningful component-level improvement: both fixed-shape VAE encoder and decoder exports now convert and execute in CI. The output errors are measured in a FLOAT16 Core ML export and still need quality acceptance criteria. This test does **not** validate causal temporal caches or the official chunk-by-chunk decoder path, and it does not test on an iPhone.

## Interpretation

- This establishes that **one fixed-shape encoder graph can be converted and evaluated** with Core ML Tools 9.0 on the CI macOS runner.
- It does **not** establish acceptable visual quality, iPhone GPU/ANE execution, memory headroom, or throughput. The measured error is recorded rather than described as lossless; max/mean absolute error alone are not sufficient to determine whether it is acceptable.
- It does **not** export or validate the VAE decoder, temporal-cache behavior, VACE conditioning, text encoder, scheduler, or diffusion transformer.
- It does **not** make the VACE feature ready for the app, and the package is a small 64×64 test graph—not a production 720p model component.

## Next required gates

1. Add a separate decoder export experiment using a fixed latent shape, then compare decoded output numerically with the PyTorch reference.
2. Verify the official VAE's causal temporal-chunk behavior. A full-sequence traced decoder is not equivalent evidence for the official cached streaming path.
3. Expand numerical reporting to include RMSE, output range, and relative error; establish acceptance tolerances from representative inputs rather than picking a threshold to make CI green.
4. Inspect Core ML model compute-unit/runtime behavior on actual iPhone hardware before claiming ANE/GPU support or estimating performance.
5. Continue auditing the VACE 14B transformer and conditioning stack independently. Passing a VAE probe is not evidence that the 14B/720p pipeline fits the iPhone 14 Pro Max.

## Reproducibility

The successful run and its logs are linked above. The workflow's diagnostic JSON is retained for 14 days. The experiment downloads only the VAE weights; it does not download the full VACE checkpoint set or call any inference service.

## Follow-up result: fixed-shape SDPA attention kernel (Probe #20)

Workflow: [Probe #20](https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/38799771780)  
Diagnostics artifact: [Download probe diagnostics](https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/38799771780/artifacts/12004155850)

The candidate scaled-dot-product attention kernel converted to a Core ML ML Program and executed on the macOS CI runner using deterministic random tensors. The test used 40 heads with head dimension 128, 32 query tokens, and 512 key/value tokens.

| Measurement | Result |
|---|---:|
| Query shape | `[1, 40, 32, 128]` |
| Key/value shape | `[1, 40, 512, 128]` |
| Output shape | `[1, 40, 32, 128]` |
| Maximum absolute error vs PyTorch | 0.000140581 |
| Mean absolute error vs PyTorch | 0.0000103899 |
| RMSE | 0.0000147773 |

This supports the narrow claim that this **fixed-shape SDPA kernel** is convertible and numerically close to the PyTorch reference under this test. It does not establish compatibility with every attention path in Wan/VACE, variable sequence lengths or masks, rotary embeddings, temporal caches, the complete transformer, iPhone execution, or long-sequence memory/performance. It is not a full replacement implementation for Wan's CUDA/FlashAttention stack.

## Current gate after Probe #20

1. Validate the official VAE's chunked temporal-cache behavior against a full-sequence reference using representative short clips and both encoder and decoder paths.
2. Establish whether a bounded, layer-wise transformer execution design is technically viable. A 14B model's raw 4-bit weight floor is about 7 GB before metadata and activations, so a fully resident model is not viable in the phone's memory class.
3. Inspect VACE-specific conditioning/injection, rotary embeddings, attention masks, and scheduler paths—not just generic SDPA.
4. Continue to withhold production UI integration until model conversion, temporal equivalence, memory, runtime, and 1920×1080 export strategy have credible evidence.

The workflow passed on a hosted macOS CI runner. No on-device iPhone inference or complete VACE edit has been demonstrated.
