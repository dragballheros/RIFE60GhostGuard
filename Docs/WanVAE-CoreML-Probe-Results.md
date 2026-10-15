# Wan VAE Core ML export probe — measured result

Date: 2026-10-15  
Branch: `research/wan21-vace-ios-feasibility`  
Workflow: [Wan VAE Core ML Export Probe #8](https://github.com/dragballheros/RIFE60GhostGuard/actions/runs/38781782053)

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
