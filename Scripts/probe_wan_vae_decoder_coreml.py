#!/usr/bin/env python3
"""Probe a fixed-shape Core ML export of Wan 2.1's VAE decoder.

This is a decoder-only conversion experiment, not validation of Wan's official
causal temporal-cache streaming path or of the full VACE editing pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=5)
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    args = parser.parse_args()

    if args.latent_frames < 1:
        parser.error("--latent-frames must be positive")
    if min(args.latent_height, args.latent_width) < 1:
        parser.error("latent spatial dimensions must be positive")
    vae_source = args.wan_repo / "wan/modules/vae.py"
    if not vae_source.is_file():
        parser.error("--wan-repo must point to a checkout of Wan-Video/Wan2.1")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    import numpy as np
    import torch
    import torch.nn.functional as F
    import coremltools as ct

    # Load the pinned official source file without executing package initializers.
    spec = importlib.util.spec_from_file_location("wan_vae_decoder_probe", vae_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official VAE source: {vae_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.WanVAE_(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).float()

    generator = torch.Generator(device="cpu").manual_seed(20261016)
    latent = torch.randn(
        (1, 16, args.latent_frames, args.latent_height, args.latent_width),
        generator=generator,
        dtype=torch.float32,
    )

    # Core ML Tools 9.0 does not implement PyTorch's nearest-exact 2D upsample
    # op. Replace it with explicit constant index selection for this fixed shape,
    # and prove the replacement matches the official implementation first.
    with torch.no_grad():
        official_forward = module.Upsample.forward
        official_reference = model.decoder(latent)

        def nearest_exact_as_gather(self, x):
            out_h = int(round(x.shape[-2] * self.scale_factor[0]))
            out_w = int(round(x.shape[-1] * self.scale_factor[1]))
            y = torch.floor(
                (torch.arange(out_h, device=x.device, dtype=torch.float32) + 0.5)
                * (x.shape[-2] / out_h)
            ).to(torch.long).clamp_(0, x.shape[-2] - 1)
            z = torch.floor(
                (torch.arange(out_w, device=x.device, dtype=torch.float32) + 0.5)
                * (x.shape[-1] / out_w)
            ).to(torch.long).clamp_(0, x.shape[-1] - 1)
            return x.index_select(-2, y).index_select(-1, z).type_as(x)

        module.Upsample.forward = nearest_exact_as_gather
        replacement_reference = model.decoder(latent)
        replacement_delta = (replacement_reference - official_reference).abs()
        replacement_max_error = float(replacement_delta.max())
        if not torch.allclose(
            replacement_reference, official_reference, rtol=0.0, atol=1e-6
        ):
            module.Upsample.forward = official_forward
            raise RuntimeError(
                "Explicit nearest-exact replacement failed equivalence check: "
                f"max_abs_error={replacement_max_error}"
            )
        traced = torch.jit.trace(model.decoder, latent, strict=True)
        reference = traced(latent)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="latent", shape=tuple(latent.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name="decoded")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    converted.save(str(args.output))

    prediction = converted.predict({"latent": latent.numpy()})["decoded"]
    reference_np = reference.detach().cpu().numpy()
    absolute_error = np.abs(prediction - reference_np)
    report = {
        "probe": "wan_vae_decoder_fixed_shape",
        "complete_vace_pipeline": False,
        "official_temporal_cache_validated": False,
        "input_shape": list(latent.shape),
        "validation_input": "deterministic random latent tensor, seed 20261016",
        "output_shape": list(reference_np.shape),
        "package_path": str(args.output),
        "nearest_exact_replacement_max_absolute_error_vs_official": replacement_max_error,
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - reference_np)))),
        "reference_output_min": float(reference_np.min()),
        "reference_output_max": float(reference_np.max()),
        "converted_output_min": float(prediction.min()),
        "converted_output_max": float(prediction.max()),
        "note": (
            "Decoder-only fixed-shape export. Explicit nearest-exact index selection "
            "is numerically checked against the official PyTorch operation first. "
            "This does not validate the official causal temporal-cache streaming path, "
            "VACE transformer, or complete pipeline."
        ),
    }
    report_path = args.output.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
