#!/usr/bin/env python3
"""Attempt a fixed-shape Core ML export of Wan 2.1's temporal VAE encoder.

This is an export probe only. It does not export VACE's diffusion transformer,
text encoder, scheduler, or a complete video-editing pipeline.

Example (run on macOS with coremltools installed):
  python Scripts/probe_wan_vae_coreml.py \
    --wan-repo Vendor/Wan2.1 \
    --checkpoint /tmp/Wan2.1_VAE.pth \
    --output /tmp/WanVAEEncoder-5x64.mlpackage

The checkpoint should come from the official Wan-AI/Wan2.1-VACE-1.3B repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5, help="Fixed input frame count; use 4n+1.")
    parser.add_argument("--height", type=int, default=64, help="Fixed test height; use a multiple of 8.")
    parser.add_argument("--width", type=int, default=64, help="Fixed test width; use a multiple of 8.")
    args = parser.parse_args()

    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error("--frames must be 4n+1 (for example 1, 5, or 9)")
    if args.height < 8 or args.height % 8 or args.width < 8 or args.width % 8:
        parser.error("--height and --width must be positive multiples of 8")
    if not (args.wan_repo / "wan/modules/vae.py").is_file():
        parser.error("--wan-repo must point to a checkout of Wan-Video/Wan2.1")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    sys.path.insert(0, str(args.wan_repo.resolve()))
    import numpy as np
    import torch
    import coremltools as ct
    from wan.modules.vae import WanVAE_

    # Match the official WanVAE default architecture used by VACE 1.3B.
    model = WanVAE_(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    # Only deserialize tensor weights; never allow arbitrary checkpoint objects.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).float()

    example = torch.zeros((1, 3, args.frames, args.height, args.width), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model.encoder, example, strict=True)
        reference = traced(example)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="video", shape=tuple(example.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name="moments")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(str(args.output))

    # Reload the converted package and compare its fixed-shape output.
    prediction = mlmodel.predict({"video": example.numpy()})["moments"]
    reference_np = reference.detach().cpu().numpy()
    absolute_error = np.abs(prediction - reference_np)
    report = {
        "probe": "wan_vae_encoder_fixed_shape",
        "complete_vace_pipeline": False,
        "input_shape": list(example.shape),
        "output_shape": list(reference_np.shape),
        "package_path": str(args.output),
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "note": "Encoder-only export; success does not establish VACE DiT, decoder, or full video editing compatibility.",
    }
    report_path = args.output.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
