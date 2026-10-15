#!/usr/bin/env python3
"""Compare official Wan VAE full-sequence and causal temporal-cache paths.

This is a PyTorch reference equivalence probe. It does not export Core ML or
validate execution on iPhone hardware.
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
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.height, args.width) < 8 or args.height % 8 or args.width % 8:
        parser.error("height and width must be multiples of 8 and at least 8")

    import numpy as np
    import torch

    source = args.wan_repo / "wan/modules/vae.py"
    if not source.is_file():
        parser.error("--wan-repo must point to the pinned Wan2.1 checkout")
    if not args.checkpoint.is_file():
        parser.error("official VAE checkpoint does not exist")

    spec = importlib.util.spec_from_file_location("wan_vae_cache_probe", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official VAE source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.WanVAE_(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
        attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).float()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    video = torch.randn((1, 3, 5, args.height, args.width),
                        generator=generator, dtype=torch.float32)
    latent = torch.randn((1, 16, 2, args.height // 8, args.width // 8),
                         generator=generator, dtype=torch.float32)

    def stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
        ref = reference.detach().cpu().numpy()
        got = candidate.detach().cpu().numpy()
        if ref.shape != got.shape:
            return {"reference_shape": list(ref.shape), "cached_shape": list(got.shape),
                    "shape_match": False}
        delta = got - ref
        return {
            "reference_shape": list(ref.shape),
            "cached_shape": list(got.shape),
            "shape_match": True,
            "max_absolute_error": float(np.abs(delta).max()),
            "mean_absolute_error": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "all_finite": bool(np.isfinite(got).all()),
        }

    with torch.no_grad():
        # Encoder reference: a single full sequence without caches.
        encoder_reference = model.encoder(video)
        # Official encode cache schedule: first frame, then four-frame chunks.
        enc_cache = [None] * module.count_conv3d(model.encoder)
        enc_parts = []
        for chunk_index, (start, end) in enumerate(((0, 1), (1, 5))):
            cursor = [0]
            enc_parts.append(model.encoder(
                video[:, :, start:end], feat_cache=enc_cache, feat_idx=cursor
            ))
        encoder_cached = torch.cat(enc_parts, dim=2)

        # Decoder reference and official cached schedule: one latent frame at a time.
        decoder_input = model.conv2(latent)
        decoder_reference = model.decoder(decoder_input)
        dec_cache = [None] * module.count_conv3d(model.decoder)
        dec_parts = []
        for frame_index in range(latent.shape[2]):
            cursor = [0]
            dec_parts.append(model.decoder(
                decoder_input[:, :, frame_index:frame_index + 1],
                feat_cache=dec_cache, feat_idx=cursor
            ))
        decoder_cached = torch.cat(dec_parts, dim=2)

    encoder_metrics = stats(encoder_reference, encoder_cached)
    decoder_metrics = stats(decoder_reference, decoder_cached)
    for metrics in (encoder_metrics, decoder_metrics):
        metrics["within_tolerance"] = bool(
            metrics.get("shape_match", False)
            and metrics.get("all_finite", True)
            and metrics.get("max_absolute_error", float("inf")) <= args.tolerance
        )

    report = {
        "probe": "wan_vae_official_temporal_cache_equivalence",
        "official_source_commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
        "complete_vace_pipeline": False,
        "iphone_execution_validated": False,
        "input_video_shape": list(video.shape),
        "latent_shape": list(latent.shape),
        "encoder_chunk_schedule": [1, 4],
        "decoder_chunk_schedule": [1, 1],
        "seed": args.seed,
        "absolute_error_tolerance": args.tolerance,
        "encoder": encoder_metrics,
        "decoder": decoder_metrics,
        "pass": encoder_metrics["within_tolerance"] and decoder_metrics["within_tolerance"],
        "note": (
            "Compares the official PyTorch full-sequence graphs with the official "
            "causal-cache calling pattern at small fixed shapes. This is not Core ML "
            "streaming export, VACE transformer validation, or an iPhone test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
