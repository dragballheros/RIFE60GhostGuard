#!/usr/bin/env python3
"""Validate the official Wan VAE cached encode/decode schedules.

The reference is the official WanVAE_ wrapper path (which itself uses the
causal temporal cache); the candidate independently reproduces its documented
chunk schedules. This validates schedule reproduction, not full-sequence
mathematical equivalence, Core ML export, VACE, or iPhone execution.
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
    parser.add_argument("--frames", type=int, default=5,
                        help="video frames; must be 1 mod 4 for the tested cache schedule")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.height, args.width) < 8 or args.height % 8 or args.width % 8:
        parser.error("height and width must be multiples of 8 and at least 8")
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error("--frames must be 1, 5, 9, 13, ... for the official [1,4,4,...] schedule")

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

    # Exact per-channel normalization used by the official WanVAE wrapper.
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=torch.float32).view(1, 16, 1, 1, 1)
    scale = [mean, 1.0 / std]

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    video = torch.randn((1, 3, args.frames, args.height, args.width),
                        generator=generator, dtype=torch.float32)
    latent_frames = 1 + (args.frames - 1) // 4
    latent = torch.randn((1, 16, latent_frames, args.height // 8, args.width // 8),
                         generator=generator, dtype=torch.float32)

    def stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
        ref = reference.detach().cpu().numpy()
        got = candidate.detach().cpu().numpy()
        result = {
            "reference_shape": list(ref.shape),
            "candidate_shape": list(got.shape),
            "shape_match": ref.shape == got.shape,
        }
        if ref.shape != got.shape:
            return result
        delta = got - ref
        result.update({
            "max_absolute_error": float(np.abs(delta).max()),
            "mean_absolute_error": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "all_finite": bool(np.isfinite(got).all()),
        })
        return result

    with torch.no_grad():
        # Official wrapper result: model.encode uses [1,4,4,...] cache chunks.
        encoder_reference = model.encode(video, scale)

        # Independently reproduce the official encode schedule and normalization.
        enc_cache = [None] * module.count_conv3d(model.encoder)
        enc_parts = []
        starts = [0] + list(range(1, args.frames, 4))
        for idx, start in enumerate(starts):
            end = min(args.frames, 1 if idx == 0 else start + 4)
            cursor = [0]
            enc_parts.append(model.encoder(
                video[:, :, start:end], feat_cache=enc_cache, feat_idx=cursor
            ))
        enc_out = torch.cat(enc_parts, dim=2)
        mu, _log_var = model.conv1(enc_out).chunk(2, dim=1)
        encoder_candidate = (mu - mean) * (1.0 / std)

        # Official wrapper result: model.decode uses one latent frame per chunk.
        decoder_reference = model.decode(latent, scale)

        # Independently reproduce decode unscaling, conv2, and per-frame cache.
        decoder_input = model.conv2(latent / (1.0 / std) + mean)
        dec_cache = [None] * module.count_conv3d(model.decoder)
        dec_parts = []
        for frame_index in range(latent.shape[2]):
            cursor = [0]
            dec_parts.append(model.decoder(
                decoder_input[:, :, frame_index:frame_index + 1],
                feat_cache=dec_cache, feat_idx=cursor
            ))
        decoder_candidate = torch.cat(dec_parts, dim=2).clamp_(-1, 1)

    encoder_metrics = stats(encoder_reference, encoder_candidate)
    decoder_metrics = stats(decoder_reference, decoder_candidate)
    for metrics in (encoder_metrics, decoder_metrics):
        metrics["within_tolerance"] = bool(
            metrics.get("shape_match", False)
            and metrics.get("all_finite", False)
            and metrics.get("max_absolute_error", float("inf")) <= args.tolerance
        )

    report = {
        "probe": "wan_vae_official_temporal_cache_schedule_reproduction",
        "official_source_commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
        "complete_vace_pipeline": False,
        "iphone_execution_validated": False,
        "input_video_shape": list(video.shape),
        "latent_shape": list(latent.shape),
        "encoder_chunk_schedule": [1] + [4] * ((args.frames - 1) // 4),
        "decoder_chunk_schedule": [1] * latent_frames,
        "seed": args.seed,
        "absolute_error_tolerance": args.tolerance,
        "encoder": encoder_metrics,
        "decoder": decoder_metrics,
        "pass": encoder_metrics["within_tolerance"] and decoder_metrics["within_tolerance"],
        "note": (
            "Compares an independent reproduction of the official cached encode/decode "
            "schedule against WanVAE_ wrapper outputs. This is not full-sequence graph "
            "equivalence, Core ML streaming export, VACE transformer validation, or an iPhone test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
