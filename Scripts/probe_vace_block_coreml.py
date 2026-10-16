#!/usr/bin/env python3
"""Export one real Wan VACE conditioning block to Core ML and check numerical parity.

This is a component-level conversion harness, not a full VACE denoiser. It imports
VaceWanAttentionBlock from the pinned upstream Wan2.1 source and instantiates it
with the published 1.3B width/depth dimensions. The weights are deterministic
random initialization unless a future checkpoint-loading path is added; therefore
this harness verifies operator/shape portability, not learned-weight quality.

The fixed test grid is 1x2x2 latent tokens and text context length 8. It deliberately
does not claim 480p throughput or iPhone memory/thermal validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

WAN_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
VACE_1_3B = {"dim": 1536, "ffn_dim": 8960, "num_heads": 12}
GRID = (1, 2, 2)
CONTEXT_TOKENS = 8


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def tensor_bytes(module: Any) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def max_abs_diff(left: Any, right: Any) -> float:
    import torch
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def build_fixed_rope(freqs: Any, grid: tuple[int, int, int]):
    """Precompute Wan's 3-axis rotary factors, preserving the upstream grid layout."""
    import torch

    frames, height, width = grid
    head_pairs = freqs.shape[1]
    quarter = head_pairs // 3
    parts = freqs.split(
        [head_pairs - 2 * quarter, quarter, quarter], dim=1
    )
    selected = torch.cat([
        parts[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
        parts[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
        parts[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
    ], dim=-1).reshape(frames * height * width, 1, -1)
    return selected.real.float().view(1, -1, 1, head_pairs), selected.imag.float().view(1, -1, 1, head_pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output .mlpackage directory")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-commit", default=WAN_COMMIT)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--max-rmse", type=float, default=0.035)
    parser.add_argument("--max-absolute-error", type=float, default=0.20)
    parser.add_argument(
        "--quantization-levels",
        nargs="*",
        choices=("int8", "int4"),
        default=["int8", "int4"],
        help="also test Core ML post-training weight compression at these bit widths",
    )
    args = parser.parse_args()
    if args.context_tokens < 1:
        parser.error("--context-tokens must be >= 1")

    repo = args.wan_repo.resolve()
    if not (repo / "wan/modules/vace_model.py").is_file():
        parser.error(f"{repo} is not an upstream Wan2.1 checkout (missing wan/modules/vace_model.py)")
    actual_commit = git_head(repo)
    if args.expected_commit and actual_commit != args.expected_commit:
        parser.error(f"Wan source commit mismatch: expected {args.expected_commit}, got {actual_commit}")
    # Import only the pinned Wan modules required by this harness. The upstream
    # package __init__ eagerly imports its full T5/distributed stack, which is
    # unnecessary here and pulls unrelated CUDA-era dependencies into CI.
    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(repo / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(repo / "wan/modules")]
    sys.modules["wan"] = wan_package
    sys.modules["wan.modules"] = modules_package

    import numpy as np
    import torch
    import torch.nn.functional as F
    import coremltools as ct
    import wan.modules.model as wan_model
    from wan.modules.vace_model import VaceWanAttentionBlock

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    dim = VACE_1_3B["dim"]
    ffn_dim = VACE_1_3B["ffn_dim"]
    heads = VACE_1_3B["num_heads"]
    head_dim = dim // heads
    frames, height, width = GRID
    tokens = frames * height * width
    if head_dim % 2:
        raise RuntimeError("Wan rotary embedding requires an even per-head dimension")

    # Build the same frequency table as WanModel.__init__ without allocating
    # the entire 30-layer 1.3B backbone.
    freqs = torch.cat([
        wan_model.rope_params(1024, head_dim - 4 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
    ], dim=1)
    grid_sizes = torch.tensor([GRID], dtype=torch.long)
    seq_lens = torch.tensor([tokens], dtype=torch.long)
    rope_cos, rope_sin = build_fixed_rope(freqs, GRID)

    # Check the real-valued rotary formulation against Wan's own complex-valued
    # implementation before replacing it for Core ML tracing.
    sample_q = torch.randn(1, tokens, heads, head_dim)
    upstream_rotated = wan_model.rope_apply(sample_q, grid_sizes, freqs)
    even = sample_q[..., 0::2]
    odd = sample_q[..., 1::2]
    expected_cos = rope_cos.expand(1, tokens, heads, head_dim // 2)
    expected_sin = rope_sin.expand(1, tokens, heads, head_dim // 2)
    real_rotated = torch.empty_like(sample_q)
    real_rotated[..., 0::2] = even * expected_cos - odd * expected_sin
    real_rotated[..., 1::2] = even * expected_sin + odd * expected_cos
    rope_error = max_abs_diff(upstream_rotated, real_rotated)
    if rope_error > 2e-5:
        raise RuntimeError(f"Fixed-grid rotary rewrite mismatch: max_abs={rope_error:.7g}")

    # Wan's stock kernel requires CUDA FlashAttention. Use SDPA as the portable
    # mathematical attention implementation for this fixed, unpadded test shape.
    # This is not claimed to match FlashAttention's fused-kernel numerics exactly.
    def portable_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mask = None
        if k_lens is not None:
            key_positions = torch.arange(k.shape[1], device=k.device)[None, :]
            mask = (key_positions < k_lens.to(k.device)[:, None])[:, None, None, :]
        result = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=mask, dropout_p=float(kwargs.get("dropout_p", 0.0)),
            is_causal=bool(kwargs.get("causal", False)),
        )
        return result.transpose(1, 2).contiguous()

    original_rope_apply = wan_model.rope_apply
    original_flash_attention = wan_model.flash_attention

    def fixed_rope_apply(x, _grid_sizes, _freqs):
        c = rope_cos.to(device=x.device, dtype=x.dtype).expand(1, tokens, 1, head_dim // 2)
        s = rope_sin.to(device=x.device, dtype=x.dtype).expand(1, tokens, 1, head_dim // 2)
        evens, odds = x[..., 0::2], x[..., 1::2]
        rotated_even = evens * c - odds * s
        rotated_odd = evens * s + odds * c
        # Avoid in-place assignment into strided slices; Core ML translates
        # tensor copy updates as a full-shaped replacement and rejects the halves.
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    wan_model.rope_apply = fixed_rope_apply
    wan_model.flash_attention = portable_attention

    block = VaceWanAttentionBlock(
        cross_attn_type="t2v_cross_attn",
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        block_id=0,
    ).eval()
    # Upstream intentionally zero-initializes the VACE before/after projections.
    # Randomly initialize them here so the conversion/parity test exercises those
    # linear operators instead of trivially returning the unchanged latent/zero hint.
    with torch.no_grad():
        for projection_name in ("before_proj", "after_proj"):
            projection = getattr(block, projection_name, None)
            if projection is not None:
                torch.nn.init.normal_(projection.weight, mean=0.0, std=0.01)
                if projection.bias is not None:
                    torch.nn.init.zeros_(projection.bias)

    class FixedShapeVACEBlock(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, conditioning, latent_tokens, time_modulation, text_context):
            return self.layer(
                conditioning,
                latent_tokens,
                e=time_modulation,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=text_context,
                context_lens=None,
            )

    wrapper = FixedShapeVACEBlock(block).eval()
    conditioning = torch.randn(1, tokens, dim, dtype=torch.float32) * 0.1
    latent_tokens = torch.randn(1, tokens, dim, dtype=torch.float32) * 0.1
    time_modulation = torch.randn(1, 6, dim, dtype=torch.float32) * 0.01
    text_context = torch.randn(1, args.context_tokens, dim, dtype=torch.float32) * 0.1
    inputs = (conditioning, latent_tokens, time_modulation, text_context)
    input_names = ["conditioning", "latent_tokens", "time_modulation", "text_context"]

    with torch.no_grad():
        start = time.perf_counter()
        reference = wrapper(*inputs)
        reference_seconds = time.perf_counter() - start
        traced = torch.jit.trace(wrapper, inputs, strict=False, check_trace=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="conditioning", shape=tuple(conditioning.shape), dtype=np.float32),
            ct.TensorType(name="latent_tokens", shape=tuple(latent_tokens.shape), dtype=np.float32),
            ct.TensorType(name="time_modulation", shape=tuple(time_modulation.shape), dtype=np.float32),
            ct.TensorType(name="text_context", shape=tuple(text_context.shape), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="conditioned_context"),
            ct.TensorType(name="skip_hint"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    conversion_seconds = time.perf_counter() - start
    converted.save(str(args.output))

    prediction_inputs = {
        name: value.detach().cpu().numpy().astype(np.float32)
        for name, value in zip(input_names, inputs)
    }
    reference_numpy = tuple(
        value.detach().cpu().numpy().astype(np.float32) for value in reference
    )

    def output_errors(prediction_map: dict[str, Any]) -> list[dict[str, Any]]:
        names = ("conditioned_context", "skip_hint")
        result = []
        for name, ref_array in zip(names, reference_numpy):
            delta = np.asarray(prediction_map[name], dtype=np.float32) - ref_array
            result.append({
                "name": name,
                "shape": list(delta.shape),
                "max_absolute_error": float(np.abs(delta).max()),
                "mean_absolute_error": float(np.abs(delta).mean()),
                "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            })
        return result

    prediction = converted.predict(prediction_inputs)
    errors = output_errors(prediction)
    quantization_reports = []
    if args.quantization_levels:
        from coremltools.optimize.coreml import (
            OptimizationConfig,
            OpLinearQuantizerConfig,
            linear_quantize_weights,
        )

        for bits in args.quantization_levels:
            compressed_path = args.output.with_name(
                f"{args.output.stem}-{bits}-weights.mlpackage"
            )
            started = time.perf_counter()
            try:
                config = OptimizationConfig(
                    global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric",
                        dtype=bits,
                        granularity="per_channel",
                    )
                )
                compressed = linear_quantize_weights(converted, config=config)
                compressed.save(str(compressed_path))
                compressed_prediction = compressed.predict(prediction_inputs)
                compressed_errors = output_errors(compressed_prediction)
                quantization_reports.append({
                    "bits": int(bits[3:]),
                    "package_path": str(compressed_path),
                    "package_size_bytes": sum(
                        item.stat().st_size for item in compressed_path.rglob("*") if item.is_file()
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                    "outputs": compressed_errors,
                    "passed_current_float_reference_thresholds": all(
                        item["rmse"] <= args.max_rmse
                        and item["max_absolute_error"] <= args.max_absolute_error
                        for item in compressed_errors
                    ),
                    "runtime_precision_note": "Weight storage is quantized; intermediate calculations remain floating point unless separately activation-quantized.",
                })
            except Exception as exc:
                quantization_reports.append({
                    "bits": int(bits[3:]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                    "passed_current_float_reference_thresholds": False,
                })

    parameter_count = sum(p.numel() for p in block.parameters())
    fp32_weight_bytes = tensor_bytes(block)
    report = {
        "probe": "official_vace_wan_attention_block_coreml_fixed_shape",
        "source": {
            "repository": "https://github.com/Wan-Video/Wan2.1",
            "commit": actual_commit,
            "expected_commit": args.expected_commit,
            "component_class": "wan.modules.vace_model.VaceWanAttentionBlock",
            "architecture": "Wan2.1 VACE 1.3B width/FFN/head dimensions",
        },
        "weights": {
            "initialization": "deterministic random initialization; no learned checkpoint weights loaded",
            "seed": args.seed,
            "parameter_count": parameter_count,
            "fp32_parameter_bytes": fp32_weight_bytes,
            "fp16_parameter_payload_estimate_bytes": parameter_count * 2,
        },
        "test_shape": {
            "grid_fhw": list(GRID),
            "latent_tokens": tokens,
            "heads": heads,
            "head_dim": head_dim,
            "context_tokens": args.context_tokens,
            "conditioning": list(conditioning.shape),
            "latent_tokens_shape": list(latent_tokens.shape),
            "time_modulation": list(time_modulation.shape),
            "text_context": list(text_context.shape),
        },
        "validation": {
            "rotary_real_vs_official_complex_max_absolute_error": rope_error,
            "outputs": errors,
            "pass_rmse_limit": args.max_rmse,
            "pass_max_absolute_error_limit": args.max_absolute_error,
            "passed": all(
                item["rmse"] <= args.max_rmse
                and item["max_absolute_error"] <= args.max_absolute_error
                for item in errors
            ),
        },
        "runtime": {
            "pytorch_reference_seconds": reference_seconds,
            "coreml_conversion_seconds": conversion_seconds,
            "fp16_package_path": str(args.output),
            "fp16_package_size_bytes": sum(
                item.stat().st_size for item in args.output.rglob("*") if item.is_file()
            ),
        },
        "weight_quantization_experiments": quantization_reports,
        "limitations": [
            "Fixed four-latent-token spatial-temporal grid and fixed context length; not a dynamic-shape model.",
            "Uses the official VACE block class and its conditioning/skip projections, but substitutes a portable SDPA path for the CUDA-only FlashAttention kernel.",
            "Deterministic random weights; this proves operator conversion/parity only, not quality with published learned weights.",
            "Does not include full 30-block transformer, text encoder, VAE, denoising schedule, clip/mask preprocessing, or end-to-end video editing.",
            "Does not measure iPhone resident memory, Metal allocations, speed, thermals, or watchdog safety.",
            "Not yet suitable for bundling into a user-facing app or claiming on-device VACE support.",
        ],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    report_path = args.report or args.output.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered)

    wan_model.rope_apply = original_rope_apply
    wan_model.flash_attention = original_flash_attention
    return 0 if report["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
