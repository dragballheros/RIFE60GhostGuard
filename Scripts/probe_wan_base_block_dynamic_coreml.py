#!/usr/bin/env python3
"""Dynamic-shape Core ML probe for an actual Wan VACE base transformer block."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_vace_block_coreml import (  # noqa: E402
    VACE_1_3B,
    WAN_COMMIT,
    build_fixed_rope,
    load_safetensors_block,
)

HEIGHT = 4
WIDTH = 4
MAX_FRAMES = 4
MAX_CONTEXT = 16


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def rotate(q, cos, sin):
    even, odd = q[..., 0::2], q[..., 1::2]
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def portable_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    )
    return y.transpose(1, 2).contiguous()


class DynamicBaseBlock(torch.nn.Module):
    """Re-express upstream base-block math with explicit dynamic RoPE and optional VACE hint."""

    def __init__(self, layer: torch.nn.Module, hint_index: int | None):
        super().__init__()
        self.layer = layer
        self.hint_index = hint_index

    def forward(
        self,
        latent_tokens,
        time_modulation,
        text_context,
        rope_cos,
        rope_sin,
        vace_hint=None,
        context_scale=None,
    ):
        layer = self.layer
        batch = latent_tokens.shape[0]
        dim = VACE_1_3B["dim"]
        heads = VACE_1_3B["num_heads"]
        head_dim = dim // heads

        e = (layer.modulation + time_modulation).chunk(6, dim=1)

        # Main self-attention. The FP32 norms and modulation intentionally mirror Wan.
        normalized = layer.norm1(latent_tokens).float() * (1 + e[1]) + e[0]
        self_attn = layer.self_attn
        q = self_attn.norm_q(self_attn.q(normalized)).reshape(batch, -1, heads, head_dim)
        k = self_attn.norm_k(self_attn.k(normalized)).reshape(batch, -1, heads, head_dim)
        v = self_attn.v(normalized).reshape(batch, -1, heads, head_dim)
        q = rotate(q, rope_cos, rope_sin)
        k = rotate(k, rope_cos, rope_sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).contiguous().reshape(batch, -1, dim)
        x = latent_tokens + self_attn.o(y) * e[2]

        # Wan's norm3 is cross-attention norm; norm2 is the FFN norm.
        cross_attn = layer.cross_attn
        normalized = layer.norm3(x)
        q = cross_attn.norm_q(cross_attn.q(normalized)).reshape(batch, -1, heads, head_dim)
        k = cross_attn.norm_k(cross_attn.k(text_context)).reshape(batch, -1, heads, head_dim)
        v = cross_attn.v(text_context).reshape(batch, -1, heads, head_dim)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).contiguous().reshape(batch, -1, dim)
        x = x + cross_attn.o(y)

        normalized = layer.norm2(x).float() * (1 + e[4]) + e[3]
        x = x + layer.ffn(normalized) * e[5]
        if self.hint_index is not None:
            x = x + vace_hint * context_scale
        return x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-commit", default=WAN_COMMIT)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--vace-layer-indices", type=int, nargs="*", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--block-prefix", default=None)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT)
    parser.add_argument("--compute-precision", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--quantization-levels", nargs="*", choices=("int8", "int4"), default=["int8"])
    parser.add_argument("--max-absolute-error", type=float, default=0.20)
    parser.add_argument("--max-rmse", type=float, default=0.035)
    parser.add_argument("--max-relative-rmse", type=float, default=0.02)
    args = parser.parse_args()

    if args.num_layers < 1 or not 0 <= args.block_index < args.num_layers:
        parser.error("--block-index must be within 0 .. --num-layers - 1")
    if not 1 <= args.max_frames <= MAX_FRAMES:
        parser.error(f"--max-frames must be in 1..{MAX_FRAMES}")
    if min(args.height, args.width) <= 0:
        parser.error("--height and --width must be positive")
    if not 1 <= args.max_context_tokens <= 64:
        parser.error("--max-context-tokens must be in 1..64")

    repo = args.wan_repo.resolve()
    if not (repo / "wan/modules/vace_model.py").is_file():
        parser.error(f"{repo} is not an upstream Wan2.1 checkout")
    actual_commit = git_head(repo)
    if args.expected_commit and actual_commit != args.expected_commit:
        parser.error(f"Wan source commit mismatch: expected {args.expected_commit}, got {actual_commit}")

    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(repo / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(repo / "wan/modules")]
    sys.modules["wan"] = wan_package
    sys.modules["wan.modules"] = modules_package
    import wan.modules.model as wan_model
    from wan.modules.vace_model import BaseWanAttentionBlock

    torch.manual_seed(20261017)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dim, ffn_dim, heads = VACE_1_3B["dim"], VACE_1_3B["ffn_dim"], VACE_1_3B["num_heads"]
    head_dim = dim // heads
    vace_layers = args.vace_layer_indices if args.vace_layer_indices is not None else list(range(0, args.num_layers, 2))
    if 0 not in vace_layers or len(set(vace_layers)) != len(vace_layers):
        parser.error("VACE layer indices must be unique and include 0")
    hint_index = vace_layers.index(args.block_index) if args.block_index in vace_layers else None
    prefix = args.block_prefix or f"blocks.{args.block_index}."

    block = BaseWanAttentionBlock(
        cross_attn_type="t2v_cross_attn",
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        block_id=hint_index,
    ).eval()
    checkpoint_metadata = None
    if args.checkpoint is not None:
        checkpoint_metadata = load_safetensors_block(block, args.checkpoint, prefix)

    freqs = torch.cat([
        wan_model.rope_params(1024, head_dim - 4 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
    ], dim=1)

    min_tokens = args.height * args.width
    max_tokens = args.max_frames * min_tokens
    token_dim = ct.RangeDim(lower_bound=min_tokens, upper_bound=max_tokens, default=min_tokens, symbol="latent_tokens")
    context_dim = ct.RangeDim(lower_bound=1, upper_bound=args.max_context_tokens, default=min(args.max_context_tokens, 8), symbol="context_tokens")
    model = DynamicBaseBlock(block, hint_index).eval()

    sample_tokens = min_tokens
    sample_context = min(args.max_context_tokens, 8)
    sample_cos, sample_sin = build_fixed_rope(freqs, (1, args.height, args.width))
    latent = torch.randn(1, sample_tokens, dim) * 0.1
    time_mod = torch.randn(1, 6, dim) * 0.01
    text = torch.randn(1, sample_context, dim) * 0.1
    example_list = [latent, time_mod, text, sample_cos, sample_sin]
    input_specs = [
        ct.TensorType(name="latent_tokens", shape=(1, token_dim, dim), dtype=np.float32),
        ct.TensorType(name="time_modulation", shape=(1, 6, dim), dtype=np.float32),
        ct.TensorType(name="text_context", shape=(1, context_dim, dim), dtype=np.float32),
        ct.TensorType(name="rope_cos", shape=(1, token_dim, 1, head_dim // 2), dtype=np.float32),
        ct.TensorType(name="rope_sin", shape=(1, token_dim, 1, head_dim // 2), dtype=np.float32),
    ]
    input_names = ["latent_tokens", "time_modulation", "text_context", "rope_cos", "rope_sin"]
    if hint_index is not None:
        hint = torch.randn(1, sample_tokens, dim) * 0.1
        scale = torch.tensor([1.0], dtype=torch.float32)
        example_list.extend([hint, scale])
        input_specs.extend([
            ct.TensorType(name="vace_hint", shape=(1, token_dim, dim), dtype=np.float32),
            ct.TensorType(name="context_scale", shape=(1,), dtype=np.float32),
        ])
        input_names.extend(["vace_hint", "context_scale"])
    else:
        hint, scale = None, None

    trace = torch.jit.trace(model, tuple(example_list), strict=False, check_trace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    converted = ct.convert(
        trace,
        convert_to="mlprogram",
        inputs=input_specs,
        outputs=[ct.TensorType(name="hidden_state")],
        compute_precision=ct.precision.FLOAT32 if args.compute_precision == "fp32" else ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    conversion_seconds = time.perf_counter() - start
    converted.save(str(args.output))

    quantized_models = []
    quantization_setup = []
    if args.quantization_levels:
        from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights
        for bits in args.quantization_levels:
            path = args.output.with_name(f"{args.output.stem}-int{bits[3:]}-weights.mlpackage")
            try:
                compressed = linear_quantize_weights(
                    converted,
                    config=OptimizationConfig(global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric", dtype=bits, granularity="per_channel"
                    )),
                )
                compressed.save(str(path))
                quantized_models.append((bits, compressed, path))
                quantization_setup.append({
                    "bits": int(bits[3:]),
                    "package_size_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
                })
            except Exception as exc:
                quantization_setup.append({"bits": int(bits[3:]), "error": f"{type(exc).__name__}: {exc}"})

    original_rope = wan_model.rope_apply
    original_attention = wan_model.flash_attention
    tests = []
    names_out = ("hidden_state",)
    max_abs = {"hidden_state": 0.0}
    max_rmse = {"hidden_state": 0.0}
    max_relative_rmse = {"hidden_state": 0.0}
    q_metrics = {
        bits: {"max_absolute_error": 0.0, "max_rmse": 0.0}
        for bits, _m, _p in quantized_models
    }

    try:
        for frame_count in range(1, args.max_frames + 1):
            token_count = frame_count * min_tokens
            cos, sin = build_fixed_rope(freqs, (frame_count, args.height, args.width))
            for ctx_count in sorted(set((1, min(5, args.max_context_tokens), min(8, args.max_context_tokens), args.max_context_tokens))):
                ctx_count = max(1, ctx_count)
                torch.manual_seed(5000 + frame_count * 100 + ctx_count)
                x = torch.randn(1, token_count, dim) * 0.1
                e = torch.randn(1, 6, dim) * 0.01
                ctx = torch.randn(1, ctx_count, dim) * 0.1
                case_hint = torch.randn(1, token_count, dim) * 0.1 if hint_index is not None else None
                case_scale = torch.tensor([1.0], dtype=torch.float32) if hint_index is not None else None
                prediction_inputs = {
                    "latent_tokens": x.numpy(),
                    "time_modulation": e.numpy(),
                    "text_context": ctx.numpy(),
                    "rope_cos": cos.numpy(),
                    "rope_sin": sin.numpy(),
                }
                if hint_index is not None:
                    prediction_inputs["vace_hint"] = case_hint.numpy()
                    prediction_inputs["context_scale"] = case_scale.numpy()

                def active_rope(z, _grid, _freq):
                    return rotate(z, cos, sin)

                wan_model.rope_apply = active_rope
                wan_model.flash_attention = portable_attention
                with torch.no_grad():
                    reference = block(
                        x,
                        hints=[case_hint] * (hint_index + 1) if hint_index is not None else [],
                        context_scale=case_scale if case_scale is not None else 1.0,
                        e=e,
                        seq_lens=torch.tensor([token_count], dtype=torch.long),
                        grid_sizes=torch.tensor([[frame_count, args.height, args.width]], dtype=torch.long),
                        freqs=freqs,
                        context=ctx,
                        context_lens=None,
                    )
                    baseline_prediction = converted.predict(prediction_inputs)["hidden_state"]
                    quantized_predictions = {
                        bits: qmodel.predict(prediction_inputs)["hidden_state"]
                        for bits, qmodel, _p in quantized_models
                    }
                ref = reference.detach().cpu().numpy().astype(np.float32)
                baseline_prediction = np.asarray(baseline_prediction, dtype=np.float32)
                delta = baseline_prediction - ref
                err = {
                    "max_absolute_error": float(np.abs(delta).max()),
                    "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                    "relative_rmse": float(np.sqrt(np.mean(np.square(delta))) / max(float(np.sqrt(np.mean(np.square(ref)))), 1e-8)),
                }
                max_abs["hidden_state"] = max(max_abs["hidden_state"], err["max_absolute_error"])
                max_rmse["hidden_state"] = max(max_rmse["hidden_state"], err["rmse"])
                max_relative_rmse["hidden_state"] = max(max_relative_rmse["hidden_state"], err["relative_rmse"])
                q_results = {}
                for bits, qpred in quantized_predictions.items():
                    q_delta = np.asarray(qpred, dtype=np.float32) - ref
                    qa = float(np.abs(q_delta).max())
                    qr = float(np.sqrt(np.mean(np.square(q_delta))))
                    q_metrics[bits]["max_absolute_error"] = max(q_metrics[bits]["max_absolute_error"], qa)
                    q_metrics[bits]["max_rmse"] = max(q_metrics[bits]["max_rmse"], qr)
                    q_results[f"int{bits[3:]}"] = {"max_absolute_error": qa, "rmse": qr}
                tests.append({
                    "frames": frame_count,
                    "latent_tokens": token_count,
                    "context_tokens": ctx_count,
                    "outputs": err,
                    "quantized_outputs": q_results,
                })
    finally:
        wan_model.rope_apply = original_rope
        wan_model.flash_attention = original_attention

    threshold_pass = (
        max_abs["hidden_state"] <= args.max_absolute_error
        and max_rmse["hidden_state"] <= args.max_rmse
        and max_relative_rmse["hidden_state"] <= args.max_relative_rmse
    )
    q_reports = []
    for bits, _m, path in quantized_models:
        good = q_metrics[bits]["max_absolute_error"] <= args.max_absolute_error and q_metrics[bits]["max_rmse"] <= args.max_rmse
        q_reports.append({
            "bits": int(bits[3:]),
            "package_size_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
            **q_metrics[bits],
            "passed": good,
        })
    passed = threshold_pass and len(q_reports) == len(args.quantization_levels) and all(x["passed"] for x in q_reports) and all("error" not in x for x in quantization_setup)
    report = {
        "probe": "wan_vace_base_block_dynamic_temporal_axis",
        "source": {
            "repository": "https://github.com/Wan-Video/Wan2.1",
            "commit": actual_commit,
            "component_class": "wan.modules.vace_model.BaseWanAttentionBlock",
            "block_index": args.block_index,
            "hint_index": hint_index,
            "vace_layer_indices": vace_layers,
            "checkpoint_prefix": prefix,
        },
        "weights": {
            "mode": "selective trained checkpoint" if checkpoint_metadata else "deterministic random initialization",
            "checkpoint": checkpoint_metadata,
        },
        "dynamic_shape": {
            "spatial_grid": [args.height, args.width],
            "frame_range": [1, args.max_frames],
            "token_range": [min_tokens, max_tokens],
            "context_range": [1, args.max_context_tokens],
            "has_vace_hint_input": hint_index is not None,
        },
        "validation": {
            "test_count": len(tests),
            "tests": tests,
            "max_absolute_error": max_abs["hidden_state"],
            "max_rmse": max_rmse["hidden_state"],
            "max_relative_rmse": max_relative_rmse["hidden_state"],
            "thresholds": {
                "max_absolute_error": args.max_absolute_error,
                "max_rmse": args.max_rmse,
                "max_relative_rmse": args.max_relative_rmse,
            },
            "passed": passed,
        },
        "quantization_setup": quantization_setup,
        "quantization_experiments": q_reports,
        "runtime": {
            "compute_precision": args.compute_precision,
            "conversion_seconds": conversion_seconds,
            "package_size_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
        },
        "limitations": [
            "Only one main block, with a 4x4 spatial test grid and 1-4 latent frames.",
            "Synthetic VACE hint; no complete conditioning path.",
            "Portable SDPA replaces upstream CUDA FlashAttention.",
            "No text encoder, full denoising schedule, temporal VAE, iPhone memory/speed/thermal measurement or video export.",
        ],
    }
    report_path = args.report or args.output.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
