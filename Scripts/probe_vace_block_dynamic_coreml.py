#!/usr/bin/env python3
"""Prototype a dynamic temporal-axis Core ML wrapper for one Wan VACE conditioning block.

The spatial latent grid is fixed for this experiment, while the number of latent
frames varies. This is a shape/operator experiment, not a full video editor.
"""
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
    max_abs_diff,
)

HEIGHT = 4
WIDTH = 4
MIN_FRAMES = 1
MAX_FRAMES = 4
CONTEXT_MIN = 1
CONTEXT_MAX = 16


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def rope_apply_explicit(x, cos, sin):
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def portable_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
    # Every dynamic test sample is unpadded, so its full token/context ranges are valid.
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous()


class DynamicVACEBlock(torch.nn.Module):
    """Same learned VACE submodules, with temporal RoPE passed as explicit inputs."""

    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer

    def forward(
        self,
        conditioning,
        latent_tokens,
        time_modulation,
        text_context,
        rope_cos,
        rope_sin,
    ):
        layer = self.layer
        batch = latent_tokens.shape[0]
        dim = VACE_1_3B["dim"]
        heads = VACE_1_3B["num_heads"]
        head_dim = dim // heads

        # VaceWanAttentionBlock's VACE input projection and ordinary Wan block
        # modulation/attention/FFN semantics, rewritten to consume dynamic RoPE.
        x = layer.before_proj(conditioning) + latent_tokens
        e = (layer.modulation + time_modulation).chunk(6, dim=1)

        self_attn = layer.self_attn
        normalized = layer.norm1(x).float() * (1 + e[1]) + e[0]
        q = self_attn.norm_q(self_attn.q(normalized)).reshape(batch, -1, heads, head_dim)
        k = self_attn.norm_k(self_attn.k(normalized)).reshape(batch, -1, heads, head_dim)
        v = self_attn.v(normalized).reshape(batch, -1, heads, head_dim)
        q = rope_apply_explicit(q, rope_cos, rope_sin)
        k = rope_apply_explicit(k, rope_cos, rope_sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).contiguous().reshape(batch, -1, dim)
        x = x + self_attn.o(y) * e[2]

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

        y = layer.ffn(layer.norm2(x).float() * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x, layer.after_proj(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-commit", default=WAN_COMMIT)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--block-prefix", default="vace_blocks.0.")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--max-context-tokens", type=int, default=CONTEXT_MAX)
    parser.add_argument("--compute-precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-absolute-error", type=float, default=0.20)
    parser.add_argument("--max-rmse", type=float, default=0.035)
    parser.add_argument("--max-relative-rmse", type=float, default=0.02)
    args = parser.parse_args()

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

    # Avoid Wan's package initializer pulling the entire T5/distributed runtime.
    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(repo / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(repo / "wan/modules")]
    sys.modules["wan"] = wan_package
    sys.modules["wan.modules"] = modules_package
    import wan.modules.model as wan_model
    from wan.modules.vace_model import VaceWanAttentionBlock

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dim, ffn_dim, heads = VACE_1_3B["dim"], VACE_1_3B["ffn_dim"], VACE_1_3B["num_heads"]
    head_dim = dim // heads
    max_tokens = args.max_frames * args.height * args.width
    min_tokens = args.height * args.width

    freqs = torch.cat([
        wan_model.rope_params(1024, head_dim - 4 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
    ], dim=1)

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
    checkpoint_metadata = None
    if args.checkpoint is not None:
        checkpoint_metadata = load_safetensors_block(block, args.checkpoint, args.block_prefix)
    else:
        # The upstream VACE input/output projections are zero-initialized; nonzero
        # test weights ensure those operations are not bypassed during conversion.
        with torch.no_grad():
            for projection_name in ("before_proj", "after_proj"):
                projection = getattr(block, projection_name)
                torch.nn.init.normal_(projection.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(projection.bias)

    dynamic_model = DynamicVACEBlock(block).eval()

    # Core ML's RangeDim provides one shared flexible temporal token dimension for
    # conditioning, latent, and RoPE tensors. The spatial grid stays fixed.
    token_dim = ct.RangeDim(
        lower_bound=min_tokens,
        upper_bound=max_tokens,
        default=min_tokens,
        symbol="latent_tokens",
    )
    context_dim = ct.RangeDim(
        lower_bound=1,
        upper_bound=args.max_context_tokens,
        default=min(args.max_context_tokens, 8),
        symbol="context_tokens",
    )

    sample_frames = 1
    sample_tokens = sample_frames * args.height * args.width
    sample_context_tokens = min(args.max_context_tokens, 8)
    sample_rope_cos, sample_rope_sin = build_fixed_rope(freqs, (sample_frames, args.height, args.width))
    example_inputs = (
        torch.randn(1, sample_tokens, dim, dtype=torch.float32) * 0.1,
        torch.randn(1, sample_tokens, dim, dtype=torch.float32) * 0.1,
        torch.randn(1, 6, dim, dtype=torch.float32) * 0.01,
        torch.randn(1, sample_context_tokens, dim, dtype=torch.float32) * 0.1,
        sample_rope_cos,
        sample_rope_sin,
    )

    traced = torch.jit.trace(dynamic_model, example_inputs, strict=False, check_trace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="conditioning", shape=(1, token_dim, dim), dtype=np.float32),
            ct.TensorType(name="latent_tokens", shape=(1, token_dim, dim), dtype=np.float32),
            ct.TensorType(name="time_modulation", shape=(1, 6, dim), dtype=np.float32),
            ct.TensorType(name="text_context", shape=(1, context_dim, dim), dtype=np.float32),
            ct.TensorType(name="rope_cos", shape=(1, token_dim, 1, head_dim // 2), dtype=np.float32),
            ct.TensorType(name="rope_sin", shape=(1, token_dim, 1, head_dim // 2), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="conditioned_context"),
            ct.TensorType(name="skip_hint"),
        ],
        compute_precision=(
            ct.precision.FLOAT16 if args.compute_precision == "fp16" else ct.precision.FLOAT32
        ),
        minimum_deployment_target=ct.target.iOS18,
    )
    conversion_seconds = time.perf_counter() - started
    converted.save(str(args.output))

    # Reference path uses the original upstream block for each tested frame count,
    # with only its global RoPE/FlashAttention kernels replaced by portable equivalents.
    original_rope_apply = wan_model.rope_apply
    original_flash_attention = wan_model.flash_attention
    tests = []
    max_abs_by_output = {"conditioned_context": 0.0, "skip_hint": 0.0}
    max_rmse_by_output = {"conditioned_context": 0.0, "skip_hint": 0.0}
    max_relative_rmse_by_output = {"conditioned_context": 0.0, "skip_hint": 0.0}

    try:
        for frame_count in range(1, args.max_frames + 1):
            token_count = frame_count * args.height * args.width
            cos, sin = build_fixed_rope(freqs, (frame_count, args.height, args.width))
            for context_count in sorted(set((1, min(5, args.max_context_tokens), min(8, args.max_context_tokens), args.max_context_tokens))):
                context_count = max(1, context_count)
                torch.manual_seed(args.seed + frame_count * 100 + context_count)
                inputs = (
                    torch.randn(1, token_count, dim, dtype=torch.float32) * 0.1,
                    torch.randn(1, token_count, dim, dtype=torch.float32) * 0.1,
                    torch.randn(1, 6, dim, dtype=torch.float32) * 0.01,
                    torch.randn(1, context_count, dim, dtype=torch.float32) * 0.1,
                    cos, sin,
                )

                def current_rope(x, _grid_sizes, _freqs):
                    return rope_apply_explicit(x, cos.to(x.device, x.dtype), sin.to(x.device, x.dtype))

                wan_model.rope_apply = current_rope
                wan_model.flash_attention = portable_attention
                with torch.no_grad():
                    reference = block(
                        inputs[0],
                        inputs[1],
                        e=inputs[2],
                        seq_lens=torch.tensor([token_count], dtype=torch.long),
                        grid_sizes=torch.tensor([[frame_count, args.height, args.width]], dtype=torch.long),
                        freqs=freqs,
                        context=inputs[3],
                        context_lens=None,
                    )
                    predicted = converted.predict({
                        "conditioning": inputs[0].numpy(),
                        "latent_tokens": inputs[1].numpy(),
                        "time_modulation": inputs[2].numpy(),
                        "text_context": inputs[3].numpy(),
                        "rope_cos": cos.numpy(),
                        "rope_sin": sin.numpy(),
                    })
                output_errors = []
                for name, ref_tensor in zip(("conditioned_context", "skip_hint"), reference):
                    prediction = np.asarray(predicted[name], dtype=np.float32)
                    ref_array = ref_tensor.detach().cpu().numpy().astype(np.float32)
                    delta = prediction - ref_array
                    max_abs = float(np.abs(delta).max())
                    rmse = float(np.sqrt(np.mean(np.square(delta))))
                    reference_rms = float(np.sqrt(np.mean(np.square(ref_array))))
                    relative_rmse = rmse / max(reference_rms, 1e-8)
                    max_abs_by_output[name] = max(max_abs_by_output[name], max_abs)
                    max_rmse_by_output[name] = max(max_rmse_by_output[name], rmse)
                    max_relative_rmse_by_output[name] = max(
                        max_relative_rmse_by_output[name], relative_rmse
                    )
                    output_errors.append({
                        "name": name,
                        "shape": list(prediction.shape),
                        "max_absolute_error": max_abs,
                        "rmse": rmse,
                        "reference_rms": reference_rms,
                        "relative_rmse": relative_rmse,
                    })
                tests.append({
                    "frames": frame_count,
                    "latent_tokens": token_count,
                    "context_tokens": context_count,
                    "outputs": output_errors,
                })
    finally:
        wan_model.rope_apply = original_rope_apply
        wan_model.flash_attention = original_flash_attention

    # Baseline gate is deliberately numerical only. Passing does not guarantee
    # memory or runtime on the phone; those remain on-device gates.
    max_allowed_abs = args.max_absolute_error
    max_allowed_rmse = args.max_rmse
    max_allowed_relative_rmse = args.max_relative_rmse
    passed = all(
        max_abs_by_output[name] <= max_allowed_abs
        and max_rmse_by_output[name] <= max_allowed_rmse
        and max_relative_rmse_by_output[name] <= max_allowed_relative_rmse
        for name in max_abs_by_output
    )
    report = {
        "probe": "wan_vace_block_coreml_dynamic_temporal_axis",
        "source": {
            "repository": "https://github.com/Wan-Video/Wan2.1",
            "commit": actual_commit,
            "component_class": "wan.modules.vace_model.VaceWanAttentionBlock",
        },
        "weights": {
            "mode": "selective trained checkpoint" if checkpoint_metadata else "deterministic random initialization",
            "checkpoint": checkpoint_metadata,
            "seed": args.seed,
        },
        "dynamic_shape": {
            "spatial_latent_grid": [args.height, args.width],
            "minimum_frames": 1,
            "maximum_frames": args.max_frames,
            "token_range": [min_tokens, max_tokens],
            "context_range": [1, args.max_context_tokens],
            "input_shapes": {
                "conditioning": ["1", "latent_tokens", dim],
                "latent_tokens": ["1", "latent_tokens", dim],
                "time_modulation": [1, 6, dim],
                "text_context": ["1", "context_tokens", dim],
                "rope_cos": ["1", "latent_tokens", 1, head_dim // 2],
                "rope_sin": ["1", "latent_tokens", 1, head_dim // 2],
            },
        },
        "validation": {
            "test_count": len(tests),
            "tests": tests,
            "max_absolute_error_by_output": max_abs_by_output,
            "max_rmse_by_output": max_rmse_by_output,
            "max_relative_rmse_by_output": max_relative_rmse_by_output,
            "max_allowed_absolute_error": max_allowed_abs,
            "max_allowed_rmse": max_allowed_rmse,
            "max_allowed_relative_rmse": max_allowed_relative_rmse,
            "passed": passed,
        },
        "runtime": {
            "compute_precision": args.compute_precision,
            "conversion_seconds": conversion_seconds,
            "package_path": str(args.output),
            "package_size_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()),
        },
        "limitations": [
            "Spatial token grid is only 4x4 by default, intentionally smaller than real 480p latents.",
            "Temporal shape varies from 1 to 4 latent frames; not an end-to-end clip or a realistic attention-memory benchmark.",
            "Portable SDPA substitutes for the official CUDA FlashAttention kernel.",
            "Random weights by default; checkpoint-backed mode must be separately run to validate learned weights.",
            "No text encoder, full denoising loop, temporal VAE, mask preprocessing, iPhone memory/thermal measurement, or final video export.",
        ],
    }
    report_path = args.report or args.output.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
