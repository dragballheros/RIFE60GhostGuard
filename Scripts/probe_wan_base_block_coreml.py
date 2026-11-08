#!/usr/bin/env python3
"""Convert and validate one trained Wan VACE base transformer block in Core ML."""
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

GRID = (1, 2, 2)
CONTEXT_TOKENS = 8


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def max_abs_diff(left: Any, right: Any) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--block-index", type=int, default=0, help="index in the main transformer stack")
    parser.add_argument("--num-layers", type=int, default=30, help="full model layer count, used to derive default VACE hint placement")
    parser.add_argument("--block-prefix", default=None, help="defaults to blocks.<block-index>.")
    parser.add_argument("--vace-layer-indices", type=int, nargs="*", default=None, help="main block indices receiving VACE hints; defaults to every other layer starting at zero")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-commit", default=WAN_COMMIT)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-rmse", type=float, default=0.035)
    parser.add_argument("--max-absolute-error", type=float, default=0.20)
    parser.add_argument("--quantization-levels", nargs="*", choices=("int8", "int4"), default=["int8", "int4"])
    args = parser.parse_args()

    repo = args.wan_repo.resolve()
    if not (repo / "wan/modules/vace_model.py").is_file():
        parser.error(f"{repo} is not an upstream Wan2.1 checkout")
    actual_commit = git_head(repo)
    if args.expected_commit and actual_commit != args.expected_commit:
        parser.error(f"Wan source commit mismatch: expected {args.expected_commit}, got {actual_commit}")

    # Import only pinned upstream model modules, not the full distributed/T5 stack.
    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(repo / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(repo / "wan/modules")]
    sys.modules["wan"] = wan_package
    sys.modules["wan.modules"] = modules_package
    import wan.modules.model as wan_model
    from wan.modules.vace_model import BaseWanAttentionBlock

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    dim, ffn_dim, heads = VACE_1_3B["dim"], VACE_1_3B["ffn_dim"], VACE_1_3B["num_heads"]
    head_dim = dim // heads
    frames, height, width = GRID
    tokens = frames * height * width
    freqs = torch.cat([
        wan_model.rope_params(1024, head_dim - 4 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
        wan_model.rope_params(1024, 2 * (head_dim // 6)),
    ], dim=1)
    grid_sizes = torch.tensor([GRID], dtype=torch.long)
    seq_lens = torch.tensor([tokens], dtype=torch.long)
    rope_cos, rope_sin = build_fixed_rope(freqs, GRID)

    original_rope_apply = wan_model.rope_apply
    original_flash_attention = wan_model.flash_attention

    # Numerical check for the fixed-grid real-valued RoPE rewrite used by Core ML.
    rotary_sample = torch.randn(1, tokens, heads, head_dim)
    official_rotary = original_rope_apply(rotary_sample, grid_sizes, freqs)
    def fixed_rope_apply(x, _grid_sizes, _freqs):
        c = rope_cos.to(device=x.device, dtype=x.dtype).expand(1, tokens, 1, head_dim // 2)
        s = rope_sin.to(device=x.device, dtype=x.dtype).expand(1, tokens, 1, head_dim // 2)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)

    rotary_error = float(
        (official_rotary.float() - fixed_rope_apply(rotary_sample, grid_sizes, freqs).float()).abs().max().item()
    )

    def portable_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
        result = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False,
        )
        return result.transpose(1, 2).contiguous()

    wan_model.rope_apply = fixed_rope_apply
    wan_model.flash_attention = portable_attention
    if args.num_layers < 1 or not 0 <= args.block_index < args.num_layers:
        parser.error("--block-index must be within 0 .. --num-layers - 1")
    block_prefix = args.block_prefix or f"blocks.{args.block_index}."
    vace_layer_indices = (
        args.vace_layer_indices
        if args.vace_layer_indices is not None
        else list(range(0, args.num_layers, 2))
    )
    if len(set(vace_layer_indices)) != len(vace_layer_indices):
        parser.error("--vace-layer-indices must not contain duplicates")
    hint_index = (
        vace_layer_indices.index(args.block_index)
        if args.block_index in vace_layer_indices
        else None
    )

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
    checkpoint_metadata = load_safetensors_block(block, args.checkpoint, block_prefix)

    class FixedShapeBaseBlock(torch.nn.Module):
        def __init__(self, layer, block_hint_index):
            super().__init__()
            self.layer = layer
            self.block_hint_index = block_hint_index

        def forward(self, latent_tokens, time_modulation, text_context, vace_hint=None, context_scale=None):
            kwargs = {
                "hints": [],
                "context_scale": 1.0,
                "e": time_modulation,
                "seq_lens": seq_lens,
                "grid_sizes": grid_sizes,
                "freqs": freqs,
                "context": text_context,
                "context_lens": None,
            }
            if self.block_hint_index is not None:
                # The upstream base block indexes its hint by the position of
                # the corresponding main-layer in vace_layers. Only this selected
                # hint is semantically used by this individual block.
                kwargs["hints"] = [vace_hint] * (self.block_hint_index + 1)
                kwargs["context_scale"] = context_scale
            return self.layer(latent_tokens, **kwargs)

    wrapper = FixedShapeBaseBlock(block, hint_index).eval()
    latent_tokens = torch.randn(1, tokens, dim, dtype=torch.float32) * 0.1
    time_modulation = torch.randn(1, 6, dim, dtype=torch.float32) * 0.01
    text_context = torch.randn(1, CONTEXT_TOKENS, dim, dtype=torch.float32) * 0.1
    inputs_list = [latent_tokens, time_modulation, text_context]
    input_specs = [
        ct.TensorType(name="latent_tokens", shape=tuple(latent_tokens.shape), dtype=np.float32),
        ct.TensorType(name="time_modulation", shape=tuple(time_modulation.shape), dtype=np.float32),
        ct.TensorType(name="text_context", shape=tuple(text_context.shape), dtype=np.float32),
    ]
    names = ["latent_tokens", "time_modulation", "text_context"]
    vace_hint = None
    context_scale = None
    if hint_index is not None:
        vace_hint = torch.randn(1, tokens, dim, dtype=torch.float32) * 0.1
        context_scale = torch.tensor([1.0], dtype=torch.float32)
        inputs_list.extend([vace_hint, context_scale])
        input_specs.extend([
            ct.TensorType(name="vace_hint", shape=tuple(vace_hint.shape), dtype=np.float32),
            ct.TensorType(name="context_scale", shape=tuple(context_scale.shape), dtype=np.float32),
        ])
        names.extend(["vace_hint", "context_scale"])
    inputs = tuple(inputs_list)

    try:
        with torch.no_grad():
            started = time.perf_counter()
            reference = wrapper(*inputs)
            reference_seconds = time.perf_counter() - started
            traced = torch.jit.trace(wrapper, inputs, strict=False, check_trace=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=input_specs,
            outputs=[ct.TensorType(name="hidden_state")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
        )
        conversion_seconds = time.perf_counter() - started
        converted.save(str(args.output))

        prediction_inputs = {
            name: value.detach().cpu().numpy().astype(np.float32)
            for name, value in zip(names, inputs)
        }
        reference_numpy = reference.detach().cpu().numpy().astype(np.float32)

        def calculate_errors(output_map: dict[str, Any]) -> list[dict[str, Any]]:
            prediction = np.asarray(output_map["hidden_state"], dtype=np.float32)
            delta = prediction - reference_numpy
            return [{
                "name": "hidden_state",
                "shape": list(prediction.shape),
                "max_absolute_error": float(np.abs(delta).max()),
                "mean_absolute_error": float(np.abs(delta).mean()),
                "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            }]

        errors = calculate_errors(converted.predict(prediction_inputs))
        quantization_reports = []
        if args.quantization_levels:
            from coremltools.optimize.coreml import (
                OptimizationConfig,
                OpLinearQuantizerConfig,
                linear_quantize_weights,
            )
            for bits in args.quantization_levels:
                quantized_path = args.output.with_name(f"{args.output.stem}-{bits}-weights.mlpackage")
                started = time.perf_counter()
                try:
                    config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric", dtype=bits, granularity="per_channel"
                    ))
                    quantized = linear_quantize_weights(converted, config=config)
                    quantized.save(str(quantized_path))
                    q_errors = calculate_errors(quantized.predict(prediction_inputs))
                    quantization_reports.append({
                        "bits": int(bits[3:]),
                        "package_size_bytes": sum(
                            p.stat().st_size for p in quantized_path.rglob("*") if p.is_file()
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                        "outputs": q_errors,
                        "passed_current_float_reference_thresholds": all(
                            e["rmse"] <= args.max_rmse
                            and e["max_absolute_error"] <= args.max_absolute_error
                            for e in q_errors
                        ),
                    })
                except Exception as exc:
                    quantization_reports.append({
                        "bits": int(bits[3:]),
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.perf_counter() - started,
                        "passed_current_float_reference_thresholds": False,
                    })

        params = sum(p.numel() for p in block.parameters())
        fp16_bytes = sum(p.numel() * p.element_size() for p in block.parameters())
        report = {
            "probe": "official_vace_base_wan_attention_block_coreml_fixed_shape",
            "source": {
                "repository": "https://github.com/Wan-Video/Wan2.1",
                "commit": actual_commit,
                "component_class": "wan.modules.vace_model.BaseWanAttentionBlock",
                "main_block_index": args.block_index,
                "main_layer_count": args.num_layers,
                "checkpoint_prefix": block_prefix,
                "vace_layer_indices": vace_layer_indices,
                "vace_hint_index": hint_index,
            },
            "weights": {
                "initialization": "selective learned checkpoint tensors",
                "checkpoint_load": checkpoint_metadata,
                "parameter_count": params,
                "fp32_parameter_bytes": fp16_bytes,
                "fp16_parameter_payload_estimate_bytes": params * 2,
            },
            "test_shape": {
                "grid_fhw": list(GRID),
                "latent_tokens": tokens,
                "context_tokens": CONTEXT_TOKENS,
                "latent_tokens_shape": list(latent_tokens.shape),
                "time_modulation": list(time_modulation.shape),
                "text_context": list(text_context.shape),
                "vace_hint": list(vace_hint.shape) if vace_hint is not None else None,
                "context_scale": list(context_scale.shape) if context_scale is not None else None,
            },
            "validation": {
                "rotary_real_vs_official_complex_max_absolute_error": rotary_error,
                "outputs": errors,
                "pass_rmse_limit": args.max_rmse,
                "pass_max_absolute_error_limit": args.max_absolute_error,
                "passed": all(
                    e["rmse"] <= args.max_rmse
                    and e["max_absolute_error"] <= args.max_absolute_error
                    for e in errors
                ),
            },
            "runtime": {
                "pytorch_reference_seconds": reference_seconds,
                "coreml_conversion_seconds": conversion_seconds,
                "fp16_package_path": str(args.output),
                "fp16_package_size_bytes": sum(
                    p.stat().st_size for p in args.output.rglob("*") if p.is_file()
                ),
            },
            "weight_quantization_experiments": quantization_reports,
            "limitations": [
                "One base transformer block only; the full 30/32-block stack is not exported or executed.",
                "Fixed four-latent-token grid and short context; not a realistic video sequence.",
                "Uses portable SDPA in place of upstream CUDA FlashAttention.",
                "The VACE hint is synthetic input; cross-block hint generation/order is not end-to-end validated.",
                "No iPhone memory, speed, thermal or visual-quality validation.",
            ],
        }
        report_path = args.report or args.output.with_suffix(".validation.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(report, indent=2) + "\n"
        report_path.write_text(rendered, encoding="utf-8")
        print(rendered)
        return 0 if report["validation"]["passed"] else 2
    finally:
        wan_model.rope_apply = original_rope_apply
        wan_model.flash_attention = original_flash_attention


if __name__ == "__main__":
    raise SystemExit(main())
