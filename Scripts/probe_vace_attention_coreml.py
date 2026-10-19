#!/usr/bin/env python3
"""Probe a Core ML replacement kernel for Wan/VACE attention.

This tests only fixed-shape scaled-dot-product attention. It is not a VACE
transformer export and does not establish efficient long-sequence attention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--key-tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    if min(args.query_tokens, args.key_tokens, args.heads, args.head_dim) < 1:
        parser.error("all tensor dimensions must be positive")

    import numpy as np
    import torch
    import torch.nn.functional as F
    import coremltools as ct

    class AttentionKernel(torch.nn.Module):
        def forward(self, query, key, value):
            return F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )

    torch.manual_seed(20261017)
    shape_q = (1, args.heads, args.query_tokens, args.head_dim)
    shape_kv = (1, args.heads, args.key_tokens, args.head_dim)
    query = torch.randn(shape_q, dtype=torch.float32) * 0.1
    key = torch.randn(shape_kv, dtype=torch.float32) * 0.1
    value = torch.randn(shape_kv, dtype=torch.float32) * 0.1
    model = AttentionKernel().eval()
    with torch.no_grad():
        reference = model(query, key, value)
        traced = torch.jit.trace(model, (query, key, value), strict=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="query", shape=shape_q, dtype=np.float32),
            ct.TensorType(name="key", shape=shape_kv, dtype=np.float32),
            ct.TensorType(name="value", shape=shape_kv, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="attended")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    converted.save(str(args.output))
    prediction = converted.predict({
        "query": query.numpy(),
        "key": key.numpy(),
        "value": value.numpy(),
    })["attended"]
    ref = reference.detach().cpu().numpy()
    delta = prediction - ref
    report = {
        "probe": "vace_attention_sdpa_kernel_fixed_shape",
        "complete_vace_pipeline": False,
        "official_flash_attention_replaced": True,
        "long_sequence_efficiency_validated": False,
        "input_shapes": {
            "query": list(shape_q),
            "key": list(shape_kv),
            "value": list(shape_kv),
        },
        "output_shape": list(ref.shape),
        "validation_input": "deterministic random tensors, seed 20261017",
        "max_absolute_error": float(np.abs(delta).max()),
        "mean_absolute_error": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "package_path": str(args.output),
        "note": (
            "Kernel-only Core ML smoke test for a candidate SDPA replacement. "
            "It does not include Wan rotary embeddings, VACE blocks, checkpoint "
            "weights, variable-length masks, or long-sequence memory/performance."
        ),
    }
    report_path = args.output.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
