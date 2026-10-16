#!/usr/bin/env python3
"""Audit official Wan VACE 14B checkpoint tensor sizes without downloading weights.

The script reads Hugging Face repository metadata and safetensors headers using
HTTP Range requests. It never downloads tensor payloads. Its output is a memory
budget input for a possible block-wise iOS runtime; it does not convert weights
or prove that the model can run on an iPhone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

DEFAULT_REPO = "Wan-AI/Wan2.1-VACE-14B"
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "RIFE60GhostGuard-VACE-Audit/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RIFE60GhostGuard-VACE-Audit/1.0",
            "Range": f"bytes={start}-{end}",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
        # Refuse a server/proxy that ignored Range: do not accidentally download
        # a multi-gigabyte checkpoint into CI memory.
        if response.status != 206:
            raise RuntimeError(f"Server did not honor byte range for {url} (HTTP {response.status})")
        return data


def safe_url(repo: str, revision: str, filename: str) -> str:
    from urllib.parse import quote
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote(filename, safe='/')}?download=true"


def safetensors_header(url: str) -> dict[str, Any]:
    prefix = fetch_range(url, 0, 7)
    if len(prefix) != 8:
        raise RuntimeError(f"Invalid safetensors header prefix at {url}")
    header_size = int.from_bytes(prefix, "little", signed=False)
    if header_size <= 1 or header_size > 128 * 1024 * 1024:
        raise RuntimeError(f"Refusing implausible safetensors header size {header_size} at {url}")
    raw = fetch_range(url, 8, 8 + header_size - 1)
    if len(raw) != header_size:
        raise RuntimeError(f"Truncated safetensors header at {url}: {len(raw)} != {header_size}")
    return json.loads(raw.decode("utf-8"))


def tensor_bytes(entry: dict[str, Any]) -> int:
    dtype = entry.get("dtype", "")
    shape = entry.get("shape", [])
    if dtype not in DTYPE_BYTES:
        raise RuntimeError(f"Unknown safetensors dtype {dtype!r}")
    elements = math.prod(int(x) for x in shape)
    return elements * DTYPE_BYTES[dtype]


def layer_bucket(name: str) -> str:
    import re
    patterns = (
        r"^(?:model\.)?diffusion_model\.blocks\.(\d+)\.",
        r"^(?:model\.)?blocks\.(\d+)\.",
        r"^(?:model\.)?transformer_blocks\.(\d+)\.",
        r"^(?:model\.)?vace_blocks\.(\d+)\.",
        r"^(?:model\.)?single_transformer_blocks\.(\d+)\.",
    )
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            return f"layer_{int(match.group(1)):03d}"
    return "non_layer_or_unrecognized"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Official Hugging Face repo ID")
    parser.add_argument("--revision", default="main", help="Pinned commit SHA recommended for reproducible audits")
    parser.add_argument("--output", required=True, help="JSON report output path")
    args = parser.parse_args()

    from pathlib import Path
    repo_api = f"https://huggingface.co/api/models/{args.repo}/tree/{args.revision}?recursive=true&expand=false"
    files = fetch_json(repo_api)
    safetensor_files = [f for f in files if f.get("type") == "file" and f.get("path", "").endswith(".safetensors")]
    if not safetensor_files:
        raise RuntimeError(f"No safetensors files found in {args.repo}@{args.revision}; refusing to guess checkpoint layout")

    all_tensors: dict[str, dict[str, Any]] = {}
    shard_reports = []
    for file_info in safetensor_files:
        filename = file_info["path"]
        url = safe_url(args.repo, args.revision, filename)
        header = safetensors_header(url)
        shard_tensor_count = 0
        shard_bytes = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            size = tensor_bytes(entry)
            all_tensors[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "bytes": size}
            shard_tensor_count += 1
            shard_bytes += size
        shard_reports.append({
            "file": filename,
            "file_bytes_from_repo_metadata": file_info.get("size"),
            "header_tensor_count": shard_tensor_count,
            "tensor_payload_bytes_from_shapes": shard_bytes,
            "header_only_download": True,
        })

    layer_sizes: dict[str, int] = defaultdict(int)
    layer_tensors: dict[str, int] = defaultdict(int)
    dtype_sizes: dict[str, int] = defaultdict(int)
    for name, entry in all_tensors.items():
        bucket = layer_bucket(name)
        layer_sizes[bucket] += entry["bytes"]
        layer_tensors[bucket] += 1
        dtype_sizes[entry["dtype"]] += entry["bytes"]

    ordered_layers = sorted(
        ({"name": name, "tensor_count": layer_tensors[name], "bytes": size}
         for name, size in layer_sizes.items()),
        key=lambda item: item["bytes"], reverse=True,
    )
    total_bytes = sum(entry["bytes"] for entry in all_tensors.values())
    report = {
        "repo": args.repo,
        "revision": args.revision,
        "method": "safetensors HTTP Range header reads only; tensor payloads not downloaded",
        "tensor_count": len(all_tensors),
        "total_tensor_payload_bytes": total_bytes,
        "estimated_all_weights_gib": total_bytes / (1024 ** 3),
        "theoretical_uniform_8bit_gib": total_bytes / (1024 ** 3) / 2,
        "theoretical_uniform_4bit_gib": total_bytes / (1024 ** 3) / 4,
        "quantization_estimates_exclude": [
            "quantization scales and zero points",
            "unquantized normalization/embedding tensors",
            "activations and attention intermediates",
            "VAE and text encoder",
            "Core ML runtime and iOS process overhead",
        ],
        "dtype_payload_bytes": dict(dtype_sizes),
        "largest_weight_groups": ordered_layers[:20],
        "unrecognized_group_bytes": layer_sizes.get("non_layer_or_unrecognized", 0),
        "shards": shard_reports,
        "can_fit_iphone_14_pro_max": False,
        "complete_pipeline_or_runtime_test": False,
        "notes": [
            "This report is an artifact-layout and lower-bound memory audit only.",
            "Uniform int4/int8 figures are theoretical storage bounds, not evidence that a supported quantized Core ML runtime exists.",
            "Layer grouping is based on recognized state-dict key patterns; inspect unrecognized groups before using the report to design streaming.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "repo": args.repo,
        "revision": args.revision,
        "tensor_count": len(all_tensors),
        "weight_gib": round(total_bytes / (1024 ** 3), 3),
        "uniform_int4_gib_lower_bound": round(total_bytes / (1024 ** 3) / 4, 3),
        "largest_groups": ordered_layers[:5],
        "report": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"VACE artifact audit failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
