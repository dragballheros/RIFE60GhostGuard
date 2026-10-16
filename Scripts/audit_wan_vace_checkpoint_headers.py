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


def ideal_quantized_payload_bytes(elements: int, bits_per_element: int) -> float:
    """Return ideal packed payload bytes, excluding all quantization metadata."""
    if elements < 0:
        raise ValueError("elements must be non-negative")
    if bits_per_element <= 0:
        raise ValueError("bits_per_element must be positive")
    return elements * bits_per_element / 8


def layer_bucket(name: str) -> str:
    """Preserve block namespace so separately indexed stacks are not merged."""
    import re
    patterns = (
        ("diffusion_model_blocks", r"^(?:model\.)?diffusion_model\.blocks\.(\d+)\."),
        ("blocks", r"^(?:model\.)?blocks\.(\d+)\."),
        ("transformer_blocks", r"^(?:model\.)?transformer_blocks\.(\d+)\."),
        ("vace_blocks", r"^(?:model\.)?vace_blocks\.(\d+)\."),
        ("single_transformer_blocks", r"^(?:model\.)?single_transformer_blocks\.(\d+)\."),
    )
    for namespace, pattern in patterns:
        match = re.match(pattern, name)
        if match:
            return f"{namespace}_layer_{int(match.group(1)):03d}"
    return "module_" + name.split(".", 1)[0]

def resolve_revision(repo: str, revision: str) -> str:
    """Resolve branch/tag names to an immutable Hugging Face commit SHA."""
    from urllib.parse import quote
    url = f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}"
    metadata = fetch_json(url)
    sha = metadata.get("sha")
    if not isinstance(sha, str) or len(sha) != 40:
        raise RuntimeError(f"Could not resolve {repo}@{revision} to a 40-character commit SHA")
    return sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Official Hugging Face repo ID")
    parser.add_argument("--revision", default="main", help="Pinned commit SHA recommended for reproducible audits")
    parser.add_argument("--output", required=True, help="JSON report output path")
    args = parser.parse_args()

    from pathlib import Path
    resolved_sha = resolve_revision(args.repo, args.revision)
    repo_api = f"https://huggingface.co/api/models/{args.repo}/tree/{resolved_sha}?recursive=true&expand=false"
    files = fetch_json(repo_api)
    safetensor_files = [f for f in files if f.get("type") == "file" and f.get("path", "").endswith(".safetensors")]
    if not safetensor_files:
        raise RuntimeError(f"No safetensors files found in {args.repo}@{args.revision}; refusing to guess checkpoint layout")

    all_tensors: dict[str, dict[str, Any]] = {}
    duplicate_tensor_names: list[str] = []
    shard_reports = []
    for file_info in safetensor_files:
        filename = file_info["path"]
        url = safe_url(args.repo, resolved_sha, filename)
        header = safetensors_header(url)
        shard_tensor_count = 0
        shard_bytes = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            size = tensor_bytes(entry)
            if name in all_tensors:
                duplicate_tensor_names.append(name)
                continue
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

    if duplicate_tensor_names:
        raise RuntimeError(f"Duplicate tensor keys across checkpoint shards: {duplicate_tensor_names[:10]}")

    layer_sizes: dict[str, int] = defaultdict(int)
    layer_tensors: dict[str, int] = defaultdict(int)
    layer_elements: dict[str, int] = defaultdict(int)
    dtype_sizes: dict[str, int] = defaultdict(int)
    dtype_elements: dict[str, int] = defaultdict(int)
    for name, entry in all_tensors.items():
        bucket = layer_bucket(name)
        elements = math.prod(int(x) for x in entry["shape"])
        layer_sizes[bucket] += entry["bytes"]
        layer_tensors[bucket] += 1
        layer_elements[bucket] += elements
        dtype_sizes[entry["dtype"]] += entry["bytes"]
        dtype_elements[entry["dtype"]] += elements

    ordered_layers = sorted(
        ({
            "name": name,
            "tensor_count": layer_tensors[name],
            "elements": layer_elements[name],
            "source_bytes": size,
            "theoretical_int8_payload_gib": ideal_quantized_payload_bytes(layer_elements[name], 8) / (1024 ** 3),
            "theoretical_int4_payload_gib": ideal_quantized_payload_bytes(layer_elements[name], 4) / (1024 ** 3),
        } for name, size in layer_sizes.items()),
        key=lambda item: item["source_bytes"], reverse=True,
    )
    namespace_summary: dict[str, dict[str, int]] = defaultdict(lambda: {"group_count": 0, "tensor_count": 0, "source_bytes": 0})
    for group in ordered_layers:
        group_name = group["name"]
        namespace = group_name.rsplit("_layer_", 1)[0] if "_layer_" in group_name else group_name
        namespace_summary[namespace]["group_count"] += 1
        namespace_summary[namespace]["tensor_count"] += group["tensor_count"]
        namespace_summary[namespace]["source_bytes"] += group["source_bytes"]

    total_bytes = sum(entry["bytes"] for entry in all_tensors.values())
    total_elements = sum(math.prod(int(x) for x in entry["shape"]) for entry in all_tensors.values())
    # Ideal payload bounds are calculated from tensor element counts, not source bytes:
    # source tensors are mixed F32/BF16, so dividing source bytes by 2 or 4 is incorrect.
    report = {
        "repo": args.repo,
        "requested_revision": args.revision,
        "resolved_commit_sha": resolved_sha,
        "method": "Safetensors HTTP Range header reads only; tensor payloads not downloaded",
        "tensor_count": len(all_tensors),
        "total_tensor_elements": total_elements,
        "total_tensor_payload_bytes": total_bytes,
        "estimated_all_weights_gib": total_bytes / (1024 ** 3),
        "theoretical_uniform_int8_payload_gib": ideal_quantized_payload_bytes(total_elements, 8) / (1024 ** 3),
        "theoretical_uniform_int4_payload_gib": ideal_quantized_payload_bytes(total_elements, 4) / (1024 ** 3),
        "theoretical_uniform_int8_payload_bytes": ideal_quantized_payload_bytes(total_elements, 8),
        "theoretical_uniform_int4_payload_bytes": ideal_quantized_payload_bytes(total_elements, 4),
        "quantization_estimates_exclude": [
            "quantization scales and zero points",
            "unquantized normalization/embedding tensors",
            "activations and attention intermediates",
            "VAE and text encoder",
            "Core ML runtime and iOS process overhead",
        ],
        "dtype_payload_bytes": dict(dtype_sizes),
        "dtype_element_counts": dict(dtype_elements),
        "weight_group_count": len(ordered_layers),
        "transformer_layer_group_count": sum("_layer_" in name for name in layer_sizes),
        "layer_namespace_summary": dict(namespace_summary),
        "largest_weight_groups": ordered_layers[:20],
        "all_weight_groups": ordered_layers,
        "named_module_group_source_bytes": {
            name: size for name, size in layer_sizes.items() if name.startswith("module_")
        },
        "shards": shard_reports,
        "can_fit_iphone_14_pro_max": False,
        "complete_pipeline_or_runtime_test": False,
        "notes": [
            "This report is an artifact-layout and lower-bound memory audit only.",
            "Uniform int4/int8 figures are theoretical storage bounds, not evidence that a supported quantized Core ML runtime exists.",
            "Block grouping preserves namespace to avoid merging same-index blocks from separate transformer stacks.",
            "Review all_weight_groups and named_module_group_source_bytes before designing any streaming runtime.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "repo": args.repo,
        "requested_revision": args.revision,
        "resolved_commit_sha": resolved_sha,
        "tensor_count": len(all_tensors),
        "total_tensor_elements": total_elements,
        "weight_gib": round(total_bytes / (1024 ** 3), 3),
        "uniform_int8_payload_gib_lower_bound": round(total_elements / (1024 ** 3), 3),
        "uniform_int4_payload_gib_lower_bound": round(total_elements * 0.5 / (1024 ** 3), 3),
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
