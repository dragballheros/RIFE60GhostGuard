#!/usr/bin/env python3
"""Audit a locally downloaded Wan2.1-VACE checkpoint without executing pickle files.

Usage:
  python3 Scripts/inspect_vace_checkpoint.py /path/to/Wan2.1-VACE-1.3B
  python3 Scripts/inspect_vace_checkpoint.py /path/to/checkpoint --json report.json

The script intentionally does not call torch.load() on .pt/.pth/.bin files. Model
checkpoints can contain executable pickle payloads; this first-pass audit only
hashes them and inspects JSON indexes. For safetensors, tensor metadata is read
with the optional safetensors package (no tensor materialization).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
METADATA_NAMES = {"config.json", "model_index.json", "tokenizer_config.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        return {"metadata_status": "not inspected (install safetensors to inspect tensor headers)"}

    tensors: list[dict[str, Any]] = []
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                view = handle.get_slice(key)
                tensors.append({
                    "name": key,
                    "shape": list(view.get_shape()),
                    "dtype": str(view.get_dtype()),
                })
    except Exception as exc:  # report malformed/unsupported file without hiding the audit
        return {"metadata_status": "error", "metadata_error": str(exc)}

    dtype_elements: dict[str, int] = {}
    total_elements = 0
    for item in tensors:
        count = 1
        for dimension in item["shape"]:
            count *= int(dimension)
        total_elements += count
        dtype_elements[item["dtype"]] = dtype_elements.get(item["dtype"], 0) + count
    return {
        "metadata_status": "ok",
        "tensor_count": len(tensors),
        "total_elements": total_elements,
        "elements_by_dtype": dtype_elements,
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path, help="also write the report to this path")
    args = parser.parse_args()

    root = args.checkpoint_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"checkpoint directory does not exist: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        parser.error(f"checkpoint directory is empty: {root}")

    report: dict[str, Any] = {
        "checkpoint_dir": str(root),
        "warning": "File hashes identify artifacts but do not prove publisher authenticity. Verify the source revision and license separately.",
        "files": [],
        "weight_files": [],
        "index_files": [],
    }
    for path in files:
        relative = path.relative_to(root).as_posix()
        item: dict[str, Any] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".safetensors":
            item["safetensors"] = inspect_safetensors(path)
            report["weight_files"].append(relative)
        elif path.suffix.lower() in WEIGHT_SUFFIXES:
            item["format_note"] = "hashed only; pickle-based formats were not deserialized"
            report["weight_files"].append(relative)
        elif path.name.endswith(".index.json"):
            try:
                index = json.loads(path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map", {})
                item["index_tensor_count"] = len(weight_map) if isinstance(weight_map, dict) else None
                item["shard_count"] = len(set(weight_map.values())) if isinstance(weight_map, dict) else None
                item["index_status"] = "ok"
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                item["index_status"] = "error"
                item["index_error"] = str(exc)
            report["index_files"].append(relative)
        elif path.name in METADATA_NAMES or path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
                item["json_status"] = "ok"
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                item["json_status"] = "error"
                item["json_error"] = str(exc)
        report["files"].append(item)

    report["file_count"] = len(files)
    report["total_size_bytes"] = sum(item["size_bytes"] for item in report["files"])
    report["weight_file_count"] = len(report["weight_files"])

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n", encoding="utf-8")

    bad_indexes = [
        item for item in report["files"]
        if item.get("index_status") == "error" or item.get("json_status") == "error"
    ]
    return 1 if bad_indexes else 0


if __name__ == "__main__":
    sys.exit(main())
