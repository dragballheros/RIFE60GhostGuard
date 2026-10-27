#!/usr/bin/env python3
"""Reject Swift that sets outputBackings and also calls batch predictions.

Core ML raises "The output backing is not supported in a batch prediction"
when MLPredictionOptions.outputBackings is passed to predictions(from:).
Single prediction(from:options:) may use output backings. A batch call must not.
"""
from __future__ import annotations

import pathlib
import sys


def strip_comments_and_strings(source: str) -> str:
    out = []
    i = 0
    n = len(source)
    while i < n:
        if source.startswith("//", i):
            i = source.find("\n", i)
            if i < 0:
                break
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if source[i] == '"':
            i += 1
            while i < n and source[i] != '"':
                if source[i] == "\\":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        out.append(source[i])
        i += 1
    return "".join(out)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    failures = []
    for path in sorted(root.rglob("*.swift")):
        if any(part in {".build", "Vendor", "DerivedData"} for part in path.parts):
            continue
        code = strip_comments_and_strings(path.read_text(encoding="utf-8", errors="replace"))
        has_backing = "outputBackings" in code
        has_batch = "predictions(from:" in code
        if has_backing and has_batch:
            failures.append(str(path.relative_to(root)))
    if failures:
        print("outputBackings and batch predictions(from:) must not appear in the same Swift file:")
        for item in failures:
            print(f"  {item}")
        return 1
    print("No Swift file pairs outputBackings with a batch prediction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
