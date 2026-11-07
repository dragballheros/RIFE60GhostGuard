#!/usr/bin/env python3
"""Reject a Swift function that both sets outputBackings and calls batch predictions.

Core ML raises "The output backing is not supported in a batch prediction"
when MLPredictionOptions.outputBackings is passed to predictions(from:).
A single prediction(from:options:) may use output backings. A batch call must not.
The two may live in the same file only in different functions.
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
            newline = source.find("\n", i)
            out.append("\n" if newline >= 0 else "")
            i = n if newline < 0 else newline + 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            chunk = source[i: n if end < 0 else end + 2]
            out.append("\n" * chunk.count("\n"))
            i = n if end < 0 else end + 2
            continue
        if source[i] == '"':
            i += 1
            while i < n and source[i] != '"':
                if source[i] == "\\":
                    i += 2
                else:
                    i += 1
            i = min(i + 1, n)
            out.append('""')
            continue
        out.append(source[i])
        i += 1
    return "".join(out)


def functions(code: str) -> list[tuple[str, str]]:
    found = []
    depth = 0
    index = 0
    length = len(code)
    while index < length:
        if code.startswith("func ", index) or code.startswith("func\n", index):
            name_end = code.find("(", index)
            name = code[index + 5: name_end if name_end > 0 else index + 5].strip()
            start_depth = depth
            body_start = index
            cursor = index
            started = False
            while cursor < length:
                char = code[cursor]
                if char == "{":
                    depth += 1
                    started = True
                elif char == "}":
                    depth -= 1
                    if started and depth == start_depth:
                        found.append((name, code[body_start: cursor + 1]))
                        index = cursor + 1
                        break
                cursor += 1
            else:
                break
            continue
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
        index += 1
    return found


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    failures = []
    for path in sorted(root.rglob("*.swift")):
        if any(part in {".build", "Vendor", "DerivedData"} for part in path.parts):
            continue
        code = strip_comments_and_strings(path.read_text(encoding="utf-8", errors="replace"))
        relative = str(path.relative_to(root))
        for name, body in functions(code):
            if "outputBackings" in body and "predictions(from:" in body:
                failures.append(f"{relative} :: {name}")
    if failures:
        print("A function sets outputBackings and also calls predictions(from:):")
        for item in failures:
            print(f"  {item}")
        return 1
    print("No Swift function pairs outputBackings with a batch prediction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
