#!/usr/bin/env python3
"""Validate DeepCreamPy v2 bar/mosaic checkpoint files without modifying them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHECKPOINTS = {
    "bar": "Train_775000",
    "mosaic": "Train_290000",
}
REQUIRED_PREFIXES = ("G_en/", "G_de/", "CB1/")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Directory containing bar/ and mosaic/ checkpoint folders",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        fail(f"checkpoint root is not a directory: {root}")

    validated: dict[str, Path] = {}
    for family, prefix_name in CHECKPOINTS.items():
        prefix = root / family / prefix_name
        expected = [
            Path(str(prefix) + ".meta"),
            Path(str(prefix) + ".index"),
            Path(str(prefix) + ".data-00000-of-00001"),
        ]
        missing = [
            str(path) for path in expected
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            fail(
                f"{family} checkpoint is incomplete; missing/empty: "
                + ", ".join(missing)
            )

        print(f"OK {family}:")
        for path in expected:
            print(f"  {path.name}: {path.stat().st_size:,} bytes")
        validated[family] = prefix

    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        print("\nTensorFlow is not installed; file-layout validation only.")
        print(
            "No claim is made that the checkpoint can be restored "
            "or that inference works."
        )
        return 0

    # Reads checkpoint tensor names and shapes without importing the training graph.
    for family, prefix in validated.items():
        try:
            variables = tf.train.list_variables(str(prefix))
        except Exception as exc:
            fail(f"TensorFlow could not read {family} checkpoint {prefix}: {exc}")

        names = [name for name, _shape in variables]
        missing_groups = [
            group for group in REQUIRED_PREFIXES
            if not any(name.startswith(group) for name in names)
        ]
        if missing_groups:
            fail(
                f"{family} checkpoint is missing expected variable groups: "
                f"{missing_groups}"
            )

        counts = {
            group: sum(name.startswith(group) for name in names)
            for group in REQUIRED_PREFIXES
        }
        print(f"\nTensorFlow inventory — {family}: {len(variables)} tensors")
        print("  " + ", ".join(f"{group} {count}" for group, count in counts.items()))
        for name, shape in variables:
            if name.startswith(REQUIRED_PREFIXES):
                print(f"  {name}: {shape}")

    print("\nPASS: checkpoint files and expected variable groups were found.")
    print(
        "This is a checkpoint preflight only; it does not prove graph restoration "
        "or inference parity."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
