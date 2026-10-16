#!/usr/bin/env python3
"""Run deterministic smoke inference directly from a DeepCreamPy v2 TF checkpoint.

Use a TensorFlow 1.x-compatible environment. This imports the checkpoint's own
MetaGraph, so its fixed training batch dimension is respected. It intentionally
does not convert or modify the checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CHECKPOINTS = {"bar": "Train_775000", "mosaic": "Train_290000"}
INPUT_NAMES = ("Placeholder:0", "Placeholder_1:0", "Placeholder_2:0")
OUTPUT_NAME = "add:0"


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def tensor_shape(tensor, name: str) -> list[int]:
    dims = tensor.shape.as_list()
    if len(dims) != 4 or any(dim is None for dim in dims[1:]):
        die(f"{name} has unsupported shape {dims}; expected fixed NHWC rank-4 tensor")
    return [int(dim) if dim is not None else -1 for dim in dims]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path,
                        help="Directory containing bar/ and mosaic/ checkpoints")
    parser.add_argument("--family", choices=("bar", "mosaic", "both"), default="both")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional directory to write deterministic .npz parity fixtures")
    args = parser.parse_args()

    try:
        import tensorflow.compat.v1 as tf
    except Exception as exc:
        die(f"TensorFlow 1.x-compatible API is required: {exc}")

    tf.disable_eager_execution()
    root = args.root.expanduser().resolve()
    families = list(CHECKPOINTS) if args.family == "both" else [args.family]
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        prefix = root / family / CHECKPOINTS[family]
        for suffix in (".meta", ".index", ".data-00000-of-00001"):
            path = Path(str(prefix) + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                die(f"{family}: missing or empty checkpoint component: {path}")

        graph = tf.Graph()
        with graph.as_default():
            saver = tf.train.import_meta_graph(str(prefix) + ".meta", clear_devices=True)
            try:
                inputs = [graph.get_tensor_by_name(name) for name in INPUT_NAMES]
                output = graph.get_tensor_by_name(OUTPUT_NAME)
            except KeyError as exc:
                die(f"{family}: expected tensor missing from MetaGraph: {exc}")

        shapes = [tensor_shape(t, n) for t, n in zip(inputs, INPUT_NAMES)]
        output_shape = tensor_shape(output, OUTPUT_NAME)
        if any(shape[1:] != [256, 256, 3] for shape in shapes):
            die(f"{family}: unexpected input shapes {shapes}; expected [batch,256,256,3]")
        if output_shape[1:] != [256, 256, 3]:
            die(f"{family}: unexpected output shape {output_shape}")

        batch = shapes[0][0]
        if batch <= 0 or any(shape[0] != batch for shape in shapes) or output_shape[0] != batch:
            die(f"{family}: incompatible fixed batch dimensions: inputs={shapes}, output={output_shape}")

        # Deterministic smoke fixture: normalized RGB, a central masked patch,
        # and three-channel preserve/reconstruct mask following upstream semantics.
        rng = np.random.RandomState(20261014)
        y = rng.uniform(-1.0, 1.0, size=(1, 256, 256, 3)).astype(np.float32)
        x = y.copy()
        mask = np.ones((1, 256, 256, 3), dtype=np.float32)
        mask[:, 80:176, 88:168, :] = 0.0
        x[:, 80:176, 88:168, :] = 0.0
        x = np.repeat(x, batch, axis=0)
        y = np.repeat(y, batch, axis=0)
        mask = np.repeat(mask, batch, axis=0)

        with tf.Session(graph=graph, config=tf.ConfigProto(device_count={"GPU": 0})) as session:
            try:
                saver.restore(session, str(prefix))
                result = session.run(output, feed_dict={
                    INPUT_NAMES[0]: x,
                    INPUT_NAMES[1]: y,
                    INPUT_NAMES[2]: mask,
                })
            except Exception as exc:
                die(f"{family}: checkpoint restore or inference failed: {exc}")

        if tuple(result.shape) != tuple(output_shape):
            die(f"{family}: runtime returned {result.shape}, graph declares {output_shape}")
        if not np.isfinite(result).all():
            die(f"{family}: output contains NaN or Inf")

        print(f"PASS {family}: restored checkpoint and ran {OUTPUT_NAME}")
        print(f"  input shapes: {shapes}")
        print(f"  output shape: {list(result.shape)}")
        print(f"  output range: [{float(result.min()):.6f}, {float(result.max()):.6f}]")
        print(f"  output mean/std: {float(result.mean()):.6f} / {float(result.std()):.6f}")

        if args.output_dir:
            np.savez_compressed(
                args.output_dir / f"deepcreampy_{family}_reference.npz",
                x=x[:1], y=y[:1], mask=mask[:1], output=result[:1],
            )
            print(f"  parity fixture: {args.output_dir / f'deepcreampy_{family}_reference.npz'}")

    print("\nPASS: requested checkpoint family/families completed TensorFlow reference inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
