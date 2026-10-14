#!/usr/bin/env python3
"""Export the official DeepCreamPy v2 TensorFlow checkpoints to fixed-shape Core ML.

The source archive is the 2019 DCPv2 release. We intentionally convert the two
trained generators independently (bar and mosaic) so the app can select the
appropriate model without changing the existing LaMa watermark remover.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import coremltools as ct
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

INPUT_NAMES = ("Placeholder", "Placeholder_1", "Placeholder_2")
OUTPUT_NAME = "add"
SHAPE = (1, 256, 256, 3)


def require_checkpoint(directory: Path, prefix: str) -> Path:
    meta = directory / f"{prefix}.meta"
    index = directory / f"{prefix}.index"
    data = directory / f"{prefix}.data-00000-of-00001"
    missing = [str(p) for p in (meta, index, data) if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError("DeepCream checkpoint is incomplete: " + ", ".join(missing))
    return directory / prefix


def export_one(checkpoint: Path, output_dir: Path, model_name: str) -> None:
    frozen_path = output_dir / f"{model_name}.pb"
    package_path = output_dir / f"{model_name}.mlpackage"
    if package_path.exists():
        shutil.rmtree(package_path)
    if frozen_path.exists():
        frozen_path.unlink()

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(str(checkpoint) + ".meta", clear_devices=True)
    with tf.compat.v1.Session(graph=graph) as session:
        saver.restore(session, str(checkpoint))
        graph_def = graph.as_graph_def()
        available = {op.name for op in graph.get_operations()}
        missing = [name for name in (*INPUT_NAMES, OUTPUT_NAME) if name not in available]
        if missing:
            raise RuntimeError(
                f"{model_name}: expected graph nodes are missing: {missing}. "
                "Refusing to export an unverified graph."
            )
        for name in INPUT_NAMES:
            shape = graph.get_tensor_by_name(name + ":0").shape.as_list()
            if shape != list(SHAPE):
                raise RuntimeError(f"{model_name}: {name}:0 has shape {shape}, expected {list(SHAPE)}")
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, graph_def, [OUTPUT_NAME]
        )
        with tf.io.gfile.GFile(str(frozen_path), "wb") as handle:
            handle.write(frozen.SerializeToString())

        # Keep the TensorFlow reference output for a numerical parity check after
        # conversion. The graph's third input is a known-pixel mask: 1=preserve,
        # 0=inpaint. The second input is the uncorrupted source image.
        rng = np.random.default_rng(20261014)
        source = rng.random(SHAPE, dtype=np.float32) * 2.0 - 1.0
        mask = np.ones(SHAPE, dtype=np.float32)
        mask[:, 80:176, 80:176, :] = 0.0
        masked = source.copy()
        masked[mask == 0.0] = -1.0
        feed = {
            INPUT_NAMES[0] + ":0": masked,
            INPUT_NAMES[1] + ":0": source,
            INPUT_NAMES[2] + ":0": mask,
        }
        reference = session.run(OUTPUT_NAME + ":0", feed_dict=feed)

    inputs = [
        ct.TensorType(name=INPUT_NAMES[0], shape=SHAPE, dtype=np.float32),
        ct.TensorType(name=INPUT_NAMES[1], shape=SHAPE, dtype=np.float32),
        ct.TensorType(name=INPUT_NAMES[2], shape=SHAPE, dtype=np.float32),
    ]
    print(f"Converting {model_name} with coremltools {ct.__version__}", flush=True)
    converted = ct.convert(
        str(frozen_path),
        source="tensorflow",
        inputs=inputs,
        outputs=[ct.TensorType(name=OUTPUT_NAME)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    converted.short_description = (
        "DeepCreamPy v2 anime-image inpainting model. Fixed 256x256 RGB input; "
        "use only with a user-provided inpainting mask."
    )
    converted.input_description[INPUT_NAMES[0]] = "Masked RGB crop, float32 in [-1,1], NHWC."
    converted.input_description[INPUT_NAMES[1]] = "Original RGB crop, float32 in [-1,1], NHWC."
    converted.input_description[INPUT_NAMES[2]] = "Known-pixel mask, float32, 1=preserve and 0=inpaint."
    converted.output_description[OUTPUT_NAME] = "Inpainted RGB crop, float32 in [-1,1], NHWC."
    converted.save(str(package_path))

    # Run the exported package on macOS, not just serialize it. Reject conversion
    # if output names/shapes or numerical behavior differ unexpectedly.
    compiled = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    result = compiled.predict({
        INPUT_NAMES[0]: masked,
        INPUT_NAMES[1]: source,
        INPUT_NAMES[2]: mask,
    })
    if OUTPUT_NAME not in result:
        raise RuntimeError(f"{model_name}: Core ML output missing {OUTPUT_NAME!r}; got {list(result)}")
    actual = np.asarray(result[OUTPUT_NAME])
    if actual.shape != reference.shape:
        raise RuntimeError(f"{model_name}: output shape {actual.shape} != TensorFlow {reference.shape}")
    if not np.isfinite(actual).all():
        raise RuntimeError(f"{model_name}: Core ML output contains NaN/Inf")
    max_error = float(np.max(np.abs(actual.astype(np.float32) - reference.astype(np.float32))))
    mean_error = float(np.mean(np.abs(actual.astype(np.float32) - reference.astype(np.float32))))
    print(f"{model_name}: parity max_abs={max_error:.6f}, mean_abs={mean_error:.6f}", flush=True)
    if max_error > 0.12 or mean_error > 0.015:
        raise RuntimeError(
            f"{model_name}: Core ML numerical parity failed "
            f"(max_abs={max_error:.6f}, mean_abs={mean_error:.6f})"
        )
    frozen_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("Models"))
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"DeepCream checkpoint archive not found: {args.archive}")
    args.output.mkdir(parents=True, exist_ok=True)
    staging = args.output / "DeepCreamSource"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Checkpoint archive is corrupt at {bad}")
        archive.extractall(staging)
    root = staging / "09-11-2019 DCPv2 model"
    if not root.is_dir():
        candidates = list(staging.rglob("Train_775000.meta"))
        if not candidates:
            raise RuntimeError("Archive does not contain the expected Train_775000.meta checkpoint")
        root = candidates[0].parent.parent
    bar = require_checkpoint(root / "bar", "Train_775000")
    mosaic = require_checkpoint(root / "mosaic", "Train_290000")
    export_one(bar, args.output, "DeepCreamBar")
    export_one(mosaic, args.output, "DeepCreamMosaic")
    shutil.rmtree(staging)
    print("DeepCream Core ML export and parity checks passed.", flush=True)


if __name__ == "__main__":
    main()
