#!/usr/bin/env python3
"""Specialize the fully convolutional Sharpie model at larger tile sizes.

The shipped network is SRVGG: convolutions and a pixel shuffle, no dense
layers. The same weights accept a larger NCHW input. Interior pixels match
the 256 tiling apart from the 20 px tile border. Benchmark mode chooses
among the bundled sizes.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import coremltools as ct


def reshape(spec, height: int, width: int) -> None:
    for feature in list(spec.description.input) + list(spec.description.output):
        kind = feature.type.WhichOneof("Type")
        if kind != "multiArrayType":
            continue
        shape = feature.type.multiArrayType.shape
        if len(shape) != 4:
            continue
        shape[2] = height
        shape[3] = width


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "Models/AnimeSharpieOutlineSubtle.mlpackage")
    model = ct.models.MLModel(str(source))
    spec = model.get_spec()
    out_dir = source.parent
    for size in (768, 1024):
        cloned = spec.__class__()
        cloned.ParseFromString(spec.SerializeToString())
        reshape(cloned, size, size)
        destination = out_dir / f"AnimeSharpieOutlineSubtle_Tile{size}.mlpackage"
        if destination.exists():
            shutil.rmtree(destination)
        specialized = ct.models.MLModel(cloned)
        specialized.save(str(destination))
        print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
