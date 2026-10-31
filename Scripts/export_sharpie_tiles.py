#!/usr/bin/env python3
"""Specialize the fully convolutional Sharpie model at larger tile sizes.

The shipped network is SRVGG: convolutions and a pixel shuffle, no dense
layers. The same weights accept a larger NCHW input. Interior pixels match
the 256 tiling apart from the 20 px tile border. Benchmark mode chooses
among the bundled sizes.

An mlProgram cannot be reconstructed from the spec alone. The weight blob
has to travel with the clone, and the MIL function input plus every rank-4
spatial dimension that is still the source tile size has to move with the
feature description. Channel sizes (3 and 32) stay put.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import coremltools as ct

SOURCE_SPATIAL = 256


def reshape_feature(spec, height: int, width: int) -> None:
    for feature in list(spec.description.input) + list(spec.description.output):
        if feature.type.WhichOneof("Type") != "multiArrayType":
            continue
        shape = feature.type.multiArrayType.shape
        if len(shape) != 4:
            continue
        shape[2] = height
        shape[3] = width


def reshape_tensor(tensor_type, height: int, width: int) -> None:
    dimensions = tensor_type.dimensions
    if len(dimensions) != 4:
        return
    for axis, size in ((2, height), (3, width)):
        dimension = dimensions[axis]
        if dimension.HasField("constant") and dimension.constant.size == SOURCE_SPATIAL:
            dimension.constant.size = size


def reshape_program(spec, height: int, width: int) -> None:
    program = spec.mlProgram
    for function in program.functions.values():
        for value in function.inputs:
            if value.type.WhichOneof("type") == "tensorType":
                reshape_tensor(value.type.tensorType, height, width)
        for block in function.block_specializations.values():
            reshape_block(block, height, width)


def reshape_block(block, height: int, width: int) -> None:
    for value in list(block.inputs):
        if value.type.WhichOneof("type") == "tensorType":
            reshape_tensor(value.type.tensorType, height, width)
    for operation in block.operations:
        for output in operation.outputs:
            if output.type.WhichOneof("type") == "tensorType":
                reshape_tensor(output.type.tensorType, height, width)
        for nested in operation.blocks:
            reshape_block(nested, height, width)


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "Models/AnimeSharpieOutlineSubtle.mlpackage")
    model = ct.models.MLModel(str(source))
    spec = model.get_spec()
    weights_dir = model.weights_dir
    if not weights_dir:
        raise SystemExit(f"{source} has no weights directory")
    out_dir = source.parent
    for size in (768, 1024):
        cloned = spec.__class__()
        cloned.ParseFromString(spec.SerializeToString())
        reshape_feature(cloned, size, size)
        reshape_program(cloned, size, size)
        destination = out_dir / f"AnimeSharpieOutlineSubtle_Tile{size}.mlpackage"
        if destination.exists():
            shutil.rmtree(destination)
        specialized = ct.models.MLModel(cloned, weights_dir=weights_dir)
        specialized.save(str(destination))
        weight = next(destination.rglob("weight.bin"), None)
        if weight is None or weight.stat().st_size == 0:
            raise SystemExit(f"{destination} was saved without weight.bin")
        saved = ct.models.MLModel(str(destination)).get_spec()
        shape = list(saved.description.input[0].type.multiArrayType.shape)
        if shape != [1, 3, size, size]:
            raise SystemExit(f"{destination} input shape is {shape}")
        print(f"Saved {destination} input={shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
