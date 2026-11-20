#!/usr/bin/env python3
"""Convert the iOS Core ML packages to ONNX for the Windows build.

The packages are not in this installer. On a Mac with the app's model files:

    python convert_coreml_to_onnx.py AnimeSharpieOutlineSubtle.mlpackage models/onnx

coremltools writes an ONNX file next to the package name. Copy the results to
%LOCALAPPDATA%\\RIFE60GhostGuard\\models\\onnx (or models\\onnx beside rife-pc.exe).

Expected file names:
    APISR2xRRDB.onnx                 input [1,3,256,256] output [1,3,512,512]
    SPAN2xHFA2k.onnx                 same shapes, named input / output
    AnimeSharpieOutlineSubtle.onnx   input x [1,3,256,256] output var_244
    AnimeWatermarkLaMa512.onnx       inputs image, mask at 512 output output
    DeepCreamBar.onnx                Placeholder, Placeholder_1, Placeholder_2
    DeepCreamMosaic.onnx             same three inputs, output add, batch 8

DeepCreamPy is CC BY-NC-SA 4.0 (non-commercial). LaMa's fine-tune is MIT and
the base model is Apache-2.0. Do not redistribute those weights inside the
setup program.
"""

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: convert_coreml_to_onnx.py MODEL.mlpackage OUTPUT_DIR")
        return 2
    source = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import coremltools as ct
    except ImportError:
        print("coremltools is not installed. Run this on macOS: pip install coremltools")
        return 1
    model = ct.models.MLModel(str(source))
    spec = model.get_spec()
    try:
        import coremltools.converters.onnx as onnx_converter  # type: ignore
    except Exception:
        onnx_converter = None
    dest = out_dir / (source.stem.replace(".mlpackage", "") + ".onnx")
    if dest.suffix != ".onnx":
        dest = dest.with_suffix(".onnx")
    try:
        onnx_model = ct.converters.onnx.convert(spec) if hasattr(ct, "converters") else None
    except Exception as exc:
        print(f"coremltools could not convert {source.name}: {exc}")
        print("Export the same network from the original PyTorch checkpoint to ONNX")
        print("and keep the input and output names listed in this script's docstring.")
        return 1
    if onnx_model is None:
        print("This coremltools build has no ONNX converter.")
        return 1
    if hasattr(onnx_model, "save"):
        onnx_model.save(str(dest))
    else:
        dest.write_bytes(bytes(onnx_model))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
