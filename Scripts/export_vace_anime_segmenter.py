#!/usr/bin/env python3
"""Convert SkyTNT anime-seg ISNet to a bundled Core ML model.

The weights and the anime-segmentation code are Apache-2.0
(https://huggingface.co/skytnt/anime-seg, owner confirmation that the
checkpoint is Apache-2.0). Conversion runs in CI. The phone only runs the
resulting mlpackage on the Neural Engine or GPU.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "https://github.com/SkyTNT/anime-segmentation.git"
CKPT_URL = (
    "https://huggingface.co/skytnt/anime-seg/resolve/"
    "976cbb9ba61600e0993d1ee2a8a247ae05d183ee/isnetis.ckpt"
)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Models" / "VACEAnimeSeg.mlpackage"


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 100_000_000:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def main() -> None:
    work = Path("/tmp/anime-segmentation")
    if not (work / "train.py").is_file():
        if work.exists():
            shutil.rmtree(work)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(work)])
    checkpoint = Path("/tmp/isnetis.ckpt")
    download(CKPT_URL, checkpoint)
    if checkpoint.stat().st_size < 100_000_000:
        raise SystemExit("anime-seg checkpoint is unexpectedly small")

    import torch
    import coremltools as ct

    original_load = torch.load

    def load_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_checkpoint
    sys.path.insert(0, str(work))
    from train import AnimeSegmentation

    network = AnimeSegmentation.try_load("isnet_is", str(checkpoint), "cpu", img_size=1024)
    network.eval()
    example = torch.zeros(1, 3, 1024, 1024)
    with torch.no_grad():
        traced = torch.jit.trace(network, example)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="img", shape=(1, 3, 1024, 1024))],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(OUTPUT))
    manifest = OUTPUT / "Manifest.json"
    if not manifest.is_file():
        raise SystemExit("Core ML package is missing Manifest.json")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
