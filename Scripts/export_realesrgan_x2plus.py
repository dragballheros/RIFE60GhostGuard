#!/usr/bin/env python3
"""Export the official RealESRGAN_x2plus checkpoint as an iOS 16 Core ML model.

The graph is the upstream RRDBNet x2plus architecture. Conversion happens on the
GitHub macOS build runner; the iPhone runs only the compiled Core ML model.
"""
from pathlib import Path
import hashlib
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

WEIGHTS = Path(os.environ.get("REALESRGAN_WEIGHT", "/tmp/RealESRGAN_x2plus.pth"))
OUTPUT = Path(os.environ.get("REALESRGAN_OUT", "Models/RealESRGAN_x2plus_Tile384.mlpackage"))
SIZE = int(os.environ.get("REALESRGAN_TILE_SIZE", "384"))
URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

def download_weights():
    if WEIGHTS.is_file() and WEIGHTS.stat().st_size > 1_000_000:
        return
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    temporary = WEIGHTS.with_suffix(".download")
    request = urllib.request.Request(URL, headers={"User-Agent": "RIFE60GhostGuard-CoreML-export"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    if temporary.stat().st_size < 10_000_000:
        raise SystemExit("RealESRGAN_x2plus checkpoint is unexpectedly small.")
    temporary.replace(WEIGHTS)
    print(f"Downloaded official RealESRGAN_x2plus checkpoint: {WEIGHTS.stat().st_size} bytes")

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RealESRGANX2Plus(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(12, 64, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB() for _ in range(23)])
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(F.pixel_unshuffle(x, 2))
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

def main():
    if SIZE < 128 or SIZE % 2:
        raise SystemExit(f"Tile size must be an even integer >=128, received {SIZE}")
    download_weights()
    checkpoint = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model = RealESRGANX2Plus().eval()
    model.load_state_dict(state, strict=True)
    print(f"Loaded official x2plus state dict with {len(state)} tensors")
    sample = torch.linspace(0, 1, 3 * SIZE * SIZE, dtype=torch.float32).reshape(1, 3, SIZE, SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, strict=True)
        reference = model(sample)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=(1, 3, SIZE, SIZE), dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    converted.author = "xinntao/Real-ESRGAN"
    converted.short_description = f"RealESRGAN_x2plus general-image 2x, fp16 compute, fixed {SIZE}x{SIZE} input tile"
    converted.version = "1.0"
    converted.save(str(OUTPUT))
    check = converted.predict({"input": sample.numpy()})
    actual = check["output"]
    expected_shape = (1, 3, SIZE * 2, SIZE * 2)
    if tuple(actual.shape) != expected_shape:
        raise SystemExit(f"Core ML output shape {actual.shape} != {expected_shape}")
    delta = np.abs(actual - reference.numpy())
    mean_delta = float(delta.mean())
    max_delta = float(delta.max())
    if not np.isfinite(mean_delta) or mean_delta > 0.025 or max_delta > 0.35:
        raise SystemExit(f"Core ML parity check outside tolerance: mean={mean_delta:.6f}, max={max_delta:.6f}")
    print(f"REALESRGAN_COREML_EXPORT_PASS: {OUTPUT} • shape={actual.shape} • meanDelta={mean_delta:.6f} • maxDelta={max_delta:.6f}")

if __name__ == "__main__":
    main()
