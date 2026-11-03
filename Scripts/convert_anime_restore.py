#!/usr/bin/env python3
"""Convert Anime Restore weights to fixed-tile fp16 Core ML packages.

APISR's pixel_unshuffle(2) is folded into conv_first as a 6x6 stride-2
convolution so Core ML never emits the rank-6 reshape the Neural Engine rejects.
SPAN is converted from the published ONNX graph. Neither package is copied
into the app bundle; the app downloads the zip on demand.
"""
import os
import sys
import zipfile

import numpy as np
import torch
from torch import nn
import coremltools as ct


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
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h, w)


class RRDBReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(12, 64, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB() for _ in range(6)])
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(pixel_unshuffle(x, 2))
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class RRDBFused(nn.Module):
    """Same network with pixel_unshuffle absorbed into conv_first."""

    def __init__(self, reference: RRDBReference):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 6, stride=2, padding=2, bias=True)
        self.body = reference.body
        self.conv_body = reference.conv_body
        self.conv_up1 = reference.conv_up1
        self.conv_up2 = reference.conv_up2
        self.conv_hr = reference.conv_hr
        self.conv_last = reference.conv_last
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self._fuse(reference.conv_first)

    def _fuse(self, conv):
        old_w = conv.weight.detach()  # 64, 12, 3, 3
        old_b = conv.bias.detach()
        new_w = torch.zeros(64, 3, 6, 6)
        for oc in range(64):
            for c in range(3):
                for ky in range(6):
                    for kx in range(6):
                        uy, ux = ky // 2, kx // 2
                        sy, sx = ky % 2, kx % 2
                        unshuffle_c = c * 4 + sy * 2 + sx
                        new_w[oc, c, ky, kx] = old_w[oc, unshuffle_c, uy, ux]
        with torch.no_grad():
            self.conv_first.weight.copy_(new_w)
            self.conv_first.bias.copy_(old_b)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_rrdb(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if "params_ema" in checkpoint:
        weight = checkpoint["params_ema"]
    elif "params" in checkpoint:
        weight = checkpoint["params"]
    elif "model_state_dict" in checkpoint:
        weight = checkpoint["model_state_dict"]
    else:
        raise SystemExit("unsupported APISR checkpoint keys: " + ",".join(checkpoint.keys()))
    cleaned = {}
    for key, value in weight.items():
        cleaned[key[10:] if key.startswith("_orig_mod.") else key] = value
    model = RRDBReference()
    model.load_state_dict(cleaned)
    model.eval()
    return model


def convert_torch(model, output_dir, name):
    example = torch.rand(1, 3, 256, 256)
    traced = torch.jit.trace(model, example)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 256, 256))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    ml.short_description = name
    ml.save(output_dir)
    spec = ml.get_spec()
    blob = str(spec)
    if "pixel_unshuffle" in blob or "PixelUnshuffle" in blob:
        raise SystemExit(name + " still contains pixel_unshuffle")


def zip_package(package_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for root, _, files in os.walk(package_dir):
            for name in files:
                full = os.path.join(root, name)
                archive.write(full, os.path.relpath(full, os.path.dirname(package_dir)))


class UnitRange(nn.Module):
    def __init__(self, net, scale_output):
        super().__init__()
        self.net = net
        self.scale_output = scale_output

    def forward(self, x):
        y = self.net(x)
        if self.scale_output != 1:
            y = y / self.scale_output
        return y.clamp(0, 1)


def load_span(path):
    import importlib.util
    import types
    fake_util = types.ModuleType("spandrel.util")
    fake_util.store_hyperparameters = lambda: (lambda cls: cls)
    sys.modules.setdefault("spandrel", types.ModuleType("spandrel"))
    sys.modules["spandrel.util"] = fake_util
    arch = os.path.expanduser("~/.local/lib/python3.12/site-packages/spandrel/architectures/SPAN/__arch/span.py")
    spec = importlib.util.spec_from_file_location("spanarch", arch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    weight = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model = module.SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=2, bias=True, norm=True)
    model.load_state_dict(weight)
    model.eval()
    with torch.no_grad():
        probe = model(torch.full((1, 3, 32, 32), 0.5))
    scale = 255.0 if probe.detach().abs().mean().item() > 2 else 1.0
    print(f"SPAN probe mean {probe.mean().item():.4f} output scale {scale}")
    return UnitRange(model, scale).eval()


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = os.path.join(root, "Models", "AnimeRestore")
    os.makedirs(out, exist_ok=True)
    weights = sys.argv[1]
    span = sys.argv[2]
    reference = load_rrdb(weights)
    fused = UnitRange(RRDBFused(reference), 1).eval()
    with torch.no_grad():
        sample = torch.rand(1, 3, 64, 64)
        a = reference(sample)
        b = fused(sample)
        err = (a.clamp(0, 1) - b).abs().max().item()
    print(f"APISR fuse max abs error {err:.8f} output {tuple(a.shape)}")
    if err > 1e-4:
        raise SystemExit("fused conv_first does not match pixel_unshuffle")
    apisr_dir = os.path.join(out, "APISR2xRRDB.mlpackage")
    convert_torch(fused, apisr_dir, "APISR 2x RRDB-6B tile 256 fp16")
    span_model = load_span(span)
    span_dir = os.path.join(out, "SPAN2xHFA2k.mlpackage")
    convert_torch(span_model, span_dir, "2xHFA2k LUDVAE SPAN tile 256 fp16")
    zip_package(apisr_dir, os.path.join(out, "APISR2xRRDB.mlpackage.zip"))
    zip_package(span_dir, os.path.join(out, "SPAN2xHFA2k.mlpackage.zip"))
    print("ANIME_RESTORE_CONVERT_OK")


if __name__ == "__main__":
    main()
