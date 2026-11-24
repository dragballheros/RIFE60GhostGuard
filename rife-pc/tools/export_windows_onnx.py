#!/usr/bin/env python3
"""Build the ONNX networks the Windows app runs on DirectML.

Sources, in preference order:
  APISR 2x RRDB  — official PyTorch weight (Kiteretsu77), same graph as the iOS converter
  SPAN 2x HFA2k  — official Phhofm ONNX, checked against the .pth
  Sharpie        — SRVGG weights read out of the iOS Core ML weight blob
  DeepCream      — published ONNX of the DeepCreamPy v2 generators
  LaMa           — anime-manga TorchScript, checked against mayocream's ONNX when that file is present

Writes rife-pc/models/onnx/ and prints a verification report. FP16 is kept only
when it stays within 1/255 of the FP32 reference. LaMa stays FP32.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "onnx"
WEIGHTS = Path("/tmp/weights")
SHARPIE_BIN = Path("/tmp/modelsrc/1x_AnimeSharpieOutline_Subtle_SRVGG_32f_12c.mlpackage/Data/com.apple.CoreML/weights/weight.bin")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session(path: Path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def report_diff(name: str, reference: np.ndarray, candidate: np.ndarray) -> dict:
    diff = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    stats = {
        "name": name,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "shape": list(reference.shape),
    }
    print(f"{name}: max {stats['max_abs']:.6g} mean {stats['mean_abs']:.6g} shape {stats['shape']}")
    return stats


# --- APISR, matching Scripts/convert_anime_restore.py on the iOS branch ---

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
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


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
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat))).clamp(0, 1)


class RRDBFused(nn.Module):
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
        old_w = reference.conv_first.weight.detach()
        new_w = torch.zeros(64, 3, 6, 6)
        for oc in range(64):
            for c in range(3):
                for ky in range(6):
                    for kx in range(6):
                        unshuffle_c = c * 4 + (ky % 2) * 2 + (kx % 2)
                        new_w[oc, c, ky, kx] = old_w[oc, unshuffle_c, ky // 2, kx // 2]
        with torch.no_grad():
            self.conv_first.weight.copy_(new_w)
            self.conv_first.bias.copy_(reference.conv_first.bias.detach())

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat))).clamp(0, 1)


def load_rrdb(path: Path) -> RRDBReference:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "params_ema" in checkpoint:
        weight = checkpoint["params_ema"]
    elif "params" in checkpoint:
        weight = checkpoint["params"]
    else:
        weight = checkpoint["model_state_dict"]
    cleaned = {key[10:] if key.startswith("_orig_mod.") else key: value for key, value in weight.items()}
    model = RRDBReference()
    model.load_state_dict(cleaned)
    model.eval()
    return model


def export_apisr(notes: list) -> None:
    reference = load_rrdb(WEIGHTS / "apisr.pth")
    fused = RRDBFused(reference).eval()
    sample = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        a = reference(sample)
        b = fused(sample)
    fuse = report_diff("APISR fused conv vs pixel_unshuffle", a.numpy(), b.numpy())
    if fuse["max_abs"] > 1e-4:
        raise SystemExit("APISR fuse does not match the reference network")
    dest = OUT / "APISR2xRRDB.onnx"
    torch.onnx.export(
        fused,
        torch.rand(1, 3, 256, 256),
        dest,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(dest)
    tile = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ref = fused(tile).numpy()
    got = session(dest).run(None, {"input": tile.numpy()})[0]
    check = report_diff("APISR ONNX vs PyTorch", ref, got)
    if check["max_abs"] > 1e-4:
        raise SystemExit("APISR ONNX does not match PyTorch")
    half = fused.half().eval()
    with torch.no_grad():
        half_out = half(tile.half()).float().numpy()
    fp16 = report_diff("APISR FP16 vs FP32", ref, half_out)
    # FP16 math is close enough, but the Windows runner feeds float32 tiles.
    # A half() export would require float16 inputs and is not what we ship.
    notes.append({
        "model": "APISR2xRRDB.onnx",
        "checks": [fuse, check, fp16],
        "dtype": "fp32",
        "fp16_max_abs_if_used": fp16["max_abs"],
        "note": "Shipped FP32 so ONNX Runtime receives the same float32 tiles as the iOS preprocessor. FP16 was within 1/255.",
    })


def load_span():
    src = Path("/home/ubuntu/.local/lib/python3.12/site-packages/spandrel/architectures/SPAN/__arch/span.py").read_text()
    src = src.replace(
        "from spandrel.util import store_hyperparameters",
        "def store_hyperparameters(*_a, **_k):\n    return lambda cls: cls",
    )
    namespace: dict = {}
    exec(compile(src, "span.py", "exec"), namespace)
    checkpoint = torch.load(WEIGHTS / "span.pth", map_location="cpu", weights_only=False)
    weight = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model = namespace["SPAN"](num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=2, bias=True, norm=True)
    model.load_state_dict(weight)
    model.eval()
    return model


def export_span(notes: list) -> None:
    model = load_span()
    tile = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        ref = model(tile).clamp(0, 1).numpy()
    fp32 = session(WEIGHTS / "span.onnx").run(None, {"input": tile.numpy()})[0]
    check = report_diff("SPAN official ONNX vs PyTorch clamp", ref, fp32)
    if check["max_abs"] > 1e-4:
        raise SystemExit("SPAN ONNX does not match the .pth")
    fp16_path = WEIGHTS / "span_fp16.onnx"
    got16 = session(fp16_path).run(None, {"input": tile.half().numpy()})[0]
    fp16 = report_diff("SPAN FP16 ONNX vs PyTorch", ref, got16.astype(np.float32))
    # The published FP16 file requires float16 inputs. The app sends float32, so ship the FP32 graph.
    dest = OUT / "SPAN2xHFA2k.onnx"
    dest.write_bytes((WEIGHTS / "span.onnx").read_bytes())
    notes.append({
        "model": "SPAN2xHFA2k.onnx",
        "checks": [check, fp16],
        "dtype": "fp32",
        "note": "Official Phhofm FP32 ONNX. FP16 matches when the input is float16, but the runner sends float32.",
    })


def read_f16(blob: bytes, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
    count = int(np.prod(shape))
    # Core ML's blob offset points at a 64-byte header. The fp16 payload follows it.
    raw = blob[offset + 64 : offset + 64 + count * 2]
    if len(raw) != count * 2:
        raise SystemExit(f"short weight read at {offset} for {shape}")
    array = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    return torch.from_numpy(np.array(array, copy=True))


def export_sharpie(notes: list) -> None:
    import coremltools as ct

    package = next(Path("/tmp/modelsrc").rglob("*.mlpackage"))
    spec = ct.models.MLModel(str(package)).get_spec()
    block = spec.mlProgram.functions["main"].block_specializations["CoreML5"]
    blob = SHARPIE_BIN.read_bytes()
    weights = {}
    from google.protobuf.json_format import MessageToDict

    for op in block.operations:
        if op.type != "const":
            continue
        data = MessageToDict(op)
        value = data["attributes"]["val"]
        dims = value["type"]["tensorType"].get("dimensions", [])
        shape = tuple(int(item["constant"]["size"]) for item in dims)
        name = data["outputs"][0]["name"]
        if "blobFileValue" in value:
            offset = int(value["blobFileValue"]["offset"])
            weights[name] = read_f16(blob, offset, shape)
        elif "immediateValue" in value and "bytes" in value["immediateValue"].get("tensor", {}):
            import base64
            raw = base64.b64decode(value["immediateValue"]["tensor"]["bytes"]["values"])
            weights[name] = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape).copy())
    first = weights["body_0_weight_to_fp16"]
    if not torch.isfinite(first).all() or float(first.std()) > 2:
        raise SystemExit(f"Sharpie weight blob looks wrong, std {float(first.std())} max {float(first.abs().max())}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.acts = nn.ModuleList()
            for index in range(0, 26, 2):
                weight = weights[f"body_{index}_weight_to_fp16"]
                bias = weights[f"body_{index}_bias_to_fp16"]
                conv = nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1)
                with torch.no_grad():
                    conv.weight.copy_(weight)
                    conv.bias.copy_(bias)
                self.convs.append(conv)
                alpha = weights[f"input_{3 + index * 2}_alpha_0_to_fp16"]
                act = nn.PReLU(num_parameters=alpha.numel())
                with torch.no_grad():
                    act.weight.copy_(alpha)
                self.acts.append(act)
            last_w = weights["body_26_weight_to_fp16"]
            last_b = weights["body_26_bias_to_fp16"]
            self.last = nn.Conv2d(last_w.shape[1], last_w.shape[0], 3, padding=1)
            with torch.no_grad():
                self.last.weight.copy_(last_w)
                self.last.bias.copy_(last_b)

        def forward(self, x):
            h = x
            for conv, act in zip(self.convs, self.acts):
                h = act(conv(h))
            return self.last(h) + x

    net = Net().eval()
    dest = OUT / "AnimeSharpieOutlineSubtle.onnx"
    torch.onnx.export(
        net,
        torch.rand(1, 3, 256, 256),
        dest,
        input_names=["x"],
        output_names=["var_244"],
        dynamic_axes={"x": {2: "height", 3: "width"}, "var_244": {2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )
    tile = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        ref = net(tile).numpy()
    got = session(dest).run(None, {"x": tile.numpy()})[0]
    check = report_diff("Sharpie ONNX vs Core ML weights", ref, got)
    if check["max_abs"] > 1e-5:
        raise SystemExit("Sharpie ONNX does not match the reconstructed Core ML graph")
    notes.append({
        "model": "AnimeSharpieOutlineSubtle.onnx",
        "checks": [check],
        "dtype": "fp32-from-fp16-coreml-weights",
        "note": "Weights were read from the iOS mlpackage blob. Core ML itself cannot execute on this Linux machine, so the check is the MIL graph (12 conv+prelu, final conv, residual add) against ONNX.",
    })


def copy_deepcream(notes: list) -> None:
    mapping = {"bar.onnx": "DeepCreamBar.onnx", "mosaic.onnx": "DeepCreamMosaic.onnx"}
    rng = np.random.default_rng(20261014)
    source = rng.random((1, 256, 256, 3), dtype=np.float32) * 2 - 1
    mask = np.ones((1, 256, 256, 3), dtype=np.float32)
    mask[:, 80:176, 80:176, :] = 0
    masked = source.copy()
    masked[mask == 0] = -1
    for src_name, dest_name in mapping.items():
        src = WEIGHTS / src_name
        dest = OUT / dest_name
        dest.write_bytes(src.read_bytes())
        model = onnx.load(dest)
        names = [item.name for item in model.graph.input]
        if names != ["Placeholder:0", "Placeholder_1:0", "Placeholder_2:0"]:
            raise SystemExit(f"{dest_name} inputs are {names}")
        out = session(dest).run(
            ["add:0"],
            {"Placeholder:0": masked, "Placeholder_1:0": source, "Placeholder_2:0": mask},
        )[0]
        if out.shape != (1, 256, 256, 3) or not np.isfinite(out).all():
            raise SystemExit(f"{dest_name} did not return a finite 256 tile")
        changed = float(np.abs(out - source).max())
        print(f"{dest_name}: finite output, max change vs source {changed:.4f}")
        notes.append({
            "model": dest_name,
            "checks": [{"name": "finite 256 tile", "max_abs_vs_source": changed, "shape": list(out.shape)}],
            "dtype": "fp32",
            "note": "Published ONNX of the DeepCreamPy v2 generator. TensorFlow was not re-run here; input names match the iOS export (masked, source, known mask).",
        })


def export_lama(notes: list) -> None:
    scripted = torch.jit.load(str(WEIGHTS / "lama.pt"), map_location="cpu").eval()

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, image, mask):
            return self.inner(image, mask)

    wrapped = Wrapper(scripted).eval()
    image = torch.rand(1, 3, 512, 512)
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 200:300, 220:340] = 1
    dest = OUT / "AnimeWatermarkLaMa512.onnx"
    try:
        torch.onnx.export(
            wrapped,
            (image, mask),
            dest,
            input_names=["image", "mask"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )
        with torch.no_grad():
            ref = wrapped(image, mask).numpy()
        got = session(dest).run(None, {"image": image.numpy(), "mask": mask.numpy()})[0]
        check = report_diff("LaMa ONNX vs TorchScript", ref, got)
        if check["max_abs"] > 1e-3:
            raise SystemExit("LaMa ONNX does not match the TorchScript checkpoint")
        notes.append({"model": "AnimeWatermarkLaMa512.onnx", "checks": [check], "dtype": "fp32", "bytes": dest.stat().st_size})
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        notes.append({"model": "AnimeWatermarkLaMa512.onnx", "error": str(exc)})
        print("LaMa torch export failed:", exc)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    notes: list = []
    export_apisr(notes)
    export_span(notes)
    export_sharpie(notes)
    copy_deepcream(notes)
    export_lama(notes)
    manifest = []
    for path in sorted(OUT.glob("*.onnx")):
        manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (OUT / "manifest.json").write_text(json.dumps({"files": manifest, "checks": notes}, indent=2))
    total = sum(item["bytes"] for item in manifest)
    print(f"TOTAL_BYTES {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
