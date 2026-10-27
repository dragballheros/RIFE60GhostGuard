import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

realcugan_dir = Path(os.environ.get("REALCUGAN_DIR", "/tmp/video_upscale/realcugan/Real-CUGAN"))
sys.path.insert(0, str(realcugan_dir))
from upcunet_v3 import RealWaifuUpScaler  # noqa: E402

weight_path = Path(os.environ["REALCUGAN_WEIGHT"])
out_path = Path(os.environ.get("REALCUGAN_OUT", "Models/RealCUGAN2xNoise3_Tile512.mlpackage"))
out_path.parent.mkdir(parents=True, exist_ok=True)

# Fixed deployment shapes let Core ML optimize each model aggressively.  The
# app keeps the original 512x512 model as a low-memory fallback and also ships
# a 704x608 model optimized for 1920x1080.  With 32px overlap that second shape
# has a 640x544 useful core, so 1080p needs only 3x2 = 6 predictions instead of
# 5x3 = 15.  This changes tiling only; weights, Noise 3 and model math are the
# same, so image quality and the app's 1.30 intensity remain unchanged.
H = int(os.environ.get("REALCUGAN_TILE_HEIGHT", "512"))
W = int(os.environ.get("REALCUGAN_TILE_WIDTH", "512"))

upsampler = RealWaifuUpScaler(scale=2, weight_path=str(weight_path), half=False, device="cpu")
pro = upsampler.pro


class Tile2x(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.unet1 = base.unet1.eval()
        self.unet2 = base.unet2.eval()

    def forward(self, image, alpha):
        x = F.pad(image, (18, 18, 18, 18), "reflect")
        x = self.unet1(x)
        x0 = self.unet2(x, alpha[0])
        x = x[:, :, 20:-20, 20:-20]
        x = x0 + x
        if pro:
            x = (x - 0.15) * (255.0 / 0.7)
        else:
            x = x * 255.0
        return torch.round(x).clamp(0.0, 255.0)


model = Tile2x(upsampler.model).eval()
# Trace compactly; Core ML receives the fixed deployment shape below.
image_example = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
alpha_example = torch.tensor([1.0], dtype=torch.float32)
with torch.no_grad():
    traced = torch.jit.trace(model, (image_example, alpha_example), strict=False, check_trace=False)

mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[
        ct.ImageType(
            name="image",
            shape=(1, 3, H, W),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        ),
        ct.TensorType(name="alpha", shape=(1,), dtype=np.float16),
    ],
    outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16,
)
mlmodel.author = "bilibili/ailab Real-CUGAN; iOS tiled conversion for RIFE60GhostGuard"
mlmodel.short_description = f"Real-CUGAN Anime 2x Noise 3, {W}x{H} tile -> {W*2}x{H*2}; native-resolution 2x stitching in app"
mlmodel.version = "2.1"
mlmodel.save(str(out_path))
print(f"Saved {out_path} ({W}x{H})")

# Optional fixed-batch asset. One prediction consumes N tiles stacked on the
# batch dimension, so a single output backing is legal. The graph is the same
# Tile2x math applied per tile. Nothing is downloaded; this writes only when
# REALCUGAN_BATCH is 2 or 4.
batch = int(os.environ.get("REALCUGAN_BATCH", "0"))
if batch in (2, 4):
    class BatchedTiles(torch.nn.Module):
        def __init__(self, single, count):
            super().__init__()
            self.single = single
            self.count = count

        def forward(self, image, alpha):
            parts = [self.single(image[index:index + 1] / 255.0, alpha) for index in range(self.count)]
            return torch.cat(parts, dim=0)

    batched = BatchedTiles(model, batch).eval()
    batch_image = torch.randint(0, 256, (batch, 3, H, W), dtype=torch.float32)
    with torch.no_grad():
        traced_batch = torch.jit.trace(batched, (batch_image, alpha_example), strict=False, check_trace=False)
        stacked = batched(batch_image, alpha_example)
        singles = torch.cat([model(batch_image[index:index + 1] / 255.0, alpha_example) for index in range(batch)], dim=0)
        delta = (stacked - singles).abs().max().item()
        if delta > 1:
            raise SystemExit(f"fixed batch diverged from per-tile by {delta}")
    batch_model = ct.convert(
        traced_batch,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="image", shape=(batch, 3, H, W), dtype=np.float32),
            ct.TensorType(name="alpha", shape=(1,), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    batch_path = out_path.with_name(f"{out_path.stem}_Batch{batch}.mlpackage")
    batch_model.short_description = f"Real-CUGAN Anime 2x Noise 3, fixed batch {batch}, {W}x{H} tiles"
    batch_model.author = mlmodel.author
    batch_model.version = "2.1"
    batch_model.save(str(batch_path))
    print(f"Saved {batch_path} (batch {batch}, max delta vs per-tile {delta})")
