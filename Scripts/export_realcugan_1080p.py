import os
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

# The workflow clones a CoreML-compatible Real-CUGAN implementation here.
realcugan_dir = Path(os.environ.get("REALCUGAN_DIR", "/tmp/video_upscale/realcugan/Real-CUGAN"))
sys.path.insert(0, str(realcugan_dir))
from upcunet_v3 import RealWaifuUpScaler  # noqa: E402

weight_path = Path(os.environ["REALCUGAN_WEIGHT"])
out_path = Path(os.environ.get("REALCUGAN_OUT", "Models/RealCUGAN2xNoise3_1080p.mlpackage"))
out_path.parent.mkdir(parents=True, exist_ok=True)

# This model is deliberately fixed to 1920x1080. Real-CUGAN's SE blocks use
# global spatial means; full-frame inference preserves the exact standard-model
# behavior and avoids the quality/seam compromises of naive independent tiles.
H, W = 1080, 1920

upsampler = RealWaifuUpScaler(
    scale=2,
    weight_path=str(weight_path),
    half=False,
    device="cpu",
)
pro = upsampler.pro

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, image, alpha):
        # alpha stays dynamic so the iOS app can reproduce the waifu2x
        # intensity control without rebuilding the network.
        y = self.model(image, tile_mode=0, cache_mode=0, alpha=alpha[0], pro=pro)
        return y.float()

model = Wrapper(upsampler.model).eval()
image_example = torch.zeros(1, 3, H, W, dtype=torch.float32)
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
mlmodel.author = "bilibili/ailab Real-CUGAN; iOS conversion for RIFE60GhostGuard"
mlmodel.short_description = "Real-CUGAN Anime 2x, Noise Level 3, full-frame 1920x1080 -> 3840x2160"
mlmodel.version = "1.0"
mlmodel.save(str(out_path))
print(f"Saved {out_path}")
