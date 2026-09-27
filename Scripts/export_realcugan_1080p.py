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
out_path = Path(os.environ.get("REALCUGAN_OUT", "Models/RealCUGAN2xNoise3_1080p.mlpackage"))
out_path.parent.mkdir(parents=True, exist_ok=True)
H, W = 1080, 1920

upsampler = RealWaifuUpScaler(scale=2, weight_path=str(weight_path), half=False, device="cpu")
pro = upsampler.pro

class FullFrame2x(torch.nn.Module):
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

model = FullFrame2x(upsampler.model).eval()
# The traced graph is fully convolutional with constant padding/crops. Trace on
# a compact tensor to avoid doing a full 1080p CPU inference during CI; Core ML
# receives the real fixed 1920x1080 deployment shape below.
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
mlmodel.author = "bilibili/ailab Real-CUGAN; iOS conversion for RIFE60GhostGuard"
mlmodel.short_description = "Real-CUGAN Anime 2x, Noise Level 3, full-frame 1920x1080 -> 3840x2160"
mlmodel.version = "1.0"
mlmodel.save(str(out_path))
print(f"Saved {out_path}")
