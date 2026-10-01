"""Convert the verified anime/manga LaMa TorchScript checkpoint to iOS 16 Core ML.
No PyTorch or third-party runtime is shipped in the app. FP32 avoids known LaMa
fp16/Neural Engine numerical issues; inference uses CPU_AND_GPU in Swift.
"""
from pathlib import Path
import hashlib
import os
import json
import numpy as np
import torch
import coremltools as ct

WEIGHT_MD5 = '29f284f36a0a510bcacf39ecf4c4d54f'
SOURCE_URL = 'https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt'

class InpaintWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, image, mask):
        binary_mask = (mask > 0.5).to(torch.float32)
        prediction = self.model(image, binary_mask)
        # Composite here too: unchanged model pixels outside the binary mask.
        return torch.clamp((prediction * binary_mask + image * (1 - binary_mask)) * 255.0, 0, 255)

def main():
    torch.set_num_threads(4)
    weight = Path(os.environ.get('ANIME_INPAINT_WEIGHT', '/tmp/anime-manga-big-lama.pt'))
    digest = hashlib.md5(weight.read_bytes()).hexdigest()
    if digest != WEIGHT_MD5:
        raise RuntimeError(f'Anime model checksum mismatch: {digest}')
    model = InpaintWrapper(torch.jit.load(str(weight), map_location='cpu').eval()).eval()
    size = 512
    image = torch.rand(1, 3, size, size)
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, 220:280, 200:320] = 1
    with torch.inference_mode():
        trace = torch.jit.trace(model, (image, mask), strict=False)
    converted = ct.convert(
        trace,
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        inputs=[ct.ImageType(name='image', shape=(1, 3, size, size), scale=1/255.0),
                ct.ImageType(name='mask', shape=(1, 1, size, size), color_layout=ct.colorlayout.GRAYSCALE, scale=1/255.0)],
        outputs=[ct.ImageType(name='output')],
        skip_model_load=True,
    )
    converted.author = 'dreMaz (anime/manga fine-tune); LaMa authors; Core ML packaging by RIFE60GhostGuard'
    converted.short_description = 'Anime/Manga Big LaMa masked inpainting, 512x512, FP32, iOS 16'
    converted.user_defined_metadata['source_url'] = SOURCE_URL
    converted.user_defined_metadata['source_md5'] = WEIGHT_MD5
    converted.user_defined_metadata['model_family'] = 'anime-manga-big-lama'
    out = Path(os.environ.get('ANIME_INPAINT_OUT', 'Models/AnimeWatermarkLaMa512.mlpackage'))
    out.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(out))
    spec = converted.get_spec()
    assert spec.specificationVersion <= 7, spec.specificationVersion
    print(json.dumps({'model': str(out), 'specificationVersion': spec.specificationVersion, 'md5': digest}))

if __name__ == '__main__':
    main()
