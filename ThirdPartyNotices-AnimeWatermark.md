# Anime watermark inpainting model attribution

This app includes a Core ML conversion of Anime/Manga Big LaMa.

- Fine-tune author/model card: dreMaz, AnimeMangaInpainting — https://huggingface.co/dreMaz/AnimeMangaInpainting
- The model card lists the fine-tune license as MIT, and states that the checkpoint was fine-tuned on 300,000 anime/manga-style samples.
- Distributed TorchScript checkpoint: https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting
- Original checkpoint MD5: 29f284f36a0a510bcacf39ecf4c4d54f
- Base architecture: LaMa, https://github.com/advimman/lama — Apache License 2.0, included in LaMa-Apache-2.0.txt.
- Authors: Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, Victor Lempitsky.
- Paper: Resolution-robust Large Mask Inpainting with Fourier Convolutions, WACV 2022, https://arxiv.org/abs/2109.07161
- Conversion reference: https://github.com/mallman/CoreMLaMa (Apache-2.0).

RIFE60GhostGuard packaging uses fixed 512×512 RGB/mask inputs, FP32 Core ML weights/calculations and an iOS 16 deployment target. The app selects CPU+GPU inference. The original fine-tuned weights are used, not retrained or relabeled as a new model. PyTorch, Python and coremltools are build-time dependencies only and are not included in the app.

The learned fill reconstructs a plausible background from surrounding pixels; it cannot recover the exact original pixels hidden by a watermark. It is an image-inpainting model, not a temporally trained video model. Fixed masks are reused per source frame before RIFE interpolation.
