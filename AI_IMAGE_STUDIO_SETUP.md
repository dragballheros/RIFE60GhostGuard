# AI Image Studio: On-device GPU setup

AI Image Studio runs image generation and compatible LoRA training directly on the iPhone. It does not require RunPod, a remote ComfyUI endpoint, a trainer server, or a paid inference service.

## Local image generation

The app embeds the open-source Draw Things MediaGenerationKit engine and defaults to the supported catalog model Animagine XL v3.1 (8-bit), file identifier animagine_xl_v3.1_q6p_q8p.ckpt. First generation downloads the checkpoint and required files to the app's Application Support directory. Allow several GB of free storage and keep the app open during the initial download and model load. Once downloaded, image sampling runs on-device through the engine's Apple GPU/Metal stack.

The generated PNG stays in the app until you save it to Photos, share/export it, or send it to the existing Real-CUGAN + Sharpie local upscale pipeline. For iPhone 14 Pro Max, start around 832 × 1216 with high-resolution diffusion disabled. For a larger final image, use the existing local upscaler after generation rather than holding a 2× diffusion pass in unified memory.

The model field accepts a model ID supported by the embedded Draw Things catalog. Arbitrary ComfyUI .safetensors filenames are not automatically recognized as model IDs.

## Local LoRA training

The LoRA Trainer invokes the Draw Things local trainer in-process. Selected images and captions are written to a temporary app-local dataset, training runs locally, and the resulting LoRA checkpoint is stored in the same on-device Models folder. The app selects the trained LoRA for subsequent local generation.

Training defaults to low-memory settings, 512-pixel training resolution, and a bounded step count. This does not override iOS memory limits. Start with a small dataset and keep the app foregrounded with the phone cooler attached. iOS can still terminate training if model loading, GPU allocations, thermal pressure, or system memory demand exceeds the available budget. If it fails repeatedly, reduce dataset size, rank, and epochs, or use a smaller compatible base model.

## Compatibility limits

The supplied source PNGs used the screenChantvMerge_v20 ANIMA checkpoint and Turbo-ANIMA-v2.9 LoRA. Those files are not automatically compatible with the supported Draw Things inference pipeline. The local build intentionally does not pretend they have been loaded. It uses Animagine XL v3.1, a supported SDXL anime model, until a native ANIMA model-import or architecture path is implemented and validated.

The WAI Illustrious checkpoint and its LoRA are also not assumed importable merely because they are SDXL-format files. They need conversion or registration through the embedded engine's supported import path before use.

## Licensing

Draw Things community code is GPL-3.0 licensed. Keep the pinned source and its license available with this project. If distributing this combined app to others, comply with GPL-3.0 source and notice requirements for the combined work.

## Pinned engine

The app build fetches drawthingsai/draw-things-community at commit 4f288803ac898525012c0fe8d998c85bcb920b70. GitHub Actions patches its Swift package manifest only to expose the existing DrawThingsCLILib and DataModels targets as products for app integration. The vendored copy is generated during CI and is not bundled as model weights in the IPA.
