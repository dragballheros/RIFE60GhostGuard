# AI Image Studio integration

This feature adds a SwiftUI image-generation interface to RIFE60GhostGuard. The iPhone is the control surface; generation and LoRA training require GPU services. Model weights are intentionally not bundled in the IPA.

## Image generation endpoint

Use a RunPod Serverless endpoint running the official RunPod ComfyUI worker or a compatible worker:

- https://github.com/runpod-workers/worker-comfyui
- RunPod deployment guide: https://www.runpod.io/blog/deploy-comfyui-as-a-serverless-api-endpoint

The app calls the endpoint's `/run` and `/status/{job_id}` routes and sends a ComfyUI API-format workflow in `input.workflow`. The worker must return image output as `output.images[].data` (base64 PNG) or `output.images[].url`.

The app now includes two distinct profile-specific workflows. Install all model files in the corresponding ComfyUI model folders:

- `models/diffusion_models/screenChantvMerge_v20.safetensors`
- `models/text_encoders/qwen_3_06b_base.safetensors`
- `models/vae/qwen_image_vae.safetensors`
- `models/loras/Turbo-ANIMA-v2.9.safetensors`
- Optional character LoRA: `models/loras/Ichinose_Chizuru.safetensors`

For **WAI Illustrious v1.3**, install:

- `models/checkpoints/waiNSFWIllustrious_v130.safetensors` (SDXL checkpoint)
- `models/vae/sdxl.vae.safetensors`
- `models/loras/のなかゆき.safetensors` (the second PNG identifies this LoRA at weight 0.8; hash `dffb5926186c` is not independently verified)
- `models/upscale_models/RealESRGAN_x4plus_anime_6B.pth

These names are configurable in the app. The checkpoint and the exact character LoRA require verification before their compatibility can be guaranteed. The character LoRA hash from the source PNG is `160fca5c6aae`, and remains unresolved. If the worker reports a missing model, install the correct file or disable the optional character LoRA. The ANIMA workflow uses `UNETLoader`, `CLIPLoader`, `VAELoader`, `LoraLoader` and `ModelSamplingAuraFlow`. WAI Illustrious uses the SDXL `CheckpointLoaderSimple` path, optional `CLIPSetLastLayer` for CLIP skip, and an external VAE and LoRA. Both profiles can run a high-resolution second pass using `UpscaleModelLoader`, `ImageUpscaleWithModel`, `ImageScale`, `VAEEncode`, `KSampler`, and `VAEDecode`. All required nodes and model files must be installed on the remote worker.

The Turbo-ANIMA profile recovers the first PNG's settings. The **WAI Illustrious v1.3** profile uses the second PNG's settings: checkpoint `waiNSFWIllustrious_v130`, LoRA `のなかゆき` at 0.8, Euler a, Automatic schedule (mapped to ComfyUI's `normal` scheduler), 30 steps, CFG 7, seed 624067427, base 896×1344, CLIP skip 2, ESRGAN Anime6B 2× high-resolution pass, 20 hires steps, and 0.5 denoise. ENSD and token-merging values are displayed for reference but are not applied by stock ComfyUI without compatible custom nodes. The Turbo-ANIMA profile recovers the supplied PNG's settings: positive/negative prompts, Euler a (ComfyUI `euler_ancestral`), Normal scheduler, 8 steps, CFG 1, shift 3, seed 1647498191, 848×1200 output, Turbo LoRA 1.0 and character LoRA 0.7. The stored PNG was 848×1200 while its generation metadata said 850×1200, so the UI defaults to the actual saved dimensions.

### Costs and credentials

Create your own RunPod endpoint and API key. Add the endpoint ID and key in AI Image Studio > Settings. Credentials are stored in iOS Keychain, not in UserDefaults or source code. Serverless GPU execution is billable; configure scale-to-zero and usage limits on the provider side. GitHub Actions builds the IPA but does not provide GPU inference.

## Existing upscale path

After a PNG is returned, **Upscale with Real-CUGAN + Sharpie** passes the generated image URL to the existing `VideoProcessorViewModel.handleImport` and `start()` image pipeline. This uses the app's currently bundled Core ML Real-CUGAN model and existing final outline/colour pipeline. At native 2× mode, an 848×1200 generated image becomes 1696×2400. It does not claim to output 4K from an 848×1200 source.

## ANIMA LoRA trainer API contract

The app's LoRA Trainer tab targets a separate, authenticated GPU service. The stock RunPod ComfyUI worker does not implement training routes. Configure a base URL for a service that implements this contract:

- `POST /api/anima/lora/train`: multipart fields `caption`, `trigger_word`, `rank`, `epochs`, `learning_rate`, `base_model=anima`, and repeated `images` files. Return JSON `{"job_id":"..."}`.
- `GET /api/anima/lora/train/{job_id}`: return `{"status":"queued|running|completed|failed|cancelled","progress":0.0,"lora_url":"https://...","error":"..."}`. The completed response must include `lora_url` or `download_url`.

The trainer should use the ANIMA-specific training entrypoint, not an SDXL/Pony training script: https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md. It requires ANIMA DiT, Qwen3 text encoder, Qwen Image VAE, a dataset configuration and a CUDA GPU. Only run training on a trusted endpoint, use images you are authorized to train on, and remove uploaded datasets when no longer needed.

This repository does not provision a paid GPU endpoint or trainer automatically, and this guide does not claim that either endpoint has been deployed. The app and request client are ready to connect once the GPU services and compatible model files are provisioned.
