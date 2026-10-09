# AI Image Studio integration

This feature adds a SwiftUI image-generation interface to RIFE60GhostGuard. The iPhone is the control surface; generation and LoRA training require GPU services. Model weights are intentionally not bundled in the IPA.

## Image generation endpoint

Use a RunPod Serverless endpoint running the official RunPod ComfyUI worker or a compatible worker:

- https://github.com/runpod-workers/worker-comfyui
- RunPod deployment guide: https://www.runpod.io/blog/deploy-comfyui-as-a-serverless-api-endpoint

A model-install script is included at `cloud/comfyui-models/install-runpod-models.sh`. **Important mount-path difference:** a RunPod Pod typically mounts its Network Volume or Global Volume at `/workspace`, while RunPod Serverless workers see the same volume at `/runpod-volume`. Run the installer from the Pod and target `/workspace/models` so model files persist on the attached volume:

```bash
curl -fsSLO https://raw.githubusercontent.com/dragballheros/RIFE60GhostGuard/main/cloud/comfyui-models/install-runpod-models.sh
bash install-runpod-models.sh all /workspace/models
# Or: bash install-runpod-models.sh anima /workspace/models
# Or: bash install-runpod-models.sh wai /workspace/models
```

The script downloads several multi-gigabyte model files. Provision at least a 30 GB Network Volume for both profiles and the official ANIMA training base (more if you will keep training outputs or additional LoRAs). Ensure the correct model licenses permit your intended use. The script validates known SHA-256 hashes for the checkpoint/LoRA/SDXL VAE files where available. It intentionally does not download either unverified character LoRA. After it completes, stop the temporary Pod, attach the same Network Volume to the Serverless ComfyUI endpoint, and restart the endpoint so ComfyUI rescans the model directories.

For current RunPod availability across data centers, a Global Volume is the simplest shared-storage choice (RunPod announced Global Volumes in beta on October 8, 2026). If using a traditional Network Volume, keep all attached resources in its data center. Attach the same volume to the temporary model-download Pod, the ComfyUI Serverless endpoint, and the trainer Pod. Model files written to `/workspace/models` from the Pod become visible under `/runpod-volume/models` to Serverless workers.

The app calls the endpoint's `/run` and `/status/{job_id}` routes and sends a ComfyUI API-format workflow in `input.workflow`. The official RunPod ComfyUI worker returns generated PNGs in `output.images[]`, normally as base64 data or as a URL when S3 output is configured. Use the **Test GPU Endpoint** button in AI Image Studio > Settings to verify endpoint reachability. This test does not validate every installed model file.

The app now includes two distinct profile-specific workflows. Install all model files in the corresponding ComfyUI model folders:

For a normal ComfyUI install, the ANIMA profile uses:
- `models/diffusion_models/screenChantvMerge_v20.safetensors`
- `models/text_encoders/qwen_3_06b_base.safetensors`
- `models/vae/qwen_image_vae.safetensors`
- `models/loras/Turbo-ANIMA-v2.9.safetensors`
- Optional character LoRA: `models/loras/Ichinose_Chizuru.safetensors`

For the **official RunPod ComfyUI worker's Network Volume mapping**, the same ANIMA files should be placed under:
- `/runpod-volume/models/unet/screenChantvMerge_v20.safetensors`
- `/runpod-volume/models/clip/qwen_3_06b_base.safetensors`
- `/runpod-volume/models/unet/anima-base-v1.0.safetensors` (training base for the LoRA trainer)
- `/runpod-volume/models/vae/qwen_image_vae.safetensors`
- `/runpod-volume/models/loras/Turbo-ANIMA-v2.9.safetensors`
- Optional: `/runpod-volume/models/loras/Ichinose_Chizuru.safetensors`

RunPod's worker maps its `UNETLoader` files to `models/unet` and CLIP/text-encoder files to `models/clip`, which differs from a standard local ComfyUI directory layout. See [the official worker model-path mapping](https://github.com/runpod-workers/worker-comfyui/blob/main/src/extra_model_paths.yaml).

For **WAI Illustrious v1.3**, install:

- `models/checkpoints/waiNSFWIllustrious_v130.safetensors` (SDXL checkpoint, 6.94 GB; a public copy has fingerprint prefix `a810e710a2`: https://huggingface.co/elski/models-moved/blob/main/waiNSFWIllustrious_v130.safetensors)
- `models/vae/sdxl.vae.safetensors`
- `models/loras/のなかゆき.safetensors` (the second PNG identifies this LoRA at weight 0.8; hash `dffb5926186c` is not independently verified)
- `models/upscale_models/RealESRGAN_x4plus_anime_6B.pth`

On the official RunPod Network Volume these paths map to `/runpod-volume/models/checkpoints`, `/runpod-volume/models/vae`, `/runpod-volume/models/loras`, and `/runpod-volume/models/upscale_models`, respectively.

These names are configurable in the app. The checkpoint and the exact character LoRA require verification before their compatibility can be guaranteed. The character LoRA hash from the source PNG is `160fca5c6aae`, and remains unresolved. If the worker reports a missing model, install the correct file or disable the optional character LoRA. The ANIMA workflow uses `UNETLoader`, `CLIPLoader`, `VAELoader`, `LoraLoader` and `ModelSamplingAuraFlow`. WAI Illustrious uses the SDXL `CheckpointLoaderSimple` path, optional `CLIPSetLastLayer` for CLIP skip, and an external VAE and LoRA. Both profiles can run a high-resolution second pass using `UpscaleModelLoader`, `ImageUpscaleWithModel`, `ImageScale`, `VAEEncode`, `KSampler`, and `VAEDecode`. All required nodes and model files must be installed on the remote worker.

The Turbo-ANIMA profile recovers the first PNG's settings. The **WAI Illustrious v1.3** profile uses the second PNG's settings: checkpoint `waiNSFWIllustrious_v130`, LoRA `のなかゆき` at 0.8, Euler a, Automatic schedule (mapped to ComfyUI's `normal` scheduler), 30 steps, CFG 7, seed 624067427, base 896×1344, CLIP skip 2, ESRGAN Anime6B 2× high-resolution pass, 20 hires steps, and 0.5 denoise. ENSD and token-merging values are displayed for reference but are not applied by stock ComfyUI without compatible custom nodes. The Turbo-ANIMA profile recovers the supplied PNG's settings: positive/negative prompts, Euler a (ComfyUI `euler_ancestral`), Normal scheduler, 8 steps, CFG 1, shift 3, seed 1647498191, 848×1200 output, Turbo LoRA 1.0 and character LoRA 0.7. The stored PNG was 848×1200 while its generation metadata said 850×1200, so the UI defaults to the actual saved dimensions.

### Costs and credentials

### Concrete deployment settings

1. **Create persistent storage.** In RunPod Storage, create a Global Volume (beta) for shared models if available. Otherwise create a Network Volume and keep all attached resources in its data center. The model installer command above writes to `/workspace/models` on the Pod; the inference worker sees the files under `/runpod-volume/models`.
2. **Install the models once.** Start a temporary RunPod Pod with the volume attached at `/workspace`, open its web Terminal, run the installer command, and wait until every checksum/download completes. Do not terminate the Pod before the script finishes.
3. **Create the image endpoint.** Create a RunPod Serverless endpoint using the official `runpod-workers/worker-comfyui` worker image with a current ComfyUI version that supports ANIMA. Attach the same Global/Network Volume. Set Active Workers to 0, Max Workers to 1 initially, and an idle timeout such as 5 minutes to avoid paying for idle inference workers. Start with a 24 GB VRAM GPU such as an RTX 4090-class worker for both profiles. Once a generated image succeeds, adjust the worker cap to suit your budget. Copy the endpoint ID into AI Image Studio > Settings, enter the RunPod API key, and tap **Test GPU Endpoint**.
4. **Deploy the ANIMA trainer separately.** The training service is not part of the ComfyUI worker. Create a second GPU Pod using the Dockerfile in `cloud/anima-lora-trainer` (build it with Docker and push to a registry RunPod can pull, or deploy the repository through RunPod's GitHub deployment flow using `cloud/anima-lora-trainer` as context and `Dockerfile` as the Dockerfile path). Attach the same Global/Network Volume at `/workspace`, expose `8080` over HTTP through RunPod's HTTPS proxy, and set these environment variables on the container:
   - `MODEL_ROOT=/workspace/models`
   - `ANIMA_DIT_PATH=/workspace/models/unet/anima-base-v1.0.safetensors`
   - `QWEN3_PATH=/workspace/models/clip/qwen_3_06b_base.safetensors`
   - `ANIMA_VAE_PATH=/workspace/models/vae/qwen_image_vae.safetensors`
   - `LORA_INSTALL_DIR=/workspace/models/loras`
   - `TRAINER_JOB_ROOT=/workspace/rife60-anima-jobs`
   - `TRAINER_API_TOKEN` set to a long, random secret kept private
5. **Connect the trainer.** Copy the Pod's HTTPS proxy URL for port 8080, typically `https://POD_ID-8080.proxy.runpod.net`, into AI Image Studio > Settings > LoRA Trainer Endpoint. Enter the same `TRAINER_API_TOKEN` value. Tap the newly added **Test LoRA Trainer Endpoint** button. The test verifies the URL, token, CUDA availability, ANIMA training files, and training script. Keep the trainer Pod running during a training job. Stop it when not in use to stop GPU charges; LoRA training will be unavailable until it is started again.
6. **Train and use the LoRA.** The trainer installs completed LoRAs into the shared volume at `/workspace/models/loras`, which the inference endpoint sees as `/runpod-volume/models/loras`. If a new LoRA is not found by a warm ComfyUI worker, restart/recycle that worker so it rescans its model folder. The app also downloads the finished LoRA to its Documents folder for export.

The RunPod console/API key and the provider account are required to provision these resources. I cannot deploy them into your account from GitHub alone. Credentials entered into the iOS app are stored in iOS Keychain, not in source code or UserDefaults. GPU execution and persistent storage are billable. Keep the inference endpoint at zero minimum workers, stop the trainer Pod after use, and configure provider spending limits. GitHub Actions builds the IPA but does not provide GPU inference.

## Existing upscale path

After a PNG is returned, **Upscale with Real-CUGAN + Sharpie** passes the generated image URL to the existing `VideoProcessorViewModel.handleImport` and `start()` image pipeline. This uses the app's currently bundled Core ML Real-CUGAN model and existing final outline/colour pipeline. At native 2× mode, an 848×1200 generated image becomes 1696×2400. It does not claim to output 4K from an 848×1200 source.

## ANIMA LoRA trainer service

A deployable companion trainer API is included in cloud/anima-lora-trainer. It uses the ANIMA-specific anima_train_network.py entrypoint and implements the routes called by the app. See cloud/anima-lora-trainer/README.md to build and run it on a trusted NVIDIA GPU machine. The stock RunPod ComfyUI worker does not implement training routes. To automatically make a newly trained LoRA available to the generation worker, configure both services to access the same writable network volume and set LORA_INSTALL_DIR to the volume's loras directory.

- `POST /api/anima/lora/train`: multipart fields `caption`, `trigger_word`, `rank`, `epochs`, `learning_rate`, `base_model=anima`, and repeated `images` files. Return JSON `{"job_id":"..."}`.
- `GET /api/anima/lora/train/{job_id}`: return `{"status":"queued|running|completed|failed|cancelled","progress":0.0,"lora_url":"https://...","error":"..."}`. The completed response must include `lora_url` or `download_url`.

The trainer should use the ANIMA-specific training entrypoint, not an SDXL/Pony training script: https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md. It requires ANIMA DiT, Qwen3 text encoder, Qwen Image VAE, a dataset configuration and a CUDA GPU. Only run training on a trusted endpoint, use images you are authorized to train on, and remove uploaded datasets when no longer needed.

The app and trainer code are committed, but GPU services are not hosted by GitHub Actions. Image generation requires your RunPod endpoint, installed model files and API key. LoRA training requires the companion trainer container to be deployed to a trusted GPU host with the ANIMA training models mounted.
