# RIFE60 ANIMA LoRA trainer service

This companion API implements the endpoints called by the AI Image Studio LoRA Trainer tab. It trains ANIMA LoRAs only, using the official kohya-ss/sd-scripts anima_train_network.py entrypoint.

## Deploy on a GPU machine

Build and run this service on a trusted NVIDIA CUDA GPU machine. It is not part of the iOS IPA and is not hosted by GitHub Actions. The easiest RunPod setup is a dedicated GPU Pod for training, started only when needed; the stock ComfyUI Serverless worker cannot serve these training routes. Keep the trainer Pod running from submission until the LoRA completes because the training subprocess runs inside that process.

Required model files under /models:

- anima-base-v1.0.safetensors: ANIMA DiT training base
- qwen_3_06b_base.safetensors: Qwen3-0.6B text encoder
- qwen_image_vae.safetensors: Qwen Image VAE

These are available from https://huggingface.co/circlestone-labs/Anima. Mount their files at /models, or set ANIMA_DIT_PATH, QWEN3_PATH, and ANIMA_VAE_PATH to their mounted paths. This service trains against official ANIMA Base v1.0. Do not train a WAI Illustrious/SDXL LoRA with this service.

Build:

```bash
docker build -t rife60-anima-lora-trainer cloud/anima-lora-trainer
```

Run with model storage mounted and a long random token:

```bash
docker run -d --name rife60-anima-lora-trainer --gpus all \
  -p 8080:8080 \
  -v /path/to/anima-models:/models:ro \
  -v rife60-anima-jobs:/workspace/rife60-anima-jobs \
  -e TRAINER_API_TOKEN='replace-with-a-long-random-secret' \
  -e LORA_INSTALL_DIR='/runpod-volume/models/loras' \
  rife60-anima-lora-trainer
```

Set AI Image Studio > Settings > LoRA Trainer Endpoint to the service's HTTPS URL and paste the same token. Do not expose the port directly to the public internet without HTTPS/reverse proxy, firewalling and a unique token. The app uploads the selected images to this server.

### Optional shared volume for automatic LoRA installation

The model installer script also downloads the official `anima-base-v1.0.safetensors` specifically for training. If your GPU host can mount the same writable network volume that the inference worker uses, mount it at `/runpod-volume` and configure these environment variables:

```bash
-e MODEL_ROOT=/workspace/models \
-e ANIMA_DIT_PATH=/workspace/models/diffusion_models/anima-base-v1.0.safetensors \
-e QWEN3_PATH=/workspace/models/text_encoders/qwen_3_06b_base.safetensors \
-e ANIMA_VAE_PATH=/workspace/models/vae/qwen_image_vae.safetensors \
-e TRAINER_JOB_ROOT=/workspace/rife60-anima-jobs \
-e LORA_INSTALL_DIR=/workspace/models/loras
```

The trainer Pod sees the attached RunPod volume at `/workspace`; a Serverless worker sees the same volume at `/runpod-volume`. The current worker image maps ComfyUI's `UNETLoader` and `CLIPLoader` through `diffusion_models` and `text_encoders`; these are the directories configured by the custom worker image built by this repository. The trainer copies a completed LoRA into the shared `loras` directory and returns `installed_filename`; the app can then select it automatically for Turbo-ANIMA. If ComfyUI does not detect a newly written file immediately, recycle its worker. If your provider cannot make the volume accessible to both services, leave `LORA_INSTALL_DIR` unset and manually upload the downloaded LoRA into the inference worker's LoRA folder after training.

## API contract

- POST /api/anima/lora/train: authenticated multipart form with caption, trigger_word, rank, epochs, learning_rate, base_model=anima, and 3 to 40 images files. Returns a job ID.
- GET /api/anima/lora/train/{job_id}: returns job status/progress and a download URL after completion.
- GET /api/anima/lora/train/{job_id}/download: authenticated trained LoRA download.
- GET /health: non-sensitive readiness summary.

Training jobs run one at a time. The service caps datasets at 180 MiB and removes uploaded training images/captions after training; output LoRAs, logs, and status remain in the persistent job volume. Remove completed job output when it is no longer needed.

## Training details

The service creates a 1024px bucketed dataset and caption files using the trigger word plus shared caption, then invokes the ANIMA training script with the configured rank, learning rate, epoch count, gradient checkpointing, latent cache and text-encoder output cache. The upstream guide is https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md.

This is deployment starter code. Test against your chosen GPU image, model versions and dataset before training valuable data. GPU time and persistent storage are billed by the hosting provider. Upload only datasets you are authorized to use.
