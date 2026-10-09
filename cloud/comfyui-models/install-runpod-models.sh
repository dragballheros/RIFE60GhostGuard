#!/usr/bin/env bash
set -Eeuo pipefail

# Run inside a Pod with the official RunPod ComfyUI Network Volume mounted.
# Usage: bash install-runpod-models.sh [all|anima|wai] [/runpod-volume/models]
PROFILE="${1:-all}"
ROOT="${2:-/runpod-volume/models}"

case "$PROFILE" in
  all|anima|wai) ;;
  *) echo "Usage: $0 [all|anima|wai] [model-root]" >&2; exit 2 ;;
esac

command -v curl >/dev/null || { echo "curl is required." >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "sha256sum is required." >&2; exit 2; }

download_model() {
  local directory="$1"
  local filename="$2"
  local url="$3"
  local expected_sha256="${4:-}"
  local destination="$ROOT/$directory/$filename"
  local partial="$destination.part"

  mkdir -p "$(dirname "$destination")"

  if [[ -f "$destination" ]]; then
    if [[ -n "$expected_sha256" ]]; then
      echo "$expected_sha256  $destination" | sha256sum --check --status || {
        echo "Checksum mismatch for existing file: $destination" >&2
        exit 1
      }
    fi
    echo "Already present: $destination"
    return
  fi

  echo "Downloading: $filename"
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$partial" "$url"
  if [[ -n "$expected_sha256" ]]; then
    echo "$expected_sha256  $partial" | sha256sum --check
  fi
  mv "$partial" "$destination"
  echo "Installed: $destination"
}

install_anima() {
  # This merge checkpoint copy was previously identified in 123543o/124052.
  download_model "unet" "screenChantvMerge_v20.safetensors" \
    "https://huggingface.co/123543o/124052/resolve/main/Test/screenChantvMerge_v20.safetensors" \
    "4dbb10d55c611492394900b39d89eea38dc02edb2ba056fc538038babf19ab1d"

  download_model "clip" "qwen_3_06b_base.safetensors" \
    "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"

  download_model "vae" "qwen_image_vae.safetensors" \
    "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"

  download_model "loras" "Turbo-ANIMA-v2.9.safetensors" \
    "https://huggingface.co/Kutches/Anim4/resolve/main/Turbo-ANIMA-v2.9.safetensors" \
    "a5135d1cb868d4b3914cca6b32b1354e748c90488de1ab44a7dd7ea6ad003475"

  cat <<'NOTE'

ANIMA base installed. The optional Ichinose_Chizuru LoRA is not downloaded because
the exact file matching source-image hash 160fca5c6aae has not been verified.
The app leaves this character LoRA disabled by default so base generation can work.
NOTE
}

install_wai() {
  download_model "checkpoints" "waiNSFWIllustrious_v130.safetensors" \
    "https://huggingface.co/elski/models-moved/resolve/main/waiNSFWIllustrious_v130.safetensors" \
    "a810e710a2ee062824da335ad202ca24be1068dd08bf5513b57ff4c18c09877d"

  download_model "vae" "sdxl.vae.safetensors" \
    "https://huggingface.co/seigonasi/sdxl-vae-fp16/resolve/main/sdxl.vae.safetensors" \
    "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1"

  download_model "upscale_models" "RealESRGAN_x4plus_anime_6B.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

  cat <<'NOTE'

WAI Illustrious files installed. The optional のなかゆき LoRA is not installed by
this script because its exact hash has not been independently verified. The app
leaves character LoRAs disabled by default so the checkpoint can generate without it.
NOTE
}

echo "Installing model profile: $PROFILE"
if [[ "$PROFILE" == "all" || "$PROFILE" == "anima" ]]; then install_anima; fi
if [[ "$PROFILE" == "all" || "$PROFILE" == "wai" ]]; then install_wai; fi

echo
echo "Model download step finished. Refresh/restart the ComfyUI worker so it rescans its model folders."
echo "Model root: $ROOT"
