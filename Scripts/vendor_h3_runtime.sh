#!/usr/bin/env bash
set -euo pipefail

# Populate the native H3 source directory expected by project.yml from the
# exact upstream revision that passed the iOS portability compile/link audit.
# Model weights are not copied into the app; this only vendors source code.
readonly H3_REPO="https://github.com/JunzoKamahara/H3cApp.git"
readonly H3_REVISION="826fafffc84a15283cb7b3f4c234e7aedf07516c"
readonly WORK_DIR="${RUNNER_TEMP:-/tmp}/h3-ios-vendor"
readonly DEST_DIR="Vendor/minimax-h3"

rm -rf "$WORK_DIR" "$DEST_DIR"
git clone --quiet --filter=blob:none "$H3_REPO" "$WORK_DIR"
git -C "$WORK_DIR" checkout --quiet "$H3_REVISION"

python3 Scripts/patch_h3_ios_availability.py "$WORK_DIR/h3_gpu.m"
mkdir -p "$DEST_DIR"

c_sources=(
  h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c
  h3_dit_schedule.c h3_dit.c h3_lora.c h3_video_vae.c h3_video_encoder.c
  h3_audio_vae.c h3_vision_encoder.c h3_multimodal.c h3_attention_cache.c
)
objc_sources=(h3_metal.m h3_gpu.m h3_tokenizer.m h3_av_writer.m h3_av_reader.m)

for source in "${c_sources[@]}" "${objc_sources[@]}"; do
  test -s "$WORK_DIR/$source"
  cp "$WORK_DIR/$source" "$DEST_DIR/$source"
done

test -s "$WORK_DIR/h3.h"
test -s "$WORK_DIR/h3_shaders.metal"
cp "$WORK_DIR/h3.h" "$DEST_DIR/h3.h"
cp "$WORK_DIR/h3_shaders.metal" "$DEST_DIR/h3_shaders.metal"

echo "Vendored pinned H3 native source revision $H3_REVISION into $DEST_DIR."
echo "C/Objective-C source files: ${#c_sources[@]}+${#objc_sources[@]}; model weights are excluded."
