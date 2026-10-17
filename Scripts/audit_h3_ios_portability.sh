#!/usr/bin/env bash
set -euo pipefail

# Compile-only portability audit. This intentionally does not claim runtime
# readiness: AVFoundation adapters, model weights, memory, and device execution
# are separate acceptance gates.
readonly H3_REPO="https://github.com/antirez/h3.c.git"
readonly H3_REVISION="8974cc055ea9c02fcd14cc27dfda3e1027c05153"
readonly H3_DIR="${RUNNER_TEMP:-/tmp}/h3.c"
readonly OBJ_DIR="${RUNNER_TEMP:-/tmp}/h3-ios-objects"
readonly SDK="$(xcrun --sdk iphoneos --show-sdk-path)"

rm -rf "$H3_DIR" "$OBJ_DIR"
git clone --quiet --filter=blob:none "$H3_REPO" "$H3_DIR"
git -C "$H3_DIR" checkout --quiet "$H3_REVISION"
mkdir -p "$OBJ_DIR"

echo "Auditing h3.c revision $H3_REVISION for arm64 iOS 16 SDK compatibility."
echo "This is a compile-only audit; it does not assert iOS runtime or device memory support."

c_sources=(
  h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c
  h3_dit_schedule.c h3_dit.c h3_video_vae.c h3_video_encoder.c
  h3_audio_vae.c h3_ffmpeg.c h3_terminal.c h3_vision_encoder.c
  h3_multimodal.c
)
objc_sources=(h3_metal.m h3_gpu.m h3_tokenizer.m)

for source in "${c_sources[@]}"; do
  echo "::group::iOS compile $source"
  xcrun --sdk iphoneos clang -std=c11 -O1 -D_DARWIN_C_SOURCE \
    -target arm64-apple-ios16.0 -isysroot "$SDK" -I "$H3_DIR" \
    -Wno-deprecated-declarations -c "$H3_DIR/$source" \
    -o "$OBJ_DIR/${source//\//_}.o"
  echo "::endgroup::"
done

for source in "${objc_sources[@]}"; do
  echo "::group::iOS compile $source"
  xcrun --sdk iphoneos clang -x objective-c -std=c11 -O1 -fobjc-arc \
    -fmodules -D_DARWIN_C_SOURCE -target arm64-apple-ios16.0 \
    -isysroot "$SDK" -I "$H3_DIR" -Wno-deprecated-declarations \
    -c "$H3_DIR/$source" -o "$OBJ_DIR/${source//\//_}.o"
  echo "::endgroup::"
done

echo "All selected h3.c translation units compile against the iOS SDK."
echo "This does not prove link success, AVFoundation integration, model parity, or target-device feasibility."
