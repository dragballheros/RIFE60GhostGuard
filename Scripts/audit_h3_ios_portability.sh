#!/usr/bin/env bash
set -euo pipefail

# Compile-and-link portability probe for the pinned AVFoundation H3 engine fork.
# This intentionally does not claim runtime readiness: API availability, app
# integration, model weights, memory, and device execution remain separate gates.
readonly H3_REPO="https://github.com/JunzoKamahara/H3cApp.git"
readonly H3_REVISION="826fafffc84a15283cb7b3f4c234e7aedf07516c"
readonly H3_DIR="${RUNNER_TEMP:-/tmp}/h3.c"
readonly OBJ_DIR="${RUNNER_TEMP:-/tmp}/h3-ios-objects"
readonly SDK="$(xcrun --sdk iphoneos --show-sdk-path)"

rm -rf "$H3_DIR" "$OBJ_DIR"
git clone --quiet --filter=blob:none "$H3_REPO" "$H3_DIR"
git -C "$H3_DIR" checkout --quiet "$H3_REVISION"
mkdir -p "$OBJ_DIR"

echo "Auditing H3cApp native engine revision $H3_REVISION for arm64 iOS 16 SDK compatibility."
echo "This is a compile-and-link portability probe; it does not assert runtime safety or device memory support."

c_sources=(
  h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c
  h3_dit_schedule.c h3_dit.c h3_lora.c h3_video_vae.c h3_video_encoder.c
  h3_audio_vae.c h3_vision_encoder.c h3_multimodal.c h3_attention_cache.c
)
objc_sources=(h3_metal.m h3_gpu.m h3_tokenizer.m h3_av_writer.m h3_av_reader.m)

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

# Build a static archive, then force-load every member into a tiny device-target
# link probe. This catches unresolved symbols/framework assumptions that a
# translation-unit-only compile cannot detect. It still does not run on device.
xcrun --sdk iphoneos ar rcs "$OBJ_DIR/libh3-ios.a" "$OBJ_DIR"/*.o
cat > "$OBJ_DIR/link_probe.c" <<'EOF'
#include "h3.h"
int main(void) {
    h3_ctx *ctx = h3_load_dir("/nonexistent-h3-model-for-link-probe");
    if (ctx) h3_free(ctx);
    return 0;
}
EOF
xcrun --sdk iphoneos clang -std=c11 -target arm64-apple-ios16.0 \
  -isysroot "$SDK" -I "$H3_DIR" "$OBJ_DIR/link_probe.c" \
  -Wl,-all_load "$OBJ_DIR/libh3-ios.a" \
  -framework Foundation -framework Metal \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
  -framework Accelerate -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework CoreAudio -framework CoreGraphics \
  -framework ImageIO -licucore -lm \
  -o "$OBJ_DIR/h3-ios-link-probe"

echo "All selected h3.c translation units compile and the iOS device-target link probe succeeds."
echo "This does not prove runtime availability safety, AVFoundation integration, model parity, or target-device feasibility."
