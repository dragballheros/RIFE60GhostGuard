#!/usr/bin/env bash
# Encode a short non-1080p MKV with the same static libraries the app links,
# then check container, pixel format, frame rate, and the x265 options SEI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${1:-/tmp/rife60-ffmpeg-x265}"
WORK="${TMPDIR:-/tmp}/rife60-x265-proof"
mkdir -p "$WORK"

bash "$ROOT/Scripts/build_ffmpeg_x265.sh" linux "$PREFIX"
g++ -std=c11 -O2 -DRIFE60_X265_PRESET_TOOL \
  -I"$PREFIX/include" \
  -x c "$ROOT/X265PresetTranscode.c" -x none \
  -Wl,--start-group \
  "$PREFIX/lib/libavformat.a" \
  "$PREFIX/lib/libavfilter.a" \
  "$PREFIX/lib/libavcodec.a" \
  "$PREFIX/lib/libswscale.a" \
  "$PREFIX/lib/libswresample.a" \
  "$PREFIX/lib/libavutil.a" \
  "$PREFIX/lib/libx265.a" \
  -Wl,--end-group \
  -lstdc++ -lm -lpthread -ldl -lz \
  -o "$WORK/x265-preset-transcode"

ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=1" \
  -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=1" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  "$WORK/sample.mkv"
"$WORK/x265-preset-transcode" "$WORK/sample.mkv" "$WORK/out.mov" > "$WORK/transcode.log" 2>&1

ffprobe -hide_banner -show_streams -show_format "$WORK/out.mov" > "$WORK/ffprobe.txt"
python3 - "$WORK/ffprobe.txt" "$WORK/out.mov" << 'PY'
import pathlib, sys
probe = pathlib.Path(sys.argv[1]).read_text()
blocks = probe.split("[STREAM]")
video = next(block for block in blocks if "codec_type=video" in block)
audio = next(block for block in blocks if "codec_type=audio" in block)

def field(block, name):
    for line in block.splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1]
    raise SystemExit(f"missing {name}")

checks = {
    "codec_name": "hevc",
    "profile": "Main 10",
    "codec_tag_string": "hvc1",
    "width": "1920",
    "height": "1080",
    "pix_fmt": "yuv420p10le",
    "r_frame_rate": "24000/1001",
}
for key, expected in checks.items():
    actual = field(video, key)
    if actual != expected:
        raise SystemExit(f"video {key}={actual}, expected {expected}")
for key, expected in {
    "codec_name": "aac",
    "profile": "LC",
    "sample_rate": "48000",
    "channels": "2",
    "channel_layout": "stereo",
}.items():
    actual = field(audio, key)
    if actual != expected:
        raise SystemExit(f"audio {key}={actual}, expected {expected}")
print("ffprobe matches Main 10 1920x1080 24000/1001 hvc1 AAC LC stereo 48 kHz")
PY
python3 - "$WORK/out.mov" "$WORK/options.txt" << 'PY'
import pathlib, sys
data = pathlib.Path(sys.argv[1]).read_bytes().decode("latin1", errors="ignore")
start = data.find("options:")
if start < 0:
    raise SystemExit("x265 options SEI was not found in the output")
text = data[start:start + 5000]
end = next((i for i, ch in enumerate(text) if ord(ch) < 32 or ord(ch) > 126), len(text))
pathlib.Path(sys.argv[2]).write_text(text[:end] + "\n", encoding="utf-8")
print(text[:end])
PY
python3 "$ROOT/Scripts/check_x265_preset_params.py" --options-file "$WORK/options.txt"
{
  echo "===== ffprobe ====="
  cat "$WORK/ffprobe.txt"
  echo "===== x265 options ====="
  cat "$WORK/options.txt"
} > /tmp/x265-preset-proof.txt
echo "x265 preset proof passed"
