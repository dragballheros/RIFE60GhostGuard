# RIFE60 GhostGuard for Windows

Command-line and windowed RIFE interpolation for a Windows 11 PC, including an Intel MacBook Pro 2019 (64 GB, AMD Radeon Pro 5300M, 5500M, or 5600M) booted into Windows. The iOS app is unchanged.

The GPU path is **ncnn Vulkan**, the same runtime as [nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan). Navi 14 (the 5300M/5500M/5600M) is supported by AMD's Windows Vulkan driver, and that project already runs this network on it: custom warp shader, fp16 storage, and the published rife-v4.x `flownet` models. A wgpu port would not load those models. The build links that runtime in-process (`rife_capi.dll`). It does not shell out to `rife-ncnn-vulkan.exe`.

The installer includes **rife-v4.6** and **rife-v4.26**. 4.6 is the model from the official ncnn-vulkan release. 4.26 is the ncnn flownet published with video2x (about 11 MB). The Windows runtime is the pinned rife-ncnn-vulkan commit, patched so a directory named `rife-v4.26` uses 64-pixel padding and `rife-v4.6` stays at 32. The window defaults to 4.26. `--rife 4.6` keeps the previous command-line model. `--model <dir>` still points at any flownet directory.

Default tile size is 512. That stays inside the 4 GB framebuffer on the 5300M and 5500M. `--tile 0` runs the whole frame when you know it fits (the 5600M has 8 GB). The loader picks a GPU whose name contains AMD or Radeon. On Boot Camp that is the discrete GPU, not the Intel iGPU. `--gpu N` overrides it.

## Installer

`build-windows.ps1` produces one setup program:

`rife-pc\installer\output\RIFE60GhostGuard-Setup.exe`

It installs to `C:\Program Files\RIFE60 GhostGuard`, adds that folder to the system PATH, adds a Start Menu entry that opens the window, and registers an uninstaller. The folder contains `rife-pc.exe`, `rife_capi.dll` (ncnn Vulkan), `realcugan-ncnn-vulkan.exe`, `ort_capi.dll` and `onnxruntime.dll` (DirectML), `ffmpeg.exe` / `ffprobe.exe` (GPL build with libx265), `models\rife-v4.6`, `models\rife-v4.26`, and `models\models-se` (Real-CUGAN 2× noise 3). Multi-gigabyte H3, Animagine, and converted ONNX networks are downloaded later.

Vulkan itself comes from the AMD driver (`vulkan-1.dll` in System32), not from this installer.

## Windows 11 setup

1. Install the latest **AMD Software: Adrenalin** driver for the Radeon Pro, then reboot. Device Manager should list the 5300M, 5500M, or 5600M, and `C:\Windows\System32\vulkan-1.dll` should exist.
2. Install [Git for Windows](https://git-scm.com/download/win).
3. Install [CMake](https://cmake.org/download/) and make sure `cmake` is on PATH.
4. Install the [Rust](https://rustup.rs/) stable toolchain (`x86_64-pc-windows-msvc`). The script installs rustup if `rustc` is missing.
5. The script installs the LunarG Vulkan SDK **1.4.363.1** if `VULKAN_SDK` is not set. That SDK is only needed to compile ncnn. The installed app uses the driver.
6. From this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build-windows.ps1
```

The script clones rife-ncnn-vulkan at `a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7`, builds `rife_capi.dll`, builds `rife-pc.exe` with `--features ncnn,gui`, downloads the FFmpeg 8.1 GPL build (libx265) and the rife-v4.6 model, then compiles the Inno Setup installer. It downloads Inno Setup 7.1.1 if `ISCC.exe` is not already installed.

7. Run `installer\output\RIFE60GhostGuard-Setup.exe`. After it finishes, open a **new** terminal so PATH is picked up.

## Run

Window (also the Start Menu shortcut):

```text
rife-pc gui
```

Command line, 2× (one interpolated frame between each source frame, output rate = source × 2):

```text
rife-pc input.mp4 -o output.mp4 --multi 2
```

`--multi 2` is the default, so this is the same:

```text
rife-pc input.mp4 -o output.mp4
```

Target 60 fps instead of a fixed multiplier:

```text
rife-pc input.mp4 -o output.mp4 --fps 60
```

`--fps` wins when both are set. Progress lines look like `rife-pc: 120/1440 frames  36.2 fps  source 60`.

Other flags follow the iOS toggles. The plain command above stays guards-on, audio-on, and does not upscale or grade. Add the flags to match a GUI setup:

`--rife 4.26`, `--no-ghost`, `--no-scene-cuts`, `--no-thin-line`, `--no-line-stabilizers`, `--no-outlier`, `--no-audio`, `--render-power`, `--tier auto|performance|balanced|safe`, `--reddit`, `--gif`, `--compression`, `--color-pop`, `--color-pop-strength 1`, `--warmth off|original|processed|full`, `--skin-tone off|original|processed|full`, `--color-rebuild`, `--outline-rebuild`, `--sharpie`, `--anime-restore`, `--anime-quality auto|apisr|span`, `--upscale`, `--upscale-first`, `--watermark`, `--watermark-model lama|bar|mosaic`, `--diagnostics`, `--export-dir`, `--also clip2.mp4`, `--sensitivity 1.35`, `--gpu N`, `--tile 512`.

`rife-pc h3 --prompt "..." --accept-license -o clip.mp4` runs MiniMax H3 through a downloaded stable-diffusion.cpp CPU build. `rife-pc studio --prompt "..."` runs Animagine XL. `rife-pc download <id>` fetches one catalog file. The window has the same controls.

## What the window adds

The iOS toggles are on the Interpolate, Picture, Scale, Watermark, Export, and Queue tabs. Wan VACE is not ported.

- **RIFE 4.26 and 4.6.** Ghost protection, scene-cut protection, thin-line preservation, line stabilizers, and the source outlier guard.
- **Render power mode** calls `SetThreadExecutionState` so Windows stays awake. It does not enlarge the RIFE tile past the 4 GB framebuffer on the 5300M and 5500M. Safe mode caps the tile at 256. Auto on a 64 GB machine with power mode on is the Performance tier.
- **Preserve original audio** stream-copies the source audio onto the final HEVC file. GIF export drops audio.
- **Reddit Mode** re-encodes toward 19.8 MB (20 MB hard cap). Stills try AVIF, then JPEG. **Convert video to GIF** searches frame rate and long-edge the way the iOS exporter does. The iOS GIF loop-seam picker is approximated by ffmpeg's palette.
- **Compression Guard, Color Pop, Color Rebuild, automatic warmth, and the skin-tone picker** are the iOS pixel formulas on the CPU.
- **Outline Rebuild** is the classical gap closer. **Final Sharpie Outline** runs `AnimeSharpieOutlineSubtle.onnx` when that file is present (94% network, 6% source, blush strokes put back). Without the file it darkens lines the mask already found and the status line says so.
- **Anime Restore** tiles 256 with a 16-pixel overlap, runs APISR or SPAN, scales the 512 output back to 256, and blends by strength. Auto prefers APISR when one tile is under the iOS 1000 ms budget (the iOS measurement is halved, same as `duplicate_divisor`). Short-side 720 and below turns it on unless "do not auto-enable" is checked. The ONNX files are not in the installer. `tools/convert_coreml_to_onnx.py` is the conversion step from the iOS Core ML packages.
- **Real-CUGAN 2×** is `realcugan-ncnn-vulkan.exe` with `models-se` noise 3 and sync gap 3. A second pass runs when the long edge is under 1080 and 4× stays within 40 megapixels. **Upscale before interpolation** matches the iOS upscale-first order. The iOS intensity 1.30 is an input of the Core ML graph and is not a separate ncnn flag. ncnn's default GPU is used unless you set the GPU index; RIFE itself still picks an AMD device by name.
- **Watermark removal** paints a normalized mask in the window. Each queue item has its own toggle. LaMa and DeepCream run through ONNX when the files exist. Otherwise the marked pixels are filled from nearby unmasked pixels. DeepCream preprocessing keeps the iOS convention: 1 means preserve, 0 means inpaint, bar models blank unknown pixels to -1, mosaic models keep the mosaic texture, and the tensor is repeated to batch 8. DeepCreamPy is CC BY-NC-SA 4.0 (non-commercial). LaMa's fine-tune is MIT and the base model is Apache-2.0.
- **Queue and export folder.** The queue runs one file after another.
- **Live readout** shows the stage, frames, fps, free RAM, and a rough ETA. **Broken-tile diagnostics** score grid-aligned luminance steps every 15 frames and write `run.json` plus up to five BMP frames.
- **MiniMax H3** and **AI Image Studio** download stable-diffusion.cpp on first use (CPU zip, or the Vulkan zip for low-VRAM offload) and the GGUF or safetensors weights into `%LOCALAPPDATA%\RIFE60GhostGuard`. They are not in the installer. H3 uses `leejet/MiniMax-H3-GGUF` Ref2VA `minimax_h3_ref2va_pruned-Q4_K_M.gguf` and `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, plus the Comfy-Org video and audio VAEs. That repo does not publish Q8_0. The Community License excludes the United States, the European Union, the United Kingdom, and South Korea unless MiniMax has authorized that use. The window shows the warning and will not download or generate until the box is checked. Generation uses `--offload-to-cpu` so the weights sit in the 64 GB of RAM. The time estimate is a rough 2019-laptop figure, not a benchmark: a 56-frame clip is hours on CPU.
- **AI Image Studio** runs Animagine XL v3.1 (832×1216, 28 steps, CFG 5, Euler a, clip skip 2) and can run Outline Rebuild on the still. Anima is listed because the iOS picker has it. Its upstream license is non-commercial, and this build does not start the Cosmos flow-matching stack.

## Licenses that affect use

MiniMax H3: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE

DeepCreamPy: CC BY-NC-SA 4.0. Do not use those weights commercially.

Anima / the iOS catalog model: non-commercial upstream terms. Animagine XL is a separate SDXL checkpoint; read its model card before redistributing the weights. The app does not bundle either file.

## Encode

ffmpeg reads and writes raw BGRA through pipes. Video is libx265, preset `medium`, profile **main10**, pixel format `yuv420p10le`, tag `hvc1`, with the DJATOM parameter string from `X265PresetParams.h` (CRF 20, psy-rd 2.00, aq-mode 3, and the rest of that preset). `fps`, `colorprim`, `transfer`, `colormatrix`, and `range` are rewritten from the source so the file keeps the real rate and the source color metadata. Audio is stream-copied (`-c:a copy`). Frames smaller than one 64×64 CTU use a smaller `ctu` so x265 can open; 1080p and above keep `ctu=64`.

## Guards

CPU, row-parallel with rayon, same thresholds as the Swift code:

- `StructuralSceneCut` and its cross-fade test
- `GhostGuard` (scene cuts repeat the nearer real frame; other rejects keep the interpolation)
- `SourceFrameOutlierGuard` (a cross-fade parked on a cut repeats the nearer neighbor; a flash is bridged)
- `ThinLinePreservationPass`
- `StaticRegionLock`
- `SmallMotionLineStabilizer`
- Stroke width from commit 7d8c9be: the in-between mean line width must stay within 20% of the nearer source, and the stroke must not be darker. Otherwise that real frame is repeated. With no verified motion and the source strokes not inside the in-between, the RIFE frame is left alone.

`cargo test` in this folder runs those checks. They mirror `Scripts/check_thin_line_preservation.swift`.

Decode, GPU inference, guards, and encode each have their own thread and a bounded queue (4 decoded frames, 2 in-flight GPU results, 2 frames waiting to encode).
