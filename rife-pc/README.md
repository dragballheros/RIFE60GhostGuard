# RIFE60 GhostGuard for Windows

Command-line and windowed RIFE interpolation for a Windows 11 PC, including an Intel MacBook Pro 2019 (64 GB, AMD Radeon Pro 5300M, 5500M, or 5600M) booted into Windows. The iOS app is unchanged.

The GPU path is **ncnn Vulkan**, the same runtime as [nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan). Navi 14 (the 5300M/5500M/5600M) is supported by AMD's Windows Vulkan driver, and that project already runs this network on it: custom warp shader, fp16 storage, and the published rife-v4.x `flownet` models. A wgpu port would not load those models. The build links that runtime in-process (`rife_capi.dll`). It does not shell out to `rife-ncnn-vulkan.exe`.

The bundled model is **rife-v4.6**, the newest v4 model in the official ncnn-vulkan release (4.6). The iOS app's 4.26 weights are a Core ML build and are not in that release. A later v4 export with the same `in0` / `in1` / `in2` / `out0` flownet can be passed with `--model`.

Default tile size is 512. That stays inside the 4 GB framebuffer on the 5300M and 5500M. `--tile 0` runs the whole frame when you know it fits (the 5600M has 8 GB). The loader picks a GPU whose name contains AMD or Radeon. On Boot Camp that is the discrete GPU, not the Intel iGPU. `--gpu N` overrides it.

## Installer

`build-windows.ps1` produces one setup program:

`rife-pc\installer\output\RIFE60GhostGuard-Setup.exe`

It installs to `C:\Program Files\RIFE60 GhostGuard`, adds that folder to the system PATH, adds a Start Menu entry that opens the window, and registers an uninstaller. The folder contains `rife-pc.exe`, `rife_capi.dll` (ncnn Vulkan), `ffmpeg.exe` / `ffprobe.exe` (GPL build with libx265), and `models\rife-v4.6`.

Vulkan itself comes from the AMD driver (`vulkan-1.dll` in System32), not from this installer.

## Windows 11 setup

1. Install the latest **AMD Software: Adrenalin** driver for the Radeon Pro, then reboot. Device Manager should list the 5300M, 5500M, or 5600M, and `C:\Windows\System32\vulkan-1.dll` should exist.
2. Install [Git for Windows](https://git-scm.com/download/win).
3. Install [CMake](https://cmake.org/download/) and make sure `cmake` is on PATH.
4. Install the [Rust](https://rustup.rs/) stable toolchain (`x86_64-pc-windows-msvc`). The script installs rustup if `rustc` is missing.
5. The script installs the LunarG Vulkan SDK **1.3.290.0** if `VULKAN_SDK` is not set. That SDK is only needed to compile ncnn. The installed app uses the driver.
6. From this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build-windows.ps1
```

The script clones rife-ncnn-vulkan at `a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7`, builds `rife_capi.dll`, builds `rife-pc.exe` with `--features ncnn,gui`, downloads the BtbN FFmpeg 7.1 GPL build (libx265) and the rife-v4.6 model, then compiles the Inno Setup installer. It downloads Inno Setup 6.3.3 if `ISCC.exe` is not already installed.

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

Other flags: `--model <dir>`, `--gpu N`, `--tile 512`, `--sensitivity 1.35` (the iOS range is 0.65 to 1.35).

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
