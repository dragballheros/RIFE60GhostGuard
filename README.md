# RIFE 60 Ghost Guard for iOS

> **Build 162 changes**
> - **Crash recovery now really resumes.** Each pass deletes the previous pass's checkpoint, so a resume now looks for the *furthest* valid checkpoint and skips everything before it (previously Compression Guard and RIFE were silently re-run). Settings of the interrupted job are restored on relaunch so the saved checkpoints always match, and the UI says so honestly if they cannot be reused.
> - **Final Sharpie is faster.** The ~120 tiles per 4K frame now run on up to 3 parallel workers (1–3 depending on the memory governor) instead of strictly one after another, and the per-pixel loops avoid per-pixel divides and branches. Output is unchanged.
> - **Images are supported** (Photos and Files). Images skip RIFE and run Compression Guard -> Real-CUGAN 2x -> Final Sharpie, saved as a lossless PNG.

A local iOS video frame-interpolation app built around **RifeMetal 0.1.6 / Practical-RIFE v4.26**.

## What this build does

- Select a video from Files.
- Generates a true **60.00 fps** output timeline, including 23.976/24/25/30 fps sources.
- Uses RIFE arbitrary-timestep interpolation rather than simply doubling frames.
- HQ / Balanced / Fast RIFE modes.
- Ghost Guard checks every synthetic frame.
- Scene cuts are not interpolated when the guard identifies them.
- Rejected synthetic frames are replaced by the nearest real source frame.
- H.264 or HEVC output.
- Original audio can be muxed back after interpolation.
- Processing is sequential so the whole video is never loaded into RAM.

## Why this is separate from waifu2x iOS

A waifu2x `.wifm` enhancement model receives one image at a time. RIFE requires two neighboring frames, so temporal interpolation cannot be correctly implemented inside the existing single-frame `.wifm` contract. Use this app for 60 fps, then process the output with the Sharpie/CUGAN `.wifm` in waifu2x iOS.

Recommended order for ordinary anime:

1. RIFE 60 Ghost Guard
2. Sharpie + CUGAN `.wifm`
3. upscale, if desired

For very blocky/compressed sources, CUGAN restoration before interpolation may produce cleaner motion estimation.

## Build without a Mac

1. Create a GitHub repository and upload this entire folder.
2. Open the **Actions** tab.
3. Open **Build Unsigned IPA**.
4. Tap **Run workflow**.
5. Download the `RIFE60GhostGuard-unsigned-ipa` artifact.
6. Extract the artifact ZIP; inside is `RIFE60GhostGuard-unsigned.ipa`.
7. Sign/sideload that IPA with your normal sideloading method.

The workflow builds on a GitHub-hosted macOS/Xcode runner, because an actual iOS device binary cannot be compiled on this Linux workspace.

## Recommended settings

- 1080p anime: **Balanced** first.
- Maximum quality / modest resolution: **HQ**.
- 4K or thermally constrained phone: **Fast**.
- Ghost Protection: **On**.
- Scene-cut protection: **On**.
- Sensitivity: **1.00** initially.
- H.264: widest compatibility.
- HEVC: smaller output at similar visual quality.

## Ghost Guard logic

Each synthetic frame is checked using downsampled luma statistics:

- endpoint scene-change magnitude,
- generated-frame temporal distance to both real frames,
- edge-energy spike detection for double contours / ghosts,
- generated midpoint deviation from a simple temporal midpoint.

If a check fails, the generated image is discarded and the nearest original frame is emitted at that 60-fps timestamp.

## Current upstream limitation

RifeMetal states that iOS 16+ targets compile, but upstream has not yet runtime-validated the package on iOS hardware. This project is structured for device testing; if an iOS-specific MPSGraph/Metal issue appears on your phone, the build log/runtime error will identify the porting point rather than requiring a redesign of the video pipeline.

## Licensing

This project references `cinemore/rife-metal` as a Swift Package dependency. RifeMetal is Apache-2.0 and its bundled Practical-RIFE-derived v4.26 weights are attributed upstream under MIT. Keep upstream license/attribution notices when redistributing.
