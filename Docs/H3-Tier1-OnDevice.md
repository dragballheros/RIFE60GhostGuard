# MiniMax H3 Tier 1 (on-device only)

Chosen approach: **on-device install of already-published compressed or pruned weights**, one file at a time. The owner is on a Chromebook, so there is no Mac quantizer step.

## What this build does

`H3Tier1OnDeviceConverter` reads a manifest the owner sets (`RIFE60.H3.Tier1ManifestURL` is the settings key; the manifest itself is JSON with `path`, `url`, `sha256`, and `bytes`). For each file it:

1. Refuses the file before any download if it is larger than the 150 GB scratch ceiling or if free space cannot hold that one file plus a 2 GB margin.
2. Downloads that file into `tier1-scratch`.
3. Checks SHA-256.
4. Moves it into `tier1-weights` only when the kept store stays at or under about 44 GB.
5. Deletes the scratch file before the next file starts.

No weight bytes are sent to a paid API. Nothing in this path runs inference or training off the phone.

## What was dropped

- Encoder-student distillation.
- DMD training.
- Any paid cloud inference or training job.

## Why GitHub Actions is not the converter

A free GitHub-hosted runner does not have disk for the official MiniMax H3 tree (the Hugging Face repo is on the order of hundreds of gigabytes; one transformer partition is about 62 GB). This repo does not upload those weights as artifacts and does not run a quantization job there.

## What is not claimed

The Draw Things palettizer is a Mac tool and is not linked into the iPhone app. This path does not turn a full official BF16 shard into a q5p/q6p checkpoint on the phone. Tier 1 storage stays near 44 GB by **keeping published compressed weights and deleting everything that does not fit**, not by training a smaller model.

The older native downloader still knows how to fetch the pinned FL2VA/Ref2VA layout for the vendored `h3.c` runtime. That layout is not the Tier 1 44 GB store. Do not start it when the goal is the Tier 1 footprint.

Flash weight streaming for a Draw Things `.minimaxH3` graph still depends on a checkpoint that runtime can read. Until a license-compatible published checkpoint is in the manifest, the converter will not invent weights and will not fill the disk with the official tree.
