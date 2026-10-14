# DeepCreamPy v2 on-device conversion work

This directory is an isolated conversion workspace for the **DeepCreamPy v2** bar and mosaic generators. It is deliberately separate from the existing watermark-removal implementation.

## Non-negotiable isolation

Do not edit, rename, replace, or route through any of these existing watermark components:

- AnimeWatermarkRemover.swift
- WatermarkConfiguration.swift
- WatermarkConfiguration.modelID = "anime-lama-512-fp32-v1"
- AnimeWatermarkLaMa512.mlmodelc / AnimeWatermarkLaMa512.mlpackage

DeepCreamPy must get its own model IDs, weights, UI selection, preprocessing, and inference implementation.

## Checkpoint layout

Unpack the supplied model archive outside the app bundle. The expected directory layout is:

    <checkpoint-root>/
      bar/
        Train_775000.meta
        Train_775000.index
        Train_775000.data-00000-of-00001
      mosaic/
        Train_290000.meta
        Train_290000.index
        Train_290000.data-00000-of-00001

Run the preflight validator:

    python3 Scripts/DeepCreamPy/inspect_checkpoints.py --root /path/to/09-11-2019-DCPv2-model

If TensorFlow checkpoint-reading APIs are installed, the script also reads the checkpoint variable inventory and checks for generator (G_en, G_de) and contextual-attention (CB1) tensors. Without TensorFlow, it only validates archive layout and file signatures; it does not claim the weights are loadable.

## Model contract to validate

The matching upstream v2 graph uses three NHWC float inputs, each shaped [1, 256, 256, 3]:

- Placeholder:0 — masked/censored RGB crop (X)
- Placeholder_1:0 — reference/original RGB crop (Y)
- Placeholder_2:0 — mask (M), where 1 means preserve and 0 means reconstruct

The inference output in the matching graph is add:0, shaped [1, 256, 256, 3]. Input scaling and exact mask construction must match the upstream decensor pipeline; do not assume that an RGB image or a hand-painted green mask can be fed directly without reproducing preprocessing.

The generator path is encoder G_en → contextual attention CB1 → decoder G_de; the discriminator and training losses are not needed at inference. The contextual-attention path uses ExtractImagePatches, so a conversion is not complete until that operation is rewritten into supported operators or implemented by a supported on-device runtime. A Python custom-op callback is not acceptable for the final iOS app.

## Conversion acceptance gates

1. **Checkpoint integrity:** both checkpoint sets pass the validator; expected generator/attention variables exist and shapes are recorded.
2. **Reference inference:** the exact upstream graph loads each checkpoint and returns finite [1,256,256,3] output for deterministic test image/mask inputs.
3. **Portable graph:** convert to ONNX with native supported operators (no ExtractImagePatches custom op), check the graph, and run inference.
4. **Parity:** compare portable-runtime output with TensorFlow reference on identical tensors. Record max absolute error, MAE, and image-level comparisons for both models.
5. **iOS acceleration:** verify the chosen runtime on device and confirm its Core ML execution provider / Core ML model uses supported ops. Do not claim GPU/ANE acceleration based only on configuration.
6. **App integration:** add independent DeepCreamPy selectors and inference code without changing watermark-removal files.
7. **Build:** run the unsigned IPA workflow and inspect the actual result. A successful Swift compile alone does not prove the model loads or runs.

## Current status

This is the initial, isolated engineering branch. It adds checkpoint preflight tooling and documents the contract and acceptance gates. It does not yet contain a converted model or app-facing inference code. Model weights are not committed to the repository: their redistribution terms need to be confirmed first.
