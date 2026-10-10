#!/usr/bin/env python3
"""Apply app-specific low-memory policy to the pinned Draw Things engine.

The upstream default treats SDXL as resident on devices with >=5 GiB physical RAM.
That is too aggressive for a multi-feature iPhone app sharing unified memory with
Metal, video buffers, and the UI. Force file-backed/on-demand weights for SDXL's
UNet, text encoder, and diffusion mapping. Keep VAE decoding on its native path.
"""
from pathlib import Path

path = Path("Vendor/draw-things-community/Libraries/LocalImageGenerator/Sources/ModelPreloader.swift")
source = path.read_text()
needle = """  ) -> Bool {
    switch externalStore {
    case .alwaysFully:
"""
replacement = """  ) -> Bool {
    // RIFE60GhostGuard runs alongside other GPU/video features on iPhone. The
    // upstream physical-RAM heuristic considers a 6 GiB iPhone a "good" device,
    // but does not account for iOS process limits or shared Metal memory. Keep
    // SDXL weight tensors file-backed and load them on demand for the heavy
    // components; this trades throughput for a lower resident working set.
    if version == .sdxlBase || version == .sdxlRefiner {
      switch variant {
      case .unet, .textEncoder, .diffusionMapping, .autoencoder:
        return DeviceCapability.externalOnDemand(
          version: version, scale: scale, force: true, suffix: suffix,
          is8BitModel: is8BitModel)
      case .control, .autoencoder:
        break
      }
    }
    switch externalStore {
    case .alwaysFully:
"""
if source.count(needle) != 1:
    raise SystemExit(f"Expected one ModelPreloader patch target, found {source.count(needle)}")
source = source.replace(needle, replacement)
path.write_text(source)
print("Applied app-specific SDXL disk-backed weight streaming policy.")
