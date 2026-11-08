#!/usr/bin/env python3
"""Resolve every audio/video VAE conv name against the published key lists.

The fixtures are the safetensors header names from
Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors (605,254,808 bytes)
and vae/minimax_h3_video_vae_fp16.safetensors (5,207,808,496 bytes). They contain
folded `.weight` tensors, not weight_g/weight_v.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO = (ROOT / "Scripts/fixtures/h3_audio_vae_fp32_keys.txt").read_text(encoding="utf-8").split()
VIDEO = (ROOT / "Scripts/fixtures/h3_video_vae_fp16_weight_keys.txt").read_text(encoding="utf-8").split()
AUDIO_SET = set(AUDIO)


def resolves(prefix: str) -> bool:
    return (
        f"{prefix}.weight" in AUDIO_SET
        or (f"{prefix}.weight_v" in AUDIO_SET and f"{prefix}.weight_g" in AUDIO_SET)
        or (
            f"{prefix}.parametrizations.weight.original0" in AUDIO_SET
            and f"{prefix}.parametrizations.weight.original1" in AUDIO_SET
        )
    )


def audio_prefixes() -> list[str]:
    prefixes = ["decoder.conv_pre", "decoder.conv_post", "encoder.block.0", "encoder.block.7"]
    for index in range(7):
        prefixes.append(f"decoder.ups.{index}.0")
        for block_index in range(3):
            global_index = index * 3 + block_index
            for pair in range(3):
                prefixes.append(f"decoder.resblocks.{global_index}.convs1.{pair}")
                prefixes.append(f"decoder.resblocks.{global_index}.convs2.{pair}")
    for stage in range(1, 6):
        for residual in range(3):
            prefixes.append(f"encoder.block.{stage}.block.{residual}.block.1")
            prefixes.append(f"encoder.block.{stage}.block.{residual}.block.3")
        prefixes.append(f"encoder.block.{stage}.block.4")
    return prefixes


def main() -> None:
    missing = [prefix for prefix in audio_prefixes() if not resolves(prefix)]
    if missing:
        raise SystemExit("audio VAE keys do not resolve: " + ", ".join(missing))
    if any(key.endswith("weight_v") or "parametrizations" in key for key in AUDIO):
        raise SystemExit("unexpected unfolder audio key; the loader must still accept folded weights")
    if any(not key.endswith(".weight") or key.endswith("weight_v") for key in VIDEO):
        raise SystemExit("video VAE fixture is not a folded .weight list")
    video = set(VIDEO)
    for required in ("encoder.conv_in.weight", "decoder.norm_out.weight", "decoder.proj_out.weight", "decoder.x_embedder.weight"):
        if required not in video:
            raise SystemExit(f"video VAE fixture is missing {required}")
    print(f"PASS: {len(audio_prefixes())} audio weight-norm convs resolve to folded .weight keys")
    print(f"PASS: {len(VIDEO)} video VAE weights are already folded .weight tensors")


if __name__ == "__main__":
    main()
