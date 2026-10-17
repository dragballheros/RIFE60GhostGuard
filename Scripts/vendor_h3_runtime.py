#!/usr/bin/env python3
"""Fetch and stage the pinned MiniMax H3 native runtime for XcodeGen builds."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = "826fafffc84a15283cb7b3f4c234e7aedf07516c"
REPOSITORY = "https://github.com/JunzoKamahara/H3cApp.git"
SOURCES = [
    "h3.c", "h3_host.c", "h3_safetensors.c", "h3_weights.c",
    "h3_text_encoder.c", "h3_dit_schedule.c", "h3_dit.c", "h3_lora.c",
    "h3_video_vae.c", "h3_video_encoder.c", "h3_audio_vae.c",
    "h3_vision_encoder.c", "h3_multimodal.c", "h3_attention_cache.c",
    "h3_metal.m", "h3_gpu.m", "h3_tokenizer.m",
    "h3_av_writer.m", "h3_av_reader.m",
    "h3.h", "h3_host.h", "h3_internal.h", "h3_safetensors.h",
    "h3_weights.h", "h3_text_encoder.h", "h3_dit_schedule.h",
    "h3_dit.h", "h3_lora.h", "h3_video_vae.h", "h3_video_encoder.h",
    "h3_audio_vae.h", "h3_vision_encoder.h", "h3_multimodal.h",
    "h3_gpu.h", "h3_metal.h",
    "h3_av_writer.h", "h3_av_reader.h", "h3_tokenizer.h",
    "h3_shaders.metal", "LICENSE",
]

def main():
    temp = ROOT / ".h3-vendor-source"
    target = ROOT / "Vendor" / "minimax-h3"
    shutil.rmtree(temp, ignore_errors=True)
    subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", REPOSITORY, str(temp)], check=True)
    subprocess.run(["git", "-C", str(temp), "checkout", "--quiet", REVISION], check=True)
    patcher = ROOT / "Scripts" / "patch_h3_ios_availability.py"
    subprocess.run(["python3", str(patcher), str(temp / "h3_gpu.m")], check=True)

    staged = target.with_name("minimax-h3.staging")
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)
    missing = []
    for name in SOURCES:
        source = temp / name
        if not source.is_file():
            missing.append(name)
        else:
            shutil.copy2(source, staged / name)
    if missing:
        shutil.rmtree(staged, ignore_errors=True)
        raise SystemExit("Pinned H3 revision is missing required files: " + ", ".join(missing))
    shutil.rmtree(target, ignore_errors=True)
    staged.rename(target)
    shutil.rmtree(temp, ignore_errors=True)
    print(f"Staged {len(SOURCES)} pinned H3 runtime files from {REVISION} at {target}")

if __name__ == "__main__":
    main()
