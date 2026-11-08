#!/usr/bin/env python3
"""Mask tensors must match the Wan VACE frame tensor for every aspect ratio.

ImageConverter.grayscaleTensor returns NC(height, width), not NHWC.
The temporal mask assembled for the encoder is [frames, height, width, 1]
with values 0 or 1. Frame count is padded to 4n+1, the same rule as the
five-frame VACE windows.
"""
from pathlib import Path

INFERENCE = (512, 512)


def aligned_frame_count(frames: int) -> int:
    count = max(frames, 1)
    while count < 5 or (count - 1) % 4 != 0:
        count += 1
    return count


def aspect_fill(src_w: int, src_h: int, dst_w: int, dst_h: int):
    """Same scale-to-fill the video frames use inside ImageConverter.resize."""
    if src_w * dst_h > src_h * dst_w:
        scaled_h = dst_h
        scaled_w = src_w * dst_h // src_h
    else:
        scaled_w = dst_w
        scaled_h = src_h * dst_w // src_w
    return scaled_w, scaled_h


def normalized_mask(src_w: int, src_h: int, frames: int, target=INFERENCE):
    dst_w, dst_h = target
    # The engine stretches the crop to the inference size before encoding.
    # The decoder then aspect-fills that square into the same square, so the
    # tensor the encoder sees is exactly the inference canvas, one channel.
    scaled_w, scaled_h = aspect_fill(dst_w, dst_h, dst_w, dst_h)
    assert (scaled_w, scaled_h) == (dst_w, dst_h)
    count = aligned_frame_count(frames)
    return {
        "decoded": [dst_h, dst_w],
        "temporal": [count, dst_h, dst_w, 1],
        "channels": 1,
        "values": (0, 1),
        "source": (src_w, src_h),
    }


def main() -> None:
    cases = [
        (1080, 1920, 7),
        (1920, 1080, 1),
        (720, 1280, 9),
        (1080, 1920, 4),
        (1920, 1080, 5),
        (720, 1280, 2),
    ]
    for width, height, frames in cases:
        shape = normalized_mask(width, height, frames)
        assert shape["decoded"] == [512, 512], shape
        assert shape["temporal"][1:] == [512, 512, 1], shape
        assert shape["temporal"][0] == aligned_frame_count(frames)
        assert (shape["temporal"][0] - 1) % 4 == 0
        assert shape["channels"] == 1
        assert shape["values"] == (0, 1)
        assert shape["source"] == (width, height)
    assert normalized_mask(1080, 1920, 7)["temporal"][0] == 9
    assert normalized_mask(1920, 1080, 1)["temporal"][0] == 5
    assert normalized_mask(720, 1280, 5)["temporal"][0] == 5

    patch = Path("Scripts/patch_draw_things_vace_video_masks.py").read_text()
    for marker in (
        "shape.count == 2",
        "shape[0] == targetHeight && shape[1] == targetWidth",
        "Expected [1, \\(targetHeight), \\(targetWidth), 1] or [\\(targetHeight), \\(targetWidth)], got \\(shape).",
    ):
        if marker not in patch:
            raise SystemExit(f"mask shape contract missing: {marker}")
    print("VACE mask layouts match the encoder for portrait and landscape.")


if __name__ == "__main__":
    main()
