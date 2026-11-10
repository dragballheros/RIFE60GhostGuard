#!/usr/bin/env python3
"""Check the fixed DJATOM x265 preset and, when given, a bitstream options string.

The params file is the x265_param_parse form for DJATOM 3.4+70. A few printed
SEI tokens use a different spelling than the parse key. Those mappings are
checked here so a token is never treated as optional.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = ROOT / "Scripts" / "x265_preset_params.txt"
HEADER = ROOT / "X265PresetParams.h"

# Tokens that must appear, space-delimited, in the x265 options SEI.
# range=0 is the printer form of range=limited. videoformat=5 / color*=2 are
# the printer forms of the "unknown" names. no-splice is hrd-concat=0.
# rc=crf is printed because crf= selects CRF. chromaloc=0 is the VUI flag
# left off; the chromaloc parse key is intentionally absent.
REQUIRED_TOKENS = [
    "ref=3",
    "no-allow-non-conformance",
    "no-repeat-headers",
    "no-aud",
    "no-hrd",
    "info",
    "hash=0",
    "no-temporal-layers",
    "open-gop",
    "min-keyint=23",
    "keyint=250",
    "gop-lookahead=0",
    "bframes=4",
    "b-adapt=2",
    "b-pyramid",
    "bframe-bias=0",
    "rc-lookahead=20",
    "lookahead-slices=6",
    "scenecut=40",
    "hist-scenecut=0",
    "radl=0",
    "no-splice",
    "no-intra-refresh",
    "ctu=64",
    "min-cu-size=8",
    "no-rect",
    "no-amp",
    "max-tu-size=32",
    "tu-inter-depth=1",
    "tu-intra-depth=1",
    "limit-tu=0",
    "rdoq-level=0",
    "dynamic-rd=0.00",
    "no-ssim-rd",
    "signhide",
    "no-tskip",
    "nr-intra=0",
    "nr-inter=0",
    "no-constrained-intra",
    "strong-intra-smoothing",
    "max-merge=3",
    "limit-refs=3",
    "no-limit-modes",
    "me=1",
    "subme=2",
    "merange=57",
    "temporal-mvp",
    "no-frame-dup",
    "no-hme",
    "weightp",
    "no-weightb",
    "deblock=0:0",
    "sao",
    "no-sao-non-deblock",
    "rd=3",
    "selective-sao=4",
    "early-skip",
    "rskip",
    "no-fast-intra",
    "no-tskip-fast",
    "no-cu-lossless",
    "b-intra",
    "no-splitrd-skip",
    "rdpenalty=0",
    "psy-rd=2.00",
    "psy-rdoq=0.00",
    "no-rd-refine",
    "no-lossless",
    "cbqpoffs=0",
    "crqpoffs=0",
    "rc=crf",
    "crf=20.0",
    "qcomp=0.60",
    "qpstep=4",
    "ipratio=1.40",
    "pbratio=1.30",
    "aq-mode=3",
    "aq-strength=1.00",
    "cutree",
    "qg-size=32",
    "no-rc-grain",
    "qpmax=69",
    "qpmin=0",
    "videoformat=5",
    "range=0",
    "colorprim=2",
    "transfer=2",
    "colormatrix=2",
    "chromaloc=0",
    "high-tier=1",
    "level-idc=0",
    "fps=24000/1001",
    "bitdepth=10",
    "input-res=1920x1080",
]

# Parse keys that must be present. Printer-only tokens are not keys.
REQUIRED_KEYS = {
    "ref": "3",
    "no-allow-non-conformance": "1",
    "no-repeat-headers": "1",
    "no-aud": "1",
    "no-hrd": "1",
    "info": "1",
    "hash": "0",
    "no-temporal-layers": "1",
    "open-gop": "1",
    "min-keyint": "23",
    "keyint": "250",
    "gop-lookahead": "0",
    "bframes": "4",
    "b-adapt": "2",
    "b-pyramid": "1",
    "bframe-bias": "0",
    "rc-lookahead": "20",
    "lookahead-slices": "6",
    "scenecut": "40",
    "hist-scenecut": "0",
    "radl": "0",
    "hrd-concat": "0",
    "no-intra-refresh": "1",
    "ctu": "64",
    "min-cu-size": "8",
    "no-rect": "1",
    "no-amp": "1",
    "max-tu-size": "32",
    "tu-inter-depth": "1",
    "tu-intra-depth": "1",
    "limit-tu": "0",
    "rdoq-level": "0",
    "dynamic-rd": "0.00",
    "no-ssim-rd": "1",
    "signhide": "1",
    "no-tskip": "1",
    "nr-intra": "0",
    "nr-inter": "0",
    "no-constrained-intra": "1",
    "strong-intra-smoothing": "1",
    "max-merge": "3",
    "limit-refs": "3",
    "no-limit-modes": "1",
    "me": "1",
    "subme": "2",
    "merange": "57",
    "temporal-mvp": "1",
    "no-frame-dup": "1",
    "no-hme": "1",
    "weightp": "1",
    "no-weightb": "1",
    "deblock": "0:0",
    "sao": "1",
    "no-sao-non-deblock": "1",
    "rd": "3",
    "selective-sao": "4",
    "early-skip": "1",
    "rskip": "1",
    "no-fast-intra": "1",
    "no-tskip-fast": "1",
    "no-cu-lossless": "1",
    "b-intra": "1",
    "no-splitrd-skip": "1",
    "rdpenalty": "0",
    "psy-rd": "2.00",
    "psy-rdoq": "0.00",
    "no-rd-refine": "1",
    "no-lossless": "1",
    "cbqpoffs": "0",
    "crqpoffs": "0",
    "crf": "20.0",
    "qcomp": "0.60",
    "qpstep": "4",
    "ipratio": "1.40",
    "pbratio": "1.30",
    "aq-mode": "3",
    "aq-strength": "1.00",
    "cutree": "1",
    "qg-size": "32",
    "no-rc-grain": "1",
    "qpmax": "69",
    "qpmin": "0",
    "videoformat": "unknown",
    "range": "limited",
    "colorprim": "unknown",
    "transfer": "unknown",
    "colormatrix": "unknown",
    "high-tier": "1",
    "level-idc": "0",
    "fps": "24000/1001",
}


def load_params(path: pathlib.Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"preset line is missing '=': {line}")
        key, value = line.split("=", 1)
        pairs.append((key, value))
    return pairs


def ffmpeg_param_string(pairs: list[tuple[str, str]]) -> str:
    parts = []
    for key, value in pairs:
        escaped = value.replace("\\", "\\\\").replace(":", "\\:")
        parts.append(f"{key}={escaped}")
    return ":".join(parts)


def c_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_header(pairs: list[tuple[str, str]], path: pathlib.Path) -> None:
    body = ffmpeg_param_string(pairs)
    path.write_text(
        "/* Generated from Scripts/x265_preset_params.txt. Do not edit. */\n"
        "#pragma once\n"
        f'static const char RIFE60_X265_PRESET_PARAMS[] = "{c_string(body)}";\n',
        encoding="utf-8",
    )


def options_tokens(text: str) -> list[str]:
    match = re.search(r"options:\s*(.*)", text)
    if not match:
        raise SystemExit("options string was not found")
    blob = match.group(1)
    blob = blob.split("\n", 1)[0]
    # Stop at the first non-printable, which ends the SEI payload.
    cleaned = []
    for ch in blob:
        if ord(ch) < 32 or ord(ch) > 126:
            break
        cleaned.append(ch)
    return "".join(cleaned).split()


def check_pairs(pairs: list[tuple[str, str]]) -> None:
    found = {key: value for key, value in pairs}
    if len(found) != len(pairs):
        raise SystemExit("duplicate preset key")
    missing = [key for key in REQUIRED_KEYS if key not in found]
    if missing:
        raise SystemExit("preset is missing keys: " + ", ".join(missing))
    mismatched = [
        f"{key}={found[key]} (expected {REQUIRED_KEYS[key]})"
        for key in REQUIRED_KEYS
        if found[key] != REQUIRED_KEYS[key]
    ]
    if mismatched:
        raise SystemExit("preset values differ: " + "; ".join(mismatched))
    if "chromaloc" in found or "rc" in found:
        raise SystemExit("chromaloc and rc must stay in their printer form, not as parse keys")
    forbidden = {"cpuid", "frame-threads", "numa-pools", "input-res", "total-frames", "input-csp", "log-level"}
    present = sorted(forbidden & set(found))
    if present:
        raise SystemExit("machine fields must not be pinned: " + ", ".join(present))


def check_options(text: str) -> None:
    tokens = set(options_tokens(text))
    missing = [token for token in REQUIRED_TOKENS if token not in tokens]
    if missing:
        raise SystemExit("options string is missing tokens: " + ", ".join(missing))
    if "chromaloc-top" in " ".join(tokens) or any(token.startswith("chromaloc-top") for token in tokens):
        raise SystemExit("chromaloc flag was turned on; the reference keeps chromaloc=0")
    print(f"options string matches {len(REQUIRED_TOKENS)} required tokens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-header", action="store_true")
    parser.add_argument("--options-file", type=pathlib.Path)
    args = parser.parse_args()
    pairs = load_params(PARAMS)
    check_pairs(pairs)
    if args.write_header:
        write_header(pairs, HEADER)
        print(f"wrote {HEADER}")
    else:
        expected = ffmpeg_param_string(pairs)
        header = HEADER.read_text(encoding="utf-8")
        if f'"{c_string(expected)}"' not in header:
            raise SystemExit("X265PresetParams.h does not match Scripts/x265_preset_params.txt")
        print(f"preset header matches {len(pairs)} parse keys")
    if args.options_file:
        check_options(args.options_file.read_text(encoding="utf-8", errors="surrogateescape"))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
