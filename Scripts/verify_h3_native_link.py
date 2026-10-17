#!/usr/bin/env python3
"""Fail the IPA workflow unless the pinned H3 engine is compiled and linked."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
products = root / "build" / "Build" / "Products" / "Release-iphoneos"
app = products / "RIFE60GhostGuard.app"
binary = app / "RIFE60GhostGuard"
objects = root / "build" / "Build" / "Intermediates.noindex" / "RIFE60GhostGuard.build" / "Release-iphoneos" / "RIFE60GhostGuard.build" / "Objects-normal" / "arm64"

required_objects = [
    "H3NativeRuntime.o",
    "h3.o",
    "h3_host.o",
    "h3_gpu.o",
    "h3_metal.o",
    "h3_tokenizer.o",
]
missing = [name for name in required_objects if not (objects / name).is_file()]
if missing:
    print("FAIL: H3 source files were not compiled into the iOS app target: " + ", ".join(missing), file=sys.stderr)
    print(f"Expected app object directory: {objects}", file=sys.stderr)
    sys.exit(1)

if not binary.is_file():
    raise SystemExit(f"FAIL: built app executable not found: {binary}")

nm = subprocess.run(
    ["xcrun", "nm", "-gU", str(binary)],
    check=False, capture_output=True, text=True,
)
if nm.returncode != 0:
    raise SystemExit("FAIL: unable to inspect app executable symbols with xcrun nm: " + nm.stderr[-2000:])
symbols = nm.stdout
required_symbols = ["_h3_load_dir", "_h3_generate", "_h3_free"]
missing_symbols = [symbol for symbol in required_symbols if symbol not in symbols]
if missing_symbols:
    raise SystemExit("FAIL: H3 native API symbols are not linked into the app executable: " + ", ".join(missing_symbols))

print("PASS: H3 Swift runtime and native C/Objective-C engine objects are compiled into the iOS app.")
print("PASS: h3_load_dir, h3_generate, and h3_free are present in the linked app executable.")
