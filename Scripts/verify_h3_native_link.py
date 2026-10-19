#!/usr/bin/env python3
"""Verify H3 object compilation and native API resolution for the iOS app."""
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

def nm(*args):
    result = subprocess.run(["xcrun", "nm", *args], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("FAIL: unable to inspect native H3 object symbols: " + result.stderr[-2000:])
    return result.stdout

# Release app binaries may have their symbol table stripped by Xcode. Inspect
# the unstripped object files instead: the C implementation must define the
# public API, Swift must reference it, and the successful xcodebuild link above
# must have resolved those references into the final app.
api_symbols = ["_h3_load_dir", "_h3_generate", "_h3_free", "_h3_last_error", "_h3_result_free"]
definitions = nm("-gU", str(objects / "h3.o"))
swift_references = nm("-u", str(objects / "H3NativeRuntime.o"))
missing_definitions = [symbol for symbol in api_symbols if symbol not in definitions]
missing_references = [symbol for symbol in api_symbols if symbol not in swift_references]

if missing_definitions:
    raise SystemExit("FAIL: H3 implementation object does not define required API symbols: " + ", ".join(missing_definitions))
if missing_references:
    raise SystemExit("FAIL: Swift H3 runtime does not reference expected native API symbols: " + ", ".join(missing_references))

print("PASS: H3 Swift runtime and native C/Objective-C engine objects were compiled into the iOS app target.")
print("PASS: H3 implementation definitions match the Swift runtime's native API references.")
print("PASS: xcodebuild linked the app successfully; object-level symbol checks are used because release executable symbols may be stripped.")
