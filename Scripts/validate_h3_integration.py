#!/usr/bin/env python3
"""Static contract validator for the local MiniMax H3 editor integration.

This checks repository wiring, not actual model inference. A passing result is
not a substitute for a device run with licensed, compatible H3 model weights.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "VACEVideoEditView.swift").read_text(encoding="utf-8")
RUNTIME = (ROOT / "H3NativeRuntime.swift").read_text(encoding="utf-8")
PREFLIGHT = (ROOT / "H3RuntimePreflight.swift").read_text(encoding="utf-8")
PROJECT = (ROOT / "project.yml").read_text(encoding="utf-8")
BRIDGE = (ROOT / "RIFE60GhostGuard-Bridging-Header.h").read_text(encoding="utf-8")
HEADER_PATH = ROOT / "Vendor" / "minimax-h3" / "h3.h"
NATIVE_PATH = ROOT / "Vendor" / "minimax-h3" / "h3_gpu.m"

checks = [
    ("H3 and Wan are separate visible backend choices", "case wanVACE" in UI and "case minimaxH3" in UI and "ForEach(LocalVideoBackend.allCases" in UI),
    ("model selection uses checkbox-style indicators", 'Image(systemName: selectedBackend == backend ? "checkmark.circle.fill" : "circle")' in UI),
    ("no dropdown picker in the editor", not re.search(r"\bPicker\s*\(", UI)),
    ("Wan VACE masked workflow remains wired", "DrawThingsLocalVACEReconstructionEngine()" in UI and "VACESelectiveVideoPipeline(" in UI and "loadMask(url:" in UI),
    ("H3 selection dispatches to its own reference workflow", "await runH3ReferenceEdit()" in UI and "generateReferenceConditionedVideo(" in UI),
    ("H3 workflow does not claim VACE mask semantics", "does not use the VACE mask" in UI and "does not claim pixel-mask preservation" in UI),
    ("H3 model folder is selected locally", '.fileImporter(isPresented: $isChoosingH3Model, allowedContentTypes: [.folder]' in UI),
    ("H3 generation fails closed on unsupported iOS", "requires iOS 18.0 or newer" in PREFLIGHT and "guard preflight.isOperatingSystemSupported" in RUNTIME),
    ("H3 uses the video reference C API", "reference.kind = H3_REFERENCE_VIDEO" in RUNTIME and "params.reference_count = 1" in RUNTIME),
    ("H3 uses disk-streaming mode", "params.ssd_streaming = 1" in RUNTIME),
    ("H3 output must exist and be non-empty", "outputSize > 0" in UI),
    ("no remote inference client in H3 UI/runtime", "URLSession" not in UI + RUNTIME and "https://" not in UI + RUNTIME),
    ("native H3 header is bundled into the app target", "Vendor/minimax-h3" in PROJECT and '#include "h3.h"' in BRIDGE and HEADER_PATH.is_file()),
    ("native iOS entry point has an OS availability guard", NATIVE_PATH.is_file() and "@available(iOS 18.0, *)" in NATIVE_PATH.read_text(encoding="utf-8")),
]

failed = []
for label, passed in checks:
    print(f"{'PASS' if passed else 'FAIL'}: {label}")
    if not passed:
        failed.append(label)

patch_tests = ROOT / "Scripts" / "test_patch_h3_ios_availability.py"
if patch_tests.is_file():
    result = subprocess.run([sys.executable, str(patch_tests)], check=False)
    if result.returncode:
        failed.append("H3 native availability patcher regression tests")
        print("FAIL: H3 native availability patcher regression tests")
    else:
        print("PASS: H3 native availability patcher regression tests")
else:
    failed.append("H3 native availability patcher regression tests are missing")
    print("FAIL: H3 native availability patcher regression tests are missing")

# These gates need runtime/model evidence and are deliberately not represented
# as passed merely because static code inspection succeeds.
release_gates = [
    ("resumable, checksum-verified H3 model asset installer", False),
    ("native progress and user cancellation connected to the editor", "params.on_progress = nil" not in RUNTIME and "cancel" in UI.lower()),
    ("successful offline reference-conditioned inference on iPhone 14 Pro Max", False),
    ("playable exported video verified on the target iPhone", False),
    ("peak memory / thermal / watchdog measurements on the target iPhone", False),
]
print("\nDevice/runtime release gates:")
for label, passed in release_gates:
    print(f"{'PASS' if passed else 'BLOCKED'}: {label}")

if failed:
    print(f"\nIntegration validation failed: {len(failed)} check(s).")
    sys.exit(1)
print(f"\nStatic integration validation passed: {len(checks) + 1} checks.")
print("Release status remains BLOCKED until every device/runtime release gate above is validated.")
if "--strict-release" in sys.argv:
    sys.exit(2 if not all(passed for _, passed in release_gates) else 0)
