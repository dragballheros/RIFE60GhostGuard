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
DOWNLOADER = (ROOT / "H3ModelDownloader.swift").read_text(encoding="utf-8")
PROJECT = (ROOT / "project.yml").read_text(encoding="utf-8")
BRIDGE = (ROOT / "RIFE60GhostGuard-Bridging-Header.h").read_text(encoding="utf-8")
HEADER_PATH = ROOT / "Vendor" / "minimax-h3" / "h3.h"
NATIVE_PATH = ROOT / "Vendor" / "minimax-h3" / "h3_gpu.m"

checks = [
    ("Swift string interpolation is not double-escaped", "\\\\(" not in UI),
    ("H3 and Wan are separate visible backend choices", "case wanVACE" in UI and "case minimaxH3" in UI and "ForEach(LocalVideoBackend.allCases" in UI),
    ("model selection uses checkbox-style indicators", 'Image(systemName: selectedBackend == backend ? "checkmark.circle.fill" : "circle")' in UI),
    ("no dropdown picker in the editor", not re.search(r"\bPicker\s*\(", UI)),
    ("Wan VACE masked workflow remains wired", "DrawThingsLocalVACEReconstructionEngine()" in UI and "VACESelectiveVideoPipeline(" in UI and "loadMask(url:" in UI),
    ("H3 selection dispatches to its own reference workflow", "await runH3ReferenceEdit()" in UI and "generateReferenceConditionedVideo(" in UI),
    ("H3 workflow does not claim VACE mask semantics", "does not use the VACE mask" in UI and "does not claim pixel-mask preservation" in UI),
    ("H3 validates Ref2VA video duration and playable track before native loading", "Official Ref2VA accepts reference clips of 2–15 seconds" in UI and "(2.0...15.0).contains(referenceDuration.seconds)" in UI and "loadTracks(withMediaType: .video)" in UI),
    ("H3 model folder is selected locally", '.fileImporter(isPresented: $isChoosingH3Model, allowedContentTypes: [.folder]' in UI),
    ("H3 model downloader is pinned to the official Ref2VA revision and preserves the repository root layout", "42ed227ee7df40d41602854ae760620d6eb651fe" in DOWNLOADER and "tree/\\(revision)?recursive=true&expand=true" in DOWNLOADER and '$0.path.hasPrefix("Ref2VA/")' in DOWNLOADER),
    ("H3 model downloads resume and verify file checksums", 'forHTTPHeaderField: "Range"' in DOWNLOADER and "SHA256()" in DOWNLOADER and "Insecure.SHA1()" in DOWNLOADER),
    ("H3 model download requires explicit territorial license authorization confirmation", "I have verified that my use is authorized by the license" in UI and "disabled(!acceptedH3License" in UI and "United States, European Union, United Kingdom, and South Korea" in UI and "MiniMax has separately authorized you" in UI),
    ("H3 generation fails closed on unsupported iOS", "requires iOS 18.0 or newer" in PREFLIGHT and "guard preflight.isOperatingSystemSupported" in RUNTIME),
    ("H3 has an available-memory floor before native loading", "minimumAvailableMemoryBytes: UInt64 = 2 * 1024 * 1024 * 1024" in PREFLIGHT and "guard preflight.hasMinimumMemory" in RUNTIME and "case insufficientMemory(String)" in RUNTIME),
    ("H3 rejects incomplete or incompatible model directory layouts before native loading", "modelLayoutIsValid" in PREFLIGHT and "Ref2VA/model_index.json" in PREFLIGHT and "Ref2VA/transformer" in PREFLIGHT and "Ref2VA/video_vae" in PREFLIGHT and "guard preflight.modelLayoutIsValid" in RUNTIME and "partial download or a GGUF-only file is not compatible" in PREFLIGHT),
    ("H3 uses the video reference C API", "reference.kind = H3_REFERENCE_VIDEO" in RUNTIME and "params.reference_count = 1" in RUNTIME),
    ("H3 uses disk-streaming mode", "params.ssd_streaming = 1" in RUNTIME),
    ("native H3 progress callback is wired", "params.on_progress = h3NativeProgressCallback" in RUNTIME and "onProgress: { phase, fraction" in UI),
    ("cancellation error is handled exhaustively", "case .cancelled:" in RUNTIME and "MiniMax H3 generation was cancelled." in RUNTIME),
    ("user cancellation reaches the native progress callback", "H3CancellationToken" in RUNTIME and "Cancel H3 generation" in UI and "return context.cancellationToken?.isCancelled == true ? 1 : 0" in RUNTIME),
    ("H3 output must exist and be non-empty", "outputSize > 0" in UI),
    ("H3 export is checked for a playable video track and duration", "loadTracks(withMediaType: .video)" in UI and "exportedDuration.seconds > 0" in UI),
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
    ("resumable, checksum-verified H3 model asset installer rejects unsafe paths", 'forHTTPHeaderField: "Range"' in DOWNLOADER and "SHA256()" in DOWNLOADER and "Insecure.SHA1()" in DOWNLOADER and "42ed227ee7df40d41602854ae760620d6eb651fe" in DOWNLOADER and '$0 != ".."' in DOWNLOADER and 'contains("\\\\")' in DOWNLOADER),
    ("native progress and user cancellation connected to the editor", "params.on_progress = h3NativeProgressCallback" in RUNTIME and "Cancel H3 generation" in UI),
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
