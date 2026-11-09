#!/usr/bin/env python3
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
support = (root / "FrameStageSupport.swift").read_text(encoding="utf-8")

required_policy = (
    "static let maxDepth = 8",
    "static let minDepth = 1",
    "private var targetDepth = maxDepth",
    "depth: Int = 8, adaptive: Bool = true",
    "os_proc_available_memory()",
    "thermal == .serious || thermal == .critical",
)
missing = [item for item in required_policy if item not in support]
if missing:
    raise SystemExit("Adaptive eight-frame cache policy is incomplete: " + "; ".join(missing))

expected_stages = {
    "RIFEVideoProcessor.swift",
    "RealCUGANPass.swift",
    "RestorationPass.swift",
    "FinalOutlinePass.swift",
    "AnimeRestorePass.swift",
}
pattern = re.compile(r"\bFrameAheadReader\s*\(([^)]*)\)")
found = {}
for path in root.glob("*.swift"):
    source = path.read_text(encoding="utf-8")
    matches = list(pattern.finditer(source))
    if not matches:
        continue
    found[path.name] = len(matches)
    for match in matches:
        args = match.group(1)
        if re.search(r"\bdepth\s*:", args):
            raise SystemExit(
                f"{path.name} overrides adaptive depth: FrameAheadReader({args})"
            )

missing_stages = expected_stages - set(found)
if missing_stages:
    raise SystemExit(
        "Expected video stages do not use FrameAheadReader: " + ", ".join(sorted(missing_stages))
    )

# Guard the working raster contract: 1080p should take one native 2x pass,
# yielding 3840x2160 rather than accidentally taking the 4x/8K branch.
preprocessor = (root / "Video720pPreprocessor.swift").read_text(encoding="utf-8")
processor = (root / "TwoPassVideoProcessor.swift").read_text(encoding="utf-8")
if "AVAssetExportPreset1920x1080" not in preprocessor:
    raise SystemExit("Video working raster is no longer normalized to 1080p")
if "if twoPixels <= 40_000_000 { return 1 }" not in processor:
    raise SystemExit("Automatic native 2x CUGAN pass rule changed unexpectedly")
source_width, source_height = 1920, 1080
native_pixels = (source_width * 2) * (source_height * 2)
passes = 2 if max(source_width, source_height) < 1080 and source_width * 4 * source_height * 4 <= 40_000_000 else (1 if native_pixels <= 40_000_000 else 0)
if passes != 1 or (source_width * 2 * passes, source_height * 2 * passes) != (3840, 2160):
    raise SystemExit("1080p no longer resolves to one native 2x pass and 3840x2160 output")

print(
    "ADAPTIVE_FRAME_CACHE_PASS: max=8, adaptive memory/thermal backoff, all video "
    "stages use default depth; 1080p -> one 2x CUGAN pass -> 3840x2160."
)
