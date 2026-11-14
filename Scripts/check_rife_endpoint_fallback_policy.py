#!/usr/bin/env python3
"""Fail CI if RIFE reintroduces an unclassified source-endpoint fallback."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
processor = (ROOT / "RIFEVideoProcessor.swift").read_text(encoding="utf-8")
guard = (ROOT / "GhostGuard.swift").read_text(encoding="utf-8")

forbidden = [
    r"chosen\\s*=\\s*t\\s*<\\s*0\\.5\\s*\\?\\s*prevPB\\s*:\\s*currentPB",
    r"chosen\\s*=\\s*prevPB\\b",
    r"chosen\\s*=\\s*currentPB\\b",
]
for pattern in forbidden:
    if re.search(pattern, processor):
        print(f"RIFE_ENDPOINT_POLICY_FAILED: forbidden direct endpoint selection: {pattern}", file=sys.stderr)
        sys.exit(1)

if processor.count("RIFEEndpointFallbackPolicy.endpointFrame(") < 2:
    print("RIFE_ENDPOINT_POLICY_FAILED: both recovery-failure paths must use the central endpoint policy", file=sys.stderr)
    sys.exit(1)

required_policy = [
    'case "scene cut / transition discontinuity", "stray frame at scene cut":',
    'case "isolated scene-end frame":',
    "default:",
    "return .retainInterpolation",
]
for marker in required_policy:
    if marker not in guard:
        print(f"RIFE_ENDPOINT_POLICY_FAILED: allowlist policy marker missing: {marker}", file=sys.stderr)
        sys.exit(1)

print("RIFE_ENDPOINT_POLICY_PASS: direct endpoint fallback removed; only explicit scene-boundary reasons are allowed")
