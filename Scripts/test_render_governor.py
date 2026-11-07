#!/usr/bin/env python3
"""Governor tier decisions must treat reclaimable pool bytes as headroom.

Safe still uses the raw 700 MB floor, thermal pressure, and memory warnings.
Performance versus Balanced uses raw plus reclaimable bytes.
"""
from pathlib import Path


def decide(raw, reclaimable, thermal, forced, previous):
    if forced:
        return "safe", "memory warning"
    if thermal:
        return "safe", "thermal"
    if raw < 700:
        return "safe", "raw floor"
    effective = raw + max(0, reclaimable)
    if effective < 1400:
        return "balanced", "effective below 1400"
    if previous == "safe" and effective < 1550:
        return "balanced", "hysteresis from safe"
    if previous == "balanced" and effective < 1650:
        return "balanced", "hysteresis"
    return "performance", "effective headroom"


def main():
    cases = [
        # Cached pools lift a 1500 MB reading back to Performance.
        ((1500, 200, False, False, "balanced"), "performance"),
        # Without reclaimable bytes, hysteresis keeps Balanced under 1650.
        ((1500, 0, False, False, "balanced"), "balanced"),
        # The raw floor ignores reclaimable bytes.
        ((600, 2000, False, False, "performance"), "safe"),
        ((3000, 0, True, False, "performance"), "safe"),
        ((3000, 0, False, True, "performance"), "safe"),
        ((1300, 200, False, False, "performance"), "performance"),
        ((1300, 0, False, False, "performance"), "balanced"),
        ((1600, 0, False, False, "safe"), "performance"),
        ((1500, 0, False, False, "safe"), "balanced"),
    ]
    for args, expected in cases:
        tier, _ = decide(*args)
        if tier != expected:
            raise SystemExit(f"{args} -> {tier}, expected {expected}")

    source = Path("PerformanceTelemetry.swift").read_text()
    for marker in (
        "rawMB + max(0, reclaimableMB)",
        "if rawMB < 700",
        "effective below 1400",
        "hysteresis",
    ):
        if marker not in source:
            raise SystemExit(f"governor source missing {marker}")
    if "Task.sleep(nanoseconds: 3_000_000)" in source:
        raise SystemExit("Balanced still sleeps 3 ms per frame")
    print("Governor tier decisions account for reclaimable bytes.")


if __name__ == "__main__":
    main()
