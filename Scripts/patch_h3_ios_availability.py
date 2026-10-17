#!/usr/bin/env python3
"""Apply the audited H3 iOS minimum-runtime guard to the pinned H3cApp source.

The app itself remains deployable to iOS 16.0. This H3 engine build refuses
initialization below iOS 18.0 because its selected MPSGraph SDPA call paths use
APIs introduced in iOS 18.0. Fail closed before Metal/graph initialization.
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_h3_ios_availability.py <h3_gpu.m>")

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
needle = """h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    @autoreleasepool {
"""
replacement = """h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    @autoreleasepool {
        /*
         * H3's selected MPSGraph SDPA paths require iOS 18.0. Keep the host
         * app's deployment target independent, but reject H3 initialization
         * before touching Metal or creating any graph on older iOS releases.
         */
        if (@available(iOS 18.0, *)) {
            /* Supported H3 runtime OS. */
        } else {
            if (error && error_size) {
                snprintf(error, error_size,
                         "MiniMax H3 requires iOS 18.0 or newer");
            }
            return NULL;
        }
"""
if replacement in source:
    print("H3 iOS 18.0 initialization guard already applied.")
    raise SystemExit(0)
if source.count(needle) != 1:
    raise SystemExit(
        "Refusing to patch: expected exactly one h3_gpu_create entry point; "
        f"found {source.count(needle)}"
    )
path.write_text(source.replace(needle, replacement), encoding="utf-8")
print("Applied fail-closed iOS 18.0 H3 engine initialization guard.")
