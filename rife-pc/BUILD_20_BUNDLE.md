# Windows Build 20 Bundle

This build is the single cumulative Windows desktop snapshot containing the performance work from builds 16 through 19.

Included:
- AMD AMF HEVC Main10 hardware encoding with libx265 fallback
- AMF encoder capability detection
- Real-CUGAN parallel worker tuning
- RIFE Vulkan VRAM detection and bounded worker pool
- RIFE/encode stage progress separation
- All Windows performance changes from the preceding stable build

Source bundle lineage:
- Build 16: AMD AMF encoder groundwork
- Build 17: AMF detection and integration
- Build 18: Real-CUGAN worker parallelism
- Build 19: encode-stage progress reporting

This file is intentionally non-runtime and exists to mark the consolidated Build 20 snapshot.
