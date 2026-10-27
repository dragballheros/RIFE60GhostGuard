#!/usr/bin/env python3
"""Apply the audited H3 iOS minimum-runtime guard to the pinned H3cApp source.

The app itself remains deployable to iOS 16.0. This H3 engine build refuses
initialization below iOS 18.0 because its selected MPSGraph SDPA call paths use
APIs introduced in iOS 18.0. Fail closed before Metal/graph initialization.
"""
from pathlib import Path
import sys

def patch_named_runtime(path: Path, source: str) -> None:
    """Optional guards for files that are not h3_gpu.m. Missing needles are left unchanged."""
    original = source
    if path.name == "h3_metal.m" and "device.architecture" in source:
        source = source.replace(
            "@available(macOS 14.0, *)",
            "@available(iOS 17.0, macOS 14.0, *)",
            1,
        )
        source = source.replace(
            "@available(macOS 26.0, *)",
            "@available(iOS 26.0, macOS 26.0, *)",
            1,
        )
    if path.name == "h3_dit.c":
        source = source.replace("pthread_t stream_thread;", "pthread_t stream_thread = 0;", 1)
        source = source.replace("pthread_t attn_thread;", "pthread_t attn_thread = 0;", 1)
        source = source.replace(
            "pthread_t matrix_threads[STREAM_MATRICES];",
            "pthread_t matrix_threads[STREAM_MATRICES] = {0};",
            1,
        )
    if source != original:
        path.write_text(source, encoding="utf-8")
        print(f"Applied optional H3 availability patches in {path.name}.")
    else:
        print(f"No optional H3 availability needles in {path.name}.")

if len(sys.argv) != 2:
    raise SystemExit("usage: patch_h3_ios_availability.py <h3_gpu.m>")

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
if path.name in {"h3_metal.m", "h3_dit.c"}:
    patch_named_runtime(path, source)
    raise SystemExit(0)
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
guard_present = replacement in source
if not guard_present:
    if source.count(needle) != 1:
        raise SystemExit(
            "Refusing to patch: expected exactly one h3_gpu_create entry point; "
            f"found {source.count(needle)}"
        )
    source = source.replace(needle, replacement, 1)

shader_needle = 'NSString *path = [NSString stringWithUTF8String:source_path];'
shader_replacement = """NSString *path = [NSString stringWithUTF8String:source_path];
        /*
         * On iOS the process working directory is not the app bundle. Resolve
         * the runtime's relative shader name against the bundled resource.
         */
        if (![path isAbsolutePath] &&
            ![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            NSString *bundledShader =
                [[NSBundle mainBundle] pathForResource:@"h3_shaders" ofType:@"metal"];
            if (bundledShader) path = bundledShader;
        }"""
shader_present = shader_replacement in source
if not shader_present:
    if source.count(shader_needle) != 1:
        raise SystemExit(
            "Refusing to patch: expected exactly one shader path resolver; "
            f"found {source.count(shader_needle)}"
        )
    source = source.replace(shader_needle, shader_replacement, 1)

path.write_text(source, encoding="utf-8")
print("H3 iOS runtime patches present: minimum iOS 18 guard and app-bundle shader resolution.")
