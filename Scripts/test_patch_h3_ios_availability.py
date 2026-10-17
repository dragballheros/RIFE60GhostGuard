#!/usr/bin/env python3
"""Regression tests for the pinned H3 iOS availability patcher."""
from pathlib import Path
import subprocess
import sys
import tempfile

PATCHER = Path(__file__).with_name("patch_h3_ios_availability.py")
ENTRY = """h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    @autoreleasepool {
"""
GUARD = "if (@available(iOS 18.0, *))"


def run(path: Path, expect_success: bool) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(PATCHER), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == expect_success, (
        f"unexpected patcher exit {result.returncode}: {result.stdout}{result.stderr}"
    )
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)

        # Expected entry point gets one fail-closed guard.
        source = root / "expected.m"
        source.write_text("prefix\n" + ENTRY + "        // body\n    }\n}\n", encoding="utf-8")
        run(source, True)
        patched = source.read_text(encoding="utf-8")
        assert patched.count(GUARD) == 1, "guard was not inserted exactly once"
        assert "MiniMax H3 requires iOS 18.0 or newer" in patched
        assert patched.index(GUARD) < patched.index("// body"), "guard must precede engine body"

        # Re-running is idempotent and must not duplicate the guard.
        run(source, True)
        assert source.read_text(encoding="utf-8").count(GUARD) == 1

        # Ambiguous or changed source must fail closed without mutation.
        ambiguous = root / "ambiguous.m"
        ambiguous.write_text("no matching function here\n", encoding="utf-8")
        before = ambiguous.read_text(encoding="utf-8")
        result = run(ambiguous, False)
        assert ambiguous.read_text(encoding="utf-8") == before, "failed patch mutated source"
        assert "found 0" in result.stderr

        duplicate = root / "duplicate.m"
        duplicate.write_text(ENTRY + "\n" + ENTRY, encoding="utf-8")
        before = duplicate.read_text(encoding="utf-8")
        result = run(duplicate, False)
        assert duplicate.read_text(encoding="utf-8") == before, "ambiguous patch mutated source"
        assert "found 2" in result.stderr

    print("H3 iOS availability patcher tests passed (insert, idempotence, fail-closed ambiguity).")


if __name__ == "__main__":
    main()
