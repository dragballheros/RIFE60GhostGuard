#!/usr/bin/env python3
"""Regression tests for Scripts/inspect_vace_checkpoint.py."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("inspect_vace_checkpoint.py")


class InspectVaceCheckpointTests(unittest.TestCase):
    def test_reports_hashes_sizes_and_valid_index_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text('{"model_type":"VaceWanModel"}', encoding="utf-8")
            (root / "weights.bin").write_bytes(b"not a real model; must only be hashed")
            (root / "diffusion.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}),
                encoding="utf-8",
            )
            output = root / "report.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root), "--json", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["file_count"], 3)
            self.assertEqual(report["weight_file_count"], 1)
            self.assertEqual(report["total_size_bytes"] > 0, True)
            index = next(item for item in report["files"] if item["path"].endswith(".index.json"))
            self.assertEqual(index["index_tensor_count"], 2)
            self.assertEqual(index["shard_count"], 2)
            weight = next(item for item in report["files"] if item["path"] == "weights.bin")
            self.assertIn("not deserialized", weight["format_note"])

    def test_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(missing)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
