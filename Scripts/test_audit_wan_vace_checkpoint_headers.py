"""Unit tests for Wan VACE checkpoint header audit arithmetic and grouping."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_wan_vace_checkpoint_headers as audit  # noqa: E402


class QuantizationEstimateTests(unittest.TestCase):
    def test_int8_payload_is_one_byte_per_element(self) -> None:
        self.assertEqual(audit.ideal_quantized_payload_bytes(123, 8), 123)

    def test_int4_payload_is_half_byte_per_element(self) -> None:
        self.assertEqual(audit.ideal_quantized_payload_bytes(123, 4), 61.5)

    def test_mixed_source_dtypes_use_element_count_not_source_bytes(self) -> None:
        # Two F32 elements occupy 8 source bytes; two BF16 occupy 4.
        # Four elements would require 2 ideal bytes at int4, not 3 source bytes.
        source_bytes = 2 * audit.DTYPE_BYTES["F32"] + 2 * audit.DTYPE_BYTES["BF16"]
        elements = 2 + 2
        self.assertEqual(source_bytes, 12)
        self.assertEqual(audit.ideal_quantized_payload_bytes(elements, 4), 2)

    def test_invalid_quantization_width_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit.ideal_quantized_payload_bytes(100, 0)

    def test_negative_element_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit.ideal_quantized_payload_bytes(-1, 4)


class CheckpointLayoutTests(unittest.TestCase):
    def test_tensor_bytes_use_dtype_and_shape(self) -> None:
        self.assertEqual(
            audit.tensor_bytes({"dtype": "F32", "shape": [2, 3]}),
            24,
        )
        self.assertEqual(
            audit.tensor_bytes({"dtype": "BF16", "shape": [2, 3]}),
            12,
        )

    def test_transformer_blocks_preserve_namespace_and_index(self) -> None:
        self.assertEqual(
            audit.layer_bucket("diffusion_model.blocks.7.attn.q.weight"),
            "diffusion_model_blocks_layer_007",
        )
        self.assertEqual(
            audit.layer_bucket("transformer_blocks.7.attn.q.weight"),
            "transformer_blocks_layer_007",
        )
        self.assertNotEqual(
            audit.layer_bucket("diffusion_model.blocks.7.attn.q.weight"),
            audit.layer_bucket("transformer_blocks.7.attn.q.weight"),
        )

    def test_named_modules_are_not_lumped_into_unknown_bucket(self) -> None:
        self.assertEqual(audit.layer_bucket("text_embedding.0.weight"), "module_text_embedding")
        self.assertEqual(audit.layer_bucket("vace_patch_embedding.weight"), "module_vace_patch_embedding")


if __name__ == "__main__":
    unittest.main()
