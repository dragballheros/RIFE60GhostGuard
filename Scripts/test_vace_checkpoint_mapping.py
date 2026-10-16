#!/usr/bin/env python3
"""Fast unit tests for translating Diffusers Wan VACE keys into upstream Wan keys."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_vace_block_coreml import load_safetensors_block  # noqa: E402


class TinyStateModule:
    """Module-shaped test double exposing only the state_dict API used by the loader."""

    def __init__(self, values: dict[str, torch.Tensor]):
        self.values = {key: value.clone() for key, value in values.items()}
        self.loaded: dict[str, torch.Tensor] = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.values

    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True) -> Any:
        missing = sorted(set(self.values) - set(state))
        unexpected = sorted(set(state) - set(self.values))
        if strict and (missing or unexpected):
            raise ValueError(f"strict load failed: missing={missing}, unexpected={unexpected}")
        self.loaded = {key: value.clone() for key, value in state.items()}
        return SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)


def source_key(upstream_key: str) -> str:
    """Reverse the mapping under test to construct Diffusers-style fixture tensors."""
    name = upstream_key
    if name.startswith("self_attn."):
        name = "attn1." + name[len("self_attn."):]
    elif name.startswith("cross_attn."):
        name = "attn2." + name[len("cross_attn."):]
    elif name.startswith("ffn.0."):
        name = "ffn.net.0.proj." + name[len("ffn.0."):]
    elif name.startswith("ffn.2."):
        name = "ffn.net.2." + name[len("ffn.2."):]
    elif name.startswith("norm3."):
        name = "norm2." + name[len("norm3."):]
    elif name.startswith("before_proj."):
        name = "proj_in." + name[len("before_proj."):]
    elif name.startswith("after_proj."):
        name = "proj_out." + name[len("after_proj."):]
    elif name == "modulation":
        name = "scale_shift_table"

    if name.startswith(("attn1.", "attn2.")):
        for source, target in (
            ("o.", "to_out.0."),
            ("q.", "to_q."),
            ("k.", "to_k."),
            ("v.", "to_v."),
        ):
            marker = name.find(source)
            if marker >= 0:
                name = name[:marker] + target + name[marker + len(source):]
                break
    return "vace_blocks.0." + name


def test_complete_mapping_and_selective_tensor_load() -> None:
    values = {
        "before_proj.weight": torch.zeros(2, 2),
        "before_proj.bias": torch.zeros(2),
        "after_proj.weight": torch.zeros(2, 2),
        "after_proj.bias": torch.zeros(2),
        "self_attn.q.weight": torch.zeros(2, 2),
        "self_attn.q.bias": torch.zeros(2),
        "self_attn.k.weight": torch.zeros(2, 2),
        "self_attn.k.bias": torch.zeros(2),
        "self_attn.v.weight": torch.zeros(2, 2),
        "self_attn.v.bias": torch.zeros(2),
        "self_attn.o.weight": torch.zeros(2, 2),
        "self_attn.o.bias": torch.zeros(2),
        "self_attn.norm_q.weight": torch.zeros(2),
        "self_attn.norm_k.weight": torch.zeros(2),
        "cross_attn.q.weight": torch.zeros(2, 2),
        "cross_attn.q.bias": torch.zeros(2),
        "cross_attn.k.weight": torch.zeros(2, 2),
        "cross_attn.k.bias": torch.zeros(2),
        "cross_attn.v.weight": torch.zeros(2, 2),
        "cross_attn.v.bias": torch.zeros(2),
        "cross_attn.o.weight": torch.zeros(2, 2),
        "cross_attn.o.bias": torch.zeros(2),
        "cross_attn.norm_q.weight": torch.zeros(2),
        "cross_attn.norm_k.weight": torch.zeros(2),
        "norm3.weight": torch.zeros(2),
        "norm3.bias": torch.zeros(2),
        "norm3.weight_unused_check": torch.zeros(2),
        "ffn.0.weight": torch.zeros(4, 2),
        "ffn.0.bias": torch.zeros(4),
        "ffn.2.weight": torch.zeros(2, 4),
        "ffn.2.bias": torch.zeros(2),
        "modulation": torch.zeros(1, 6, 2),
    }
    # Remove the deliberately malformed key after asserting the fixture explicitly
    # includes all required mapping families; it is not a real upstream parameter.
    values.pop("norm3.weight_unused_check")
    source_values = {
        source_key(name): torch.full_like(value, fill_value=(index + 1) / 100.0)
        for index, (name, value) in enumerate(values.items())
    }
    # An unrelated block and an unrelated top-level tensor must remain untouched.
    source_values["vace_blocks.1.attn1.to_q.weight"] = torch.ones(2, 2)
    source_values["blocks.0.attn1.to_q.weight"] = torch.ones(2, 2)

    module = TinyStateModule(values)
    with tempfile.TemporaryDirectory(prefix="vace-key-map-") as temp_dir:
        checkpoint = Path(temp_dir) / "synthetic.safetensors"
        save_file(source_values, str(checkpoint))
        report = load_safetensors_block(module, checkpoint, "vace_blocks.0.")

    assert report["loaded_tensor_count"] == len(values), report
    assert set(module.loaded) == set(values)
    assert report["source_format"] == "Diffusers Wan2.1 VACE SafeTensors"
    assert report["loading_policy"].startswith("SafeTensors header indexed")
    assert "vace_blocks.1.attn1.to_q.weight" not in report["ignored_source_tensors"]
    assert all(torch.count_nonzero(value).item() > 0 for value in module.loaded.values())
    print(
        "PASS: mapped all",
        len(values),
        "synthetic VACE block tensors; unrelated checkpoint tensors were not loaded.",
    )


if __name__ == "__main__":
    test_complete_mapping_and_selective_tensor_load()
