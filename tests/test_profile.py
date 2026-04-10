"""Tests for quantbench.profile — GGUF and safetensors parsing."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from quantbench._types import (
    DType,
    QuantbenchError,
    QuantFormat,
    TensorInfo,
)
from quantbench.profile import (
    _build_quant_profile,
    _group_tensors_into_layers,
    profile_from_dict,
    profile_gguf,
    profile_safetensors,
)


def _write_gguf(path: Path, tensors_info: list, metadata: dict | None = None) -> None:
    """Write a minimal GGUF v3 file for testing."""
    metadata = metadata or {}
    with open(path, "wb") as f:
        # Magic
        f.write(struct.pack("<I", 0x46475547))
        # Version 3
        f.write(struct.pack("<I", 3))
        # n_tensors
        f.write(struct.pack("<Q", len(tensors_info)))
        # n_kv
        f.write(struct.pack("<Q", len(metadata)))

        # Write metadata
        for key, value in metadata.items():
            # key string
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            # value type: string (8)
            f.write(struct.pack("<I", 8))
            val_bytes = str(value).encode("utf-8")
            f.write(struct.pack("<Q", len(val_bytes)))
            f.write(val_bytes)

        # Write tensor info
        data_offset = 0
        for name, shape, dtype_id in tensors_info:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", dtype_id))
            f.write(struct.pack("<Q", data_offset))
            # Calculate dummy data size
            n_elements = 1
            for dim in shape:
                n_elements *= dim
            data_offset += n_elements * 2  # dummy size


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Write a minimal safetensors file for testing."""
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        n_elements = 1
        for dim in shape:
            n_elements *= dim
        byte_size = n_elements * 2  # assume 2 bytes per element
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size

    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        # Write dummy tensor data
        f.write(b"\x00" * offset)


class TestProfileGGUF:
    def test_basic(self, tmp_path):
        path = tmp_path / "model.gguf"
        _write_gguf(path, [
            ("blk.0.attn_q.weight", [128, 128], 15),   # Q4_K_M = 15
            ("blk.0.attn_v.weight", [128, 128], 15),
            ("blk.1.ffn.weight", [128, 512], 14),       # Q4_K_S = 14
        ], metadata={"general.name": "test-model"})

        profile = profile_gguf(path)
        assert profile.name == "test-model"
        assert profile.format == QuantFormat.GGUF
        assert len(profile.tensors) == 3
        assert profile.total_params > 0

    def test_metadata(self, tmp_path):
        path = tmp_path / "model.gguf"
        _write_gguf(path, [("w", [10], 1)], metadata={"general.name": "my-model"})
        profile = profile_gguf(path)
        assert profile.metadata.get("general.name") == "my-model"

    def test_not_found(self, tmp_path):
        with pytest.raises(QuantbenchError):
            profile_gguf(tmp_path / "missing.gguf")

    def test_bad_magic(self, tmp_path):
        path = tmp_path / "bad.gguf"
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 0xDEADBEEF))
            f.write(b"\x00" * 100)
        with pytest.raises(QuantbenchError):
            profile_gguf(path)

    def test_layers_grouped(self, tmp_path):
        path = tmp_path / "model.gguf"
        _write_gguf(path, [
            ("blk.0.attn_q.weight", [64, 64], 1),
            ("blk.0.attn_v.weight", [64, 64], 1),
            ("blk.1.attn_q.weight", [64, 64], 1),
            ("output.weight", [64, 32], 1),
        ])
        profile = profile_gguf(path)
        layer_names = [l.name for l in profile.layers]
        assert any("blk.0" in n for n in layer_names)
        assert any("blk.1" in n for n in layer_names)


class TestProfileSafetensors:
    def test_basic(self, tmp_path):
        path = tmp_path / "model.safetensors"
        _write_safetensors(path, {
            "model.layers.0.self_attn.q_proj.weight": ("F16", [128, 128]),
            "model.layers.0.mlp.gate_proj.weight": ("F16", [128, 512]),
        })
        profile = profile_safetensors(path)
        assert profile.format == QuantFormat.SAFETENSORS
        assert len(profile.tensors) == 2
        assert profile.total_params > 0

    def test_not_found(self, tmp_path):
        with pytest.raises(QuantbenchError):
            profile_safetensors(tmp_path / "missing.safetensors")


class TestGroupTensors:
    def test_numbered_layers(self):
        tensors = [
            TensorInfo(name="blk.0.attn_q", shape=[10], dtype=DType.F16),
            TensorInfo(name="blk.0.attn_v", shape=[10], dtype=DType.F16),
            TensorInfo(name="blk.1.ffn", shape=[10], dtype=DType.F16),
        ]
        layers = _group_tensors_into_layers(tensors)
        names = {l.name for l in layers}
        assert "blk.0" in names
        assert "blk.1" in names

    def test_ungrouped(self):
        tensors = [TensorInfo(name="standalone", shape=[10], dtype=DType.F16)]
        layers = _group_tensors_into_layers(tensors)
        assert len(layers) == 1
        assert layers[0].name == "other"


class TestBuildQuantProfile:
    def test_all_fp16(self):
        tensors = [TensorInfo(name="w", shape=[100], dtype=DType.F16)]
        qp = _build_quant_profile(tensors, {})
        assert qp.avg_bits_per_weight == 16.0
        assert qp.n_full_precision_layers == 1
        assert qp.n_quantized_layers == 0

    def test_mixed(self):
        tensors = [
            TensorInfo(name="a", shape=[100], dtype=DType.F16),
            TensorInfo(name="b", shape=[100], dtype=DType.Q4_0),
        ]
        qp = _build_quant_profile(tensors, {})
        assert 4.0 < qp.avg_bits_per_weight < 16.0
        assert qp.n_quantized_layers == 1
        assert qp.n_full_precision_layers == 1

    def test_empty(self):
        qp = _build_quant_profile([], {})
        assert qp.avg_bits_per_weight == 0.0


class TestProfileFromDict:
    def test_basic(self):
        d = {
            "name": "test",
            "format": "gguf",
            "total_params": 1000,
            "total_size_bytes": 500,
            "quant": {"method": "ggml", "avg_bits_per_weight": 4.5},
        }
        p = profile_from_dict(d)
        assert p.name == "test"
        assert p.format == QuantFormat.GGUF
