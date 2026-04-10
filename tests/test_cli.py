"""Tests for quantbench.cli."""

import struct
from pathlib import Path

import pytest
from click.testing import CliRunner

from quantbench.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def gguf_file(tmp_path):
    """Create a minimal GGUF file for testing."""
    path = tmp_path / "model.gguf"
    with open(path, "wb") as f:
        # Magic GGUF
        f.write(struct.pack("<I", 0x46475547))
        # Version 3
        f.write(struct.pack("<I", 3))
        # n_tensors = 2
        f.write(struct.pack("<Q", 2))
        # n_kv = 1
        f.write(struct.pack("<Q", 1))

        # Metadata: general.name = "test"
        key = b"general.name"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 8))  # string type
        val = b"test-model"
        f.write(struct.pack("<Q", len(val)))
        f.write(val)

        # Tensor 1
        name1 = b"blk.0.attn.weight"
        f.write(struct.pack("<Q", len(name1)))
        f.write(name1)
        f.write(struct.pack("<I", 2))  # n_dims
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<I", 15))  # Q4_K_M
        f.write(struct.pack("<Q", 0))   # offset

        # Tensor 2
        name2 = b"blk.0.ffn.weight"
        f.write(struct.pack("<Q", len(name2)))
        f.write(name2)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<Q", 256))
        f.write(struct.pack("<I", 15))
        f.write(struct.pack("<Q", 4096))

    return str(path)


class TestProfileCommand:
    def test_basic(self, runner, gguf_file):
        result = runner.invoke(cli, ["profile", gguf_file])
        assert result.exit_code == 0

    def test_markdown(self, runner, gguf_file):
        result = runner.invoke(cli, ["profile", gguf_file, "--markdown"])
        assert result.exit_code == 0
        assert "#" in result.output

    def test_json_output(self, runner, gguf_file, tmp_path):
        out = str(tmp_path / "report.json")
        result = runner.invoke(cli, ["profile", gguf_file, "-o", out])
        assert result.exit_code == 0
        assert Path(out).exists()


class TestCompareCommand:
    def test_basic(self, runner, gguf_file):
        result = runner.invoke(cli, ["compare", gguf_file, gguf_file])
        assert result.exit_code == 0


class TestLayersCommand:
    def test_basic(self, runner, gguf_file):
        result = runner.invoke(cli, ["layers", gguf_file])
        assert result.exit_code == 0


class TestRecommendCommand:
    def test_basic(self, runner, gguf_file):
        result = runner.invoke(cli, ["recommend", gguf_file])
        assert result.exit_code == 0
