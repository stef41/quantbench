"""Tests for quantbench.matrix."""

import pytest

from quantbench.matrix import (
    KNOWN_FORMATS,
    ComparisonMatrix,
    FormatComparison,
    QuantFormatSpec,
    format_comparison_table,
    _accuracy_estimate,
    _speed_estimate,
    _model_size_gb,
)


# ── QuantFormatSpec ──────────────────────────────────────────────────────────


class TestQuantFormatSpec:
    def test_basic_creation(self):
        fmt = QuantFormatSpec(name="FP16", bits=16, description="half precision")
        assert fmt.name == "FP16"
        assert fmt.bits == 16
        assert fmt.block_size is None
        assert fmt.symmetric is True

    def test_bytes_per_weight_no_block(self):
        fp16 = QuantFormatSpec(name="FP16", bits=16)
        assert fp16.bytes_per_weight == 2.0

    def test_bytes_per_weight_block_symmetric(self):
        q4 = QuantFormatSpec(name="Q4", bits=4, block_size=32, symmetric=True)
        # 0.5 + 2/32 = 0.5625
        assert abs(q4.bytes_per_weight - 0.5625) < 1e-6

    def test_bytes_per_weight_block_asymmetric(self):
        fmt = QuantFormatSpec(name="A4", bits=4, block_size=128, symmetric=False)
        # 0.5 + 2/128 + 2/128 = 0.53125
        assert abs(fmt.bytes_per_weight - 0.53125) < 1e-6


# ── KNOWN_FORMATS ────────────────────────────────────────────────────────────


class TestKnownFormats:
    def test_all_nine_present(self):
        expected = {"FP16", "BF16", "INT8", "INT4", "GPTQ", "AWQ", "GGUF_Q4_0", "GGUF_Q5_1", "NF4"}
        assert set(KNOWN_FORMATS.keys()) == expected

    def test_formats_are_quantformatspec(self):
        for fmt in KNOWN_FORMATS.values():
            assert isinstance(fmt, QuantFormatSpec)


# ── FormatComparison ─────────────────────────────────────────────────────────


class TestFormatComparison:
    def test_dataclass_fields(self):
        comp = FormatComparison(
            format_a="A", format_b="B",
            size_ratio=0.5, accuracy_delta=-0.01, speed_ratio=1.5,
        )
        assert comp.format_a == "A"
        assert comp.size_ratio == 0.5


# ── ComparisonMatrix ─────────────────────────────────────────────────────────


class TestComparisonMatrix:
    def test_default_formats(self):
        m = ComparisonMatrix()
        assert len(m.formats) == len(KNOWN_FORMATS)

    def test_custom_formats(self):
        fmts = [QuantFormatSpec(name="X", bits=8), QuantFormatSpec(name="Y", bits=4)]
        m = ComparisonMatrix(formats=fmts)
        assert set(m.formats.keys()) == {"X", "Y"}

    def test_add_format(self):
        m = ComparisonMatrix(formats=[])
        m.add_format(QuantFormatSpec(name="Z", bits=3))
        assert "Z" in m.formats

    def test_compare_pair_identity(self):
        m = ComparisonMatrix()
        comp = m.compare_pair("FP16", "FP16")
        assert comp.size_ratio == 1.0
        assert comp.accuracy_delta == 0.0
        assert comp.speed_ratio == 1.0

    def test_compare_pair_int4_vs_fp16(self):
        m = ComparisonMatrix()
        comp = m.compare_pair("INT4", "FP16")
        # INT4 is much smaller than FP16
        assert comp.size_ratio < 0.5
        # FP16 is more accurate
        assert comp.accuracy_delta < 0
        # INT4 is faster
        assert comp.speed_ratio > 1.0

    def test_compare_pair_unknown_raises(self):
        m = ComparisonMatrix()
        with pytest.raises(KeyError):
            m.compare_pair("NONEXIST", "FP16")

    def test_full_matrix_shape(self):
        fmts = [QuantFormatSpec(name=f"F{i}", bits=i * 4) for i in range(1, 4)]
        m = ComparisonMatrix(formats=fmts)
        mat = m.full_matrix()
        assert len(mat) == 3
        assert all(len(row) == 3 for row in mat)

    def test_full_matrix_diagonal(self):
        m = ComparisonMatrix()
        mat = m.full_matrix()
        for row in mat:
            for comp in row:
                if comp.format_a == comp.format_b:
                    assert comp.size_ratio == 1.0

    def test_rank_by_size(self):
        m = ComparisonMatrix()
        ranked = m.rank_by("size")
        # Should be sorted ascending by bytes_per_weight
        for i in range(len(ranked) - 1):
            assert ranked[i].bytes_per_weight <= ranked[i + 1].bytes_per_weight

    def test_rank_by_speed(self):
        m = ComparisonMatrix()
        ranked = m.rank_by("speed")
        speeds = [_speed_estimate(f) for f in ranked]
        assert speeds == sorted(speeds, reverse=True)

    def test_rank_by_accuracy(self):
        m = ComparisonMatrix()
        ranked = m.rank_by("accuracy")
        accs = [_accuracy_estimate(f) for f in ranked]
        assert accs == sorted(accs, reverse=True)

    def test_rank_by_invalid_raises(self):
        m = ComparisonMatrix()
        with pytest.raises(ValueError):
            m.rank_by("nonexistent")

    def test_recommend_max_bits(self):
        m = ComparisonMatrix()
        recs = m.recommend({"max_bits": 4})
        assert all(f.bits <= 4 for f in recs)
        assert len(recs) > 0

    def test_recommend_min_accuracy(self):
        m = ComparisonMatrix()
        recs = m.recommend({"min_accuracy": 0.99})
        for f in recs:
            assert _accuracy_estimate(f) >= 0.99

    def test_recommend_empty_when_impossible(self):
        m = ComparisonMatrix()
        recs = m.recommend({"max_bits": 1})
        assert recs == []


# ── format_comparison_table ──────────────────────────────────────────────────


class TestFormatComparisonTable:
    def test_ascii_table_output(self):
        fmts = [
            QuantFormatSpec(name="FP16", bits=16),
            QuantFormatSpec(name="INT8", bits=8),
        ]
        m = ComparisonMatrix(formats=fmts)
        table = format_comparison_table(m.full_matrix())
        assert "FP16" in table
        assert "INT8" in table
        assert "---" in table

    def test_empty_matrix(self):
        assert format_comparison_table([]) == "(empty matrix)"


# ── Heuristic helpers ────────────────────────────────────────────────────────


class TestHeuristics:
    def test_accuracy_fp16(self):
        assert _accuracy_estimate(KNOWN_FORMATS["FP16"]) == 1.0

    def test_accuracy_int4_lower(self):
        assert _accuracy_estimate(KNOWN_FORMATS["INT4"]) < 1.0

    def test_speed_fp16_baseline(self):
        assert _speed_estimate(KNOWN_FORMATS["FP16"]) == 1.0

    def test_speed_int4_faster(self):
        assert _speed_estimate(KNOWN_FORMATS["INT4"]) > 1.0

    def test_model_size_fp16(self):
        size = _model_size_gb(KNOWN_FORMATS["FP16"], 7.0)
        assert abs(size - 7.0) < 1e-6

    def test_model_size_int8_half(self):
        size = _model_size_gb(KNOWN_FORMATS["INT8"], 7.0)
        assert abs(size - 3.5) < 1e-6
