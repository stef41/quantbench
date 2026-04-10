"""Profile quantized models from GGUF, safetensors, or dict metadata."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

from quantbench._types import (
    DType,
    LayerInfo,
    ModelProfile,
    QuantFormat,
    QuantMethod,
    QuantProfile,
    QuantbenchError,
    TensorInfo,
)

# ── GGUF constants ──
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_DTYPE_MAP = {
    0: DType.F32, 1: DType.F16, 2: DType.Q4_0, 3: DType.Q4_1,
    6: DType.Q5_0, 7: DType.Q5_1, 8: DType.Q8_0,
    10: DType.Q2_K, 11: DType.Q3_K_S, 12: DType.Q3_K_M,
    13: DType.Q3_K_L, 14: DType.Q4_K_S, 15: DType.Q4_K_M,
    16: DType.Q5_K_S, 17: DType.Q5_K_M, 18: DType.Q6_K,
    19: DType.IQ2_XXS, 20: DType.IQ3_XXS, 26: DType.IQ4_XS,
    24: DType.IQ1_S, 28: DType.BF16,
}

# Type tags for GGUF metadata values
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# safetensors dtype sizes
_SAFETENSORS_DTYPE_BPW = {
    "F32": 32, "F16": 16, "BF16": 16, "I64": 64, "I32": 32,
    "I16": 16, "I8": 8, "U8": 8, "BOOL": 1, "F8_E4M3": 8, "F8_E5M2": 8,
}
_SAFETENSORS_DTYPE_MAP = {
    "F32": DType.F32, "F16": DType.F16, "BF16": DType.BF16,
    "I8": DType.Q8_0, "U8": DType.Q8_0,
}


def _read_gguf_string(f: BinaryIO) -> str:
    """Read a GGUF string (uint64 length + bytes)."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f: BinaryIO, vtype: int) -> Any:
    """Read a single GGUF metadata value."""
    if vtype == _GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == _GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == _GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == _GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == _GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == _GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == _GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == _GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif vtype == _GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    elif vtype == _GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == _GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == _GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif vtype == _GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    else:
        raise QuantbenchError(f"Unknown GGUF value type: {vtype}")


def profile_gguf(path: str | Path) -> ModelProfile:
    """Parse a GGUF file and return a model profile.

    This is a pure-Python parser — no external dependencies required.
    Reads only the header (metadata + tensor info), not the actual weights.
    """
    path = Path(path)
    if not path.exists():
        raise QuantbenchError(f"File not found: {path}")

    with open(path, "rb") as f:
        # Read magic number
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise QuantbenchError(f"Not a GGUF file (magic: {magic:#x})")

        # Read version
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise QuantbenchError(f"Unsupported GGUF version: {version}")

        # Read counts
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Read metadata
        metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = _read_gguf_value(f, vtype)
            metadata[key] = value

        # Read tensor info
        tensors: List[TensorInfo] = []
        for _ in range(n_tensors):
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype_id = struct.unpack("<I", f.read(4))[0]
            _offset = struct.unpack("<Q", f.read(8))[0]  # tensor data offset

            dtype = GGUF_DTYPE_MAP.get(dtype_id, DType.UNKNOWN)
            tensors.append(TensorInfo(name=name, shape=shape, dtype=dtype))

    # Build layers from tensor names
    layers = _group_tensors_into_layers(tensors)

    # Build quant profile
    quant = _build_quant_profile(tensors, metadata)

    total_params = sum(t.n_elements for t in tensors)
    total_size = sum(t.size_bytes for t in tensors)

    model_name = metadata.get("general.name", path.stem)
    if not isinstance(model_name, str):
        model_name = str(model_name)

    return ModelProfile(
        name=model_name,
        format=QuantFormat.GGUF,
        total_params=total_params,
        total_size_bytes=total_size,
        tensors=tensors,
        layers=layers,
        quant=quant,
        metadata={k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))},
    )


def profile_safetensors(path: str | Path) -> ModelProfile:
    """Parse a safetensors file header and return a model profile.

    Only reads the JSON header — does not load weights into memory.
    """
    path = Path(path)
    if not path.exists():
        raise QuantbenchError(f"File not found: {path}")

    with open(path, "rb") as f:
        # First 8 bytes: header size as uint64
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size > 100_000_000:  # sanity check: 100MB max header
            raise QuantbenchError(f"Header too large: {header_size}")

        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)

    # Extract tensor metadata (skip __metadata__ key)
    tensors: List[TensorInfo] = []
    meta = header.pop("__metadata__", {})

    for tensor_name, info in header.items():
        dtype_str = info.get("dtype", "F32")
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [0, 0])

        dtype = _SAFETENSORS_DTYPE_MAP.get(dtype_str, DType.UNKNOWN)
        size_bytes = offsets[1] - offsets[0] if len(offsets) == 2 else 0

        ti = TensorInfo(name=tensor_name, shape=shape, dtype=dtype, size_bytes=size_bytes)
        tensors.append(ti)

    layers = _group_tensors_into_layers(tensors)
    quant = _build_quant_profile(tensors, meta)

    total_params = sum(t.n_elements for t in tensors)
    total_size = sum(t.size_bytes for t in tensors)

    return ModelProfile(
        name=path.stem,
        format=QuantFormat.SAFETENSORS,
        total_params=total_params,
        total_size_bytes=total_size,
        tensors=tensors,
        layers=layers,
        quant=quant,
        metadata=meta if isinstance(meta, dict) else {},
    )


def profile_from_dict(data: Dict[str, Any]) -> ModelProfile:
    """Create a ModelProfile from a dictionary (e.g. loaded from JSON)."""
    return ModelProfile.from_dict(data)


def _group_tensors_into_layers(tensors: List[TensorInfo]) -> List[LayerInfo]:
    """Group tensors into logical layers based on naming conventions."""
    layer_map: Dict[str, List[TensorInfo]] = {}

    for t in tensors:
        # Extract layer identifier from tensor name
        # Common patterns: "blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"
        parts = t.name.split(".")
        layer_name = "other"

        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:
                layer_name = ".".join(parts[: i + 1])
                break
            if part in ("blk", "layers", "block", "h"):
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_name = ".".join(parts[: i + 2])
                    break

        if layer_name not in layer_map:
            layer_map[layer_name] = []
        layer_map[layer_name].append(t)

    return [LayerInfo(name=name, tensors=ts) for name, ts in sorted(layer_map.items())]


def _build_quant_profile(tensors: List[TensorInfo], metadata: Dict[str, Any]) -> QuantProfile:
    """Build quantization profile from tensor list and metadata."""
    if not tensors:
        return QuantProfile()

    # Count dtype distribution
    total_elements = sum(t.n_elements for t in tensors)
    dtype_counts: Dict[str, int] = {}
    for t in tensors:
        key = t.dtype.value
        dtype_counts[key] = dtype_counts.get(key, 0) + t.n_elements

    dtype_dist = {}
    if total_elements > 0:
        dtype_dist = {k: round(v / total_elements, 4) for k, v in dtype_counts.items()}

    # Compute average bits per weight
    avg_bpw = 0.0
    if total_elements > 0:
        avg_bpw = sum(t.n_elements * t.bits_per_weight for t in tensors) / total_elements

    # Count quantized vs full precision layers
    fp_dtypes = {DType.F32, DType.F16, DType.BF16}
    n_quant = sum(1 for t in tensors if t.dtype not in fp_dtypes)
    n_fp = sum(1 for t in tensors if t.dtype in fp_dtypes)

    # Detect method from metadata
    method = QuantMethod.UNKNOWN
    meta_str = str(metadata).lower()
    if "gptq" in meta_str:
        method = QuantMethod.GPTQ
    elif "awq" in meta_str:
        method = QuantMethod.AWQ
    elif "ggml" in meta_str or "llama.cpp" in meta_str:
        method = QuantMethod.GGML
    elif "bitsandbytes" in meta_str:
        method = QuantMethod.BITSANDBYTES
    elif avg_bpw > 15:
        method = QuantMethod.FP16
    elif avg_bpw > 7:
        method = QuantMethod.INT8
    elif avg_bpw > 3:
        method = QuantMethod.INT4

    return QuantProfile(
        method=method,
        avg_bits_per_weight=round(avg_bpw, 2),
        dtype_distribution=dtype_dist,
        n_quantized_layers=n_quant,
        n_full_precision_layers=n_fp,
    )
