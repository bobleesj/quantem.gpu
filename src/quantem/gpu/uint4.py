"""Packed unsigned 4-bit detector-count storage.

``uint4`` is a storage contract, not a NumPy dtype. Values are detector counts
that must fit in ``0..15`` and are packed two pixels per byte, low nibble first.
Backends use the logical ``shape`` for indexing and the physical ``buffer`` for
device-resident kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np


@dataclass
class PackedUInt4Array:
    """Logical array backed by two-counts-per-byte unsigned 4-bit storage."""

    buffer: Any
    shape: tuple[int, ...]
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    _is_packed_uint4: bool = True

    @property
    def dtype(self) -> str:
        return "uint4"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(prod(self.shape))

    @property
    def nbytes(self) -> int:
        return int(getattr(self.buffer, "nbytes", (self.size + 1) // 2))

    @property
    def packed_size(self) -> int:
        return (self.size + 1) // 2

    @property
    def device(self) -> Any:
        return getattr(self.buffer, "device", self.backend)

    def reshape(self, *shape: int | tuple[int, ...]) -> "PackedUInt4Array":
        """Return a logical reshaped view of the same packed buffer."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        values = [int(v) for v in shape]
        inferred = [i for i, value in enumerate(values) if value == -1]
        if len(inferred) > 1:
            raise ValueError("can only specify one unknown packed uint4 dimension")
        if inferred:
            known = prod(value for value in values if value != -1)
            if known == 0 or self.size % known != 0:
                raise ValueError(
                    f"cannot reshape packed uint4 array of size {self.size} "
                    f"into shape {tuple(values)}"
                )
            values[inferred[0]] = self.size // known
        normalized = tuple(values)
        if prod(normalized) != self.size:
            raise ValueError(
                f"cannot reshape packed uint4 array of size {self.size} "
                f"into shape {normalized}"
            )
        return PackedUInt4Array(
            self.buffer,
            normalized,
            self.backend,
            dict(self.metadata),
        )

    def as_uint8(self) -> Any:
        """Return an unpacked uint8 array on the same backend when supported."""
        if self.backend == "cuda":
            return unpack_uint4_cupy(self)
        if self.backend == "numpy":
            return unpack_uint4_numpy(self)
        raise NotImplementedError(
            f"Unpacking packed uint4 backend={self.backend!r} is not implemented."
        )

    def get(self) -> np.ndarray:
        """Return an unpacked NumPy uint8 copy."""
        unpacked = self.as_uint8()
        if hasattr(unpacked, "get"):
            unpacked = unpacked.get()
        return np.asarray(unpacked, dtype=np.uint8)

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self.get()
        return arr.astype(dtype, copy=False) if dtype is not None else arr


def is_packed_uint4(value: Any) -> bool:
    """Return whether ``value`` follows the packed uint4 storage contract."""
    return bool(getattr(value, "_is_packed_uint4", False))


def _require_uint4_range_numpy(data: np.ndarray) -> None:
    if data.size == 0:
        return
    if np.asarray(data).dtype.kind not in {"u", "i", "b"}:
        raise TypeError("uint4 packing requires integer detector counts.")
    mn = int(np.min(data))
    mx = int(np.max(data))
    if mn < 0 or mx > 15:
        raise ValueError(
            "dtype='u4' requires every corrected detector count to fit in "
            f"0..15; observed min={mn}, max={mx}. Use dtype='uint8' or "
            "dtype='uint16' for this dataset."
        )


def pack_uint4_numpy(data: np.ndarray) -> PackedUInt4Array:
    """Pack a NumPy integer array into ``PackedUInt4Array`` with exact audit."""
    arr = np.asarray(data)
    _require_uint4_range_numpy(arr)
    flat = arr.reshape(-1).astype(np.uint8, copy=False)
    packed = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    packed[: flat[0::2].size] = flat[0::2] & np.uint8(0x0F)
    if flat.size > 1:
        packed[: flat[1::2].size] |= (flat[1::2] & np.uint8(0x0F)) << np.uint8(4)
    return PackedUInt4Array(packed, tuple(int(x) for x in arr.shape), "numpy")


def unpack_uint4_numpy(data: PackedUInt4Array) -> np.ndarray:
    """Unpack a NumPy-backed ``PackedUInt4Array`` to uint8."""
    packed = np.asarray(data.buffer, dtype=np.uint8).reshape(-1)
    out = np.empty(data.size, dtype=np.uint8)
    out[0::2] = packed[: out[0::2].size] & np.uint8(0x0F)
    if out.size > 1:
        out[1::2] = (packed[: out[1::2].size] >> np.uint8(4)) & np.uint8(0x0F)
    return out.reshape(data.shape)


def pack_uint4_cupy(data: Any) -> PackedUInt4Array:
    """Pack a CuPy integer array into ``PackedUInt4Array`` with exact audit."""
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        raise TypeError("pack_uint4_cupy expects a CuPy array.")
    if data.dtype.kind not in "uib":
        raise TypeError("uint4 packing requires integer detector counts.")
    if data.size:
        mx = int(data.max().get())
        mn = int(data.min().get())
        if mn < 0 or mx > 15:
            raise ValueError(
                "dtype='u4' requires every corrected detector count to fit in "
                f"0..15; observed min={mn}, max={mx}. Use dtype='uint8' or "
                "dtype='uint16' for this dataset."
            )
    flat = cp.ascontiguousarray(data.reshape(-1))
    lo = flat[0::2].astype(cp.uint8, copy=False) & cp.uint8(0x0F)
    packed = cp.empty((int(flat.size) + 1) // 2, dtype=cp.uint8)
    packed[: int(lo.size)] = lo
    if int(flat.size) > 1:
        hi = (flat[1::2].astype(cp.uint8, copy=False) & cp.uint8(0x0F)) << cp.uint8(4)
        packed[: int(hi.size)] |= hi
        if int(lo.size) > int(hi.size):
            packed[-1] &= cp.uint8(0x0F)
    return PackedUInt4Array(packed, tuple(int(x) for x in data.shape), "cuda")


def unpack_uint4_cupy(data: PackedUInt4Array) -> Any:
    """Unpack a CUDA-backed ``PackedUInt4Array`` to a CuPy uint8 array."""
    import cupy as cp

    packed = cp.asarray(data.buffer, dtype=cp.uint8).reshape(-1)
    out = cp.empty(data.size, dtype=cp.uint8)
    out[0::2] = packed[: int(out[0::2].size)] & cp.uint8(0x0F)
    if int(out.size) > 1:
        out[1::2] = (packed[: int(out[1::2].size)] >> cp.uint8(4)) & cp.uint8(0x0F)
    return out.reshape(data.shape)
