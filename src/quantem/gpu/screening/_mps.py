"""MPS adapters for streamed screening products."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np


def _clear_mps_transients() -> None:
    """Release Python references after a chunked MPS cache-build step."""
    gc.collect()


def _metadata_dtype(metadata: dict[str, Any], fallback) -> np.dtype:
    """Return the native detector dtype when metadata provides it."""
    dtype = metadata.get("dtype")
    if dtype is None:
        return np.dtype(fallback)
    try:
        return np.dtype(dtype)
    except TypeError:
        return np.dtype(fallback)


def _metal_buffer_for(array):
    """Return an MPS ndarray's backing Metal buffer, following base views."""
    current = array
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        buffer = getattr(current, "_mtl", None)
        if buffer is not None:
            return buffer
        current = getattr(current, "base", None)
    return None


def _mps_chunked_frames_for(data):
    """Return a chunk-backed view over one Metal-resident load result."""
    if getattr(data, "_is_gpu_frames", False):
        return data
    metal_buffer = _metal_buffer_for(data)
    if metal_buffer is None:
        raise RuntimeError(
            "MPS screening products require Metal-backed load output; got "
            f"{type(data).__name__}. Use backend='cuda' or backend='mps'."
        )
    from quantem.gpu.detector.compute.mps.kernels import ChunkedFrames
    from quantem.gpu.io.backends import mps

    array = data
    if array.ndim == 4:
        flat_shape = (
            int(array.shape[0]) * int(array.shape[1]),
            *array.shape[-2:],
        )
        array = array.reshape(flat_shape)
    elif array.ndim != 3:
        raise ValueError(
            f"Expected 3D or 4D MPS detector data, got shape {array.shape}."
        )
    try:
        array._mtl = metal_buffer
    except AttributeError:
        array = np.asarray(array).reshape(array.shape).view(mps._MtlArray)
        array._mtl = metal_buffer
    return ChunkedFrames([array])


def _mps_mean_dp(frames) -> np.ndarray:
    """Return the float32 mean diffraction pattern from Metal-resident frames."""
    return np.asarray(frames.vi.detector_sum(), dtype=np.float32) / int(frames.vi.n)
