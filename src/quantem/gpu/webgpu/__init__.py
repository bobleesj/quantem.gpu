"""Canonical WebGPU source files for QuantEM browser compute.

The browser still executes these kernels from ``quantem.widget`` bundles, but
the reusable WebGPU engine sources live here so CUDA, MPS, and WebGPU product
math can be reviewed and versioned together.
"""
from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable

WEBGPU_SOURCE_NAMES: tuple[str, ...] = (
    "webgpu/bslz4.ts",
    "webgpu/compute.ts",
    "webgpu/device.ts",
    "webgpu/fft-shader.ts",
    "webgpu/h5reader.ts",
    "webgpu/lazy.ts",
    "webgpu/local-h5.ts",
    "ssb/compute/webgpu/backend.ts",
    "ssb/compute/webgpu/optimizer.ts",
    "ssb/compute/webgpu/protocol.ts",
    "ssb/compute/webgpu/kernels/common.ts",
    "ssb/compute/webgpu/kernels/fft128.ts",
    "ssb/compute/webgpu/kernels/fft256.ts",
    "ssb/compute/webgpu/kernels/fft512.ts",
    "ssb/compute/webgpu/kernels/fft1024.ts",
    "ssb/compute/webgpu/kernels/index.ts",
)

__all__ = [
    "WEBGPU_SOURCE_NAMES",
    "source_names",
    "source_path",
    "source_text",
]


def source_names() -> tuple[str, ...]:
    """Return the canonical WebGPU source filenames shipped with the package."""
    return WEBGPU_SOURCE_NAMES


def source_path(name: str) -> Traversable:
    """Return an importlib resource handle for one canonical WebGPU source file."""
    if name not in WEBGPU_SOURCE_NAMES:
        allowed = ", ".join(WEBGPU_SOURCE_NAMES)
        raise ValueError(f"Unknown WebGPU source {name!r}; expected one of: {allowed}.")
    return files("quantem.gpu").joinpath(name)


def source_text(name: str) -> str:
    """Read one canonical WebGPU source file as UTF-8 text."""
    return source_path(name).read_text(encoding="utf-8")
