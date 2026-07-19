"""Virtual-image custom-kernel support checks.

The support probe is shape-only when needed, so future large cases such as
``1024x1024x192x192 uint8`` can be checked without allocating the full stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


VirtualImageBackend = Literal["cuda", "mps", "webgpu", "torch", "cpu"]


@dataclass(frozen=True)
class VirtualImageKernelSupport:
    """Support status for one virtual-image backend target."""

    backend: str
    available: bool
    custom_kernel: bool
    kernel: str
    dtype: str
    shape: tuple[int, ...] | None
    scan_shape: tuple[int, ...] | None
    det_shape: tuple[int, int] | None
    resident_gib: float | None
    mask_paths: dict[str, str]
    notes: tuple[str, ...]


def _module_root(value: Any) -> str:
    return type(value).__module__.split(".", 1)[0]


def _shape_from_data(data: Any | None) -> tuple[int, ...] | None:
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    return tuple(int(x) for x in shape)


def _dtype_from_data(data: Any | None) -> np.dtype | None:
    dtype = getattr(data, "dtype", None)
    if dtype is None:
        return None
    try:
        return np.dtype(dtype)
    except TypeError:
        return None


def _scan_det_shape(
    shape: tuple[int, ...] | None,
) -> tuple[tuple[int, ...] | None, tuple[int, int] | None]:
    if shape is None:
        return None, None
    if len(shape) == 4:
        return (int(shape[0]), int(shape[1])), (int(shape[2]), int(shape[3]))
    if len(shape) == 3:
        n = int(shape[0])
        side = int(math.isqrt(n))
        scan_shape = (side, side) if side * side == n else (n,)
        return scan_shape, (int(shape[1]), int(shape[2]))
    return None, None


def _resident_gib(shape: tuple[int, ...] | None, dtype: np.dtype | None) -> float | None:
    if shape is None or dtype is None:
        return None
    nbytes = int(np.prod(shape, dtype=np.uint64)) * int(dtype.itemsize)
    return nbytes / float(1 << 30)


def _detector_mask_pixels(det_shape: tuple[int, int], lo: float, hi: float) -> int:
    rows = np.arange(det_shape[0], dtype=np.float32)[:, None]
    cols = np.arange(det_shape[1], dtype=np.float32)[None, :]
    center = ((det_shape[0] - 1) / 2.0, (det_shape[1] - 1) / 2.0)
    dist = np.sqrt((rows - center[0]) ** 2 + (cols - center[1]) ** 2)
    return int(((dist >= float(lo)) & (dist <= float(hi))).sum())


def _uint32_accum_safe(n_pixels: int, dtype: np.dtype) -> bool:
    if not np.issubdtype(dtype, np.integer):
        return False
    info = np.iinfo(dtype)
    return int(n_pixels) * int(info.max) <= int(np.iinfo(np.uint32).max)


def _cuda_mask_paths(
    det_shape: tuple[int, int] | None,
    dtype: np.dtype | None,
    bf_radius: float,
) -> dict[str, str]:
    if det_shape is None or dtype is None or dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        return {}
    n_det = int(det_shape[0] * det_shape[1])
    bf = _detector_mask_pixels(det_shape, 0.0, bf_radius)
    adf = _detector_mask_pixels(det_shape, bf_radius, bf_radius * 2.0)
    df_selected = n_det - bf
    paths: dict[str, str] = {}
    paths["BF"] = "cuda_rawkernel_selected" if _uint32_accum_safe(bf, dtype) else "fallback"
    paths["ADF"] = "cuda_rawkernel_selected" if _uint32_accum_safe(adf, dtype) else "fallback"
    paths["DF"] = (
        "cuda_rawkernel_total_minus_complement"
        if _uint32_accum_safe(n_det - df_selected, dtype)
        else "fallback"
    )
    return paths


def _infer_backend(data: Any | None, backend: str) -> str:
    if backend != "auto":
        return backend
    if data is None:
        return "cuda"
    root = _module_root(data)
    if root == "cupy":
        return "cuda"
    if getattr(data, "_is_gpu_frames", False):
        return "mps"
    if root == "torch":
        return "torch"
    return "cpu"


def virtual_image_kernel_support(
    data: Any | None = None,
    *,
    backend: str = "auto",
    shape: tuple[int, ...] | None = None,
    dtype: Any | None = None,
    bf_radius: float = 30.0,
) -> VirtualImageKernelSupport:
    """Return whether BF/DF/ADF virtual images have a custom-kernel path.

    Parameters
    ----------
    data
        Optional resident data object. CuPy arrays map to CUDA and MPS
        chunk-backed objects map to the Metal path.
    backend
        ``"auto"``, ``"cuda"``, ``"mps"``, ``"webgpu"``, ``"torch"``, or
        ``"cpu"``. ``"webgpu"`` reports the Show4DSTEM browser WGSL contract
        shipped in ``quantem.gpu.webgpu`` and bundled by ``quantem.widget``.
    shape, dtype
        Optional shape-only probe. Use this for large future systems such as
        ``shape=(1024, 1024, 192, 192), dtype=np.uint8``.
    bf_radius
        Bright-field radius in detector pixels for estimating BF/ADF/DF kernel
        paths.
    """
    shape = tuple(int(x) for x in (shape or _shape_from_data(data) or ())) or None
    dtype_np = np.dtype(dtype) if dtype is not None else _dtype_from_data(data)
    scan_shape, det_shape = _scan_det_shape(shape)
    selected_backend = _infer_backend(data, str(backend).lower())
    dtype_name = "unknown" if dtype_np is None else str(dtype_np)
    resident = _resident_gib(shape, dtype_np)
    notes: list[str] = []

    if selected_backend == "cuda":
        available = dtype_np in (np.dtype("uint8"), np.dtype("uint16")) and det_shape is not None
        if shape is not None and len(shape) not in (3, 4):
            notes.append("CUDA virtual-image kernels expect 3D or 4D 4D-STEM data.")
        if dtype_np not in (np.dtype("uint8"), np.dtype("uint16")):
            notes.append("CUDA RawKernel path currently supports resident uint8/uint16 arrays.")
        mask_paths = _cuda_mask_paths(det_shape, dtype_np, bf_radius)
        return VirtualImageKernelSupport(
            backend="cuda",
            available=bool(available),
            custom_kernel=bool(available),
            kernel="quantem.gpu.compute.cuda RawKernel selected-sum",
            dtype=dtype_name,
            shape=shape,
            scan_shape=scan_shape,
            det_shape=det_shape,
            resident_gib=resident,
            mask_paths=mask_paths,
            notes=tuple(notes),
        )

    if selected_backend == "mps":
        available = (
            dtype_np in (np.dtype("uint8"), np.dtype("uint16"))
            and det_shape is not None
        )
        if dtype_np not in (np.dtype("uint8"), np.dtype("uint16")):
            notes.append(
                "MPS MetalVirtualImage currently targets uint8/uint16 "
                "chunk-backed data."
            )
        return VirtualImageKernelSupport(
            backend="mps",
            available=bool(available),
            custom_kernel=bool(available),
            kernel="quantem.gpu.compute.mps MetalVirtualImage",
            dtype=dtype_name,
            shape=shape,
            scan_shape=scan_shape,
            det_shape=det_shape,
            resident_gib=resident,
            mask_paths={
                "BF": "mps_metal_selected",
                "ADF": "mps_metal_selected",
                "DF": "mps_metal_total_minus_complement",
            }
            if available
            else {},
            notes=tuple(notes),
        )

    if selected_backend == "webgpu":
        available = (
            dtype_np in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"))
            and det_shape is not None
        )
        if dtype_np not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")):
            notes.append(
                "Show4DSTEM WebGPU masked-sum supports uint8, uint16, "
                "and float32 packed data."
            )
        notes.append(
            "Browser adapter must be verified at runtime; SwiftShader is not "
            "a performance pass."
        )
        return VirtualImageKernelSupport(
            backend="webgpu",
            available=bool(available),
            custom_kernel=bool(available),
            kernel="quantem.gpu.webgpu Show4DSTEMCompute WGSL selected-index reducer",
            dtype=dtype_name,
            shape=shape,
            scan_shape=scan_shape,
            det_shape=det_shape,
            resident_gib=resident,
            mask_paths={
                "BF": "webgpu_wgsl_selected",
                "ADF": "webgpu_wgsl_selected",
                "DF": "webgpu_wgsl_selected",
            }
            if available
            else {},
            notes=tuple(notes),
        )

    return VirtualImageKernelSupport(
        backend=selected_backend,
        available=False,
        custom_kernel=False,
        kernel="fallback",
        dtype=dtype_name,
        shape=shape,
        scan_shape=scan_shape,
        det_shape=det_shape,
        resident_gib=resident,
        mask_paths={},
        notes=("No custom virtual-image kernel path for this backend target.",),
    )
