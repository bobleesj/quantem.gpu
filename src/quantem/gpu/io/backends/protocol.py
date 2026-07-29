"""Backend selection contract for accelerated STEM I/O."""
from __future__ import annotations

from typing import Literal, TypeAlias

BackendName: TypeAlias = Literal["cuda", "mps", "cpu"]
_VALID: tuple[BackendName, ...] = ("cuda", "mps", "cpu")


def _nvidia_gpu_present() -> bool:
    """Return whether an NVIDIA device is present without importing CuPy."""
    from quantem.gpu.device import _nvidia_gpu_present as _probe

    return _probe()


def _has_cuda() -> bool:
    from quantem.gpu.device import _cuda_probe

    available, _count, _error = _cuda_probe()
    return available


def _has_mps() -> bool:
    from quantem.gpu.device import _mps_probe

    available, _error = _mps_probe()
    return available


def detect_backend() -> Literal["cuda", "mps"]:
    """Select an accelerated backend without silently falling back to CPU."""
    if _has_cuda():
        return "cuda"
    if _nvidia_gpu_present():
        raise RuntimeError(
            "An NVIDIA GPU is present, but the CUDA I/O backend is unavailable. "
            "Install a CuPy build matching the installed CUDA toolkit and retry."
        )
    if _has_mps():
        return "mps"
    raise RuntimeError(
        "No accelerated QuantEM I/O backend is available. Install CuPy for "
        "CUDA or run on Apple Silicon with MPS. CPU I/O is a reference/test "
        "path only and is never selected by backend='auto'; request "
        "backend='cpu' explicitly for a reference comparison."
    )


def resolve_backend(backend: str | None) -> BackendName:
    """Validate an explicit backend or resolve ``None``/``auto``."""
    if backend in (None, "auto"):
        return detect_backend()
    if backend not in _VALID:
        allowed = ", ".join(repr(name) for name in _VALID)
        raise ValueError(
            f"Unknown I/O backend {backend!r}. Use 'auto' or one of {allowed}."
        )
    return backend
