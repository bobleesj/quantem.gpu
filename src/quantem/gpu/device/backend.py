"""Accelerator discovery and explicit backend resolution."""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import Literal


DeviceName = Literal["cuda", "mps", "webgpu"]
NativeDeviceName = Literal["cuda", "mps"]


def profile() -> dict[str, str | bool | None]:
    """Return a notebook-friendly summary of the active compute environment.

    The function is diagnostic only: it never raises when no accelerator is
    available and it does not print. In a notebook, use ``profile()`` as the
    final expression of a cell to render the host, Python environment, and
    resolved CUDA/MPS/CPU backend.
    """
    cuda_available = False
    mps_available = False
    cuda_name: str | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_name = str(torch.cuda.get_device_name(0))
        mps_available = bool(torch.backends.mps.is_available())
        torch_version = str(torch.__version__)
    except Exception:  # noqa: BLE001 - diagnostics must remain non-blocking
        torch_version = None

    if cuda_available:
        backend = "cuda"
        device = "cuda:0"
        device_name = cuda_name
    elif mps_available:
        backend = "mps"
        device = "mps"
        device_name = "Apple Metal (MPS)"
    else:
        backend = "cpu"
        device = "cpu"
        device_name = "CPU"

    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.executable,
        "torch": torch_version,
        "backend": backend,
        "device": device,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
    }


def _nvidia_gpu_present() -> bool:
    return sys.platform.startswith("linux") and os.path.exists("/dev/nvidia0")


def _cuda_probe() -> tuple[bool, str | None]:
    try:
        cupy_spec = importlib.util.find_spec("cupy")
    except ModuleNotFoundError:
        cupy_spec = None
    if cupy_spec is None:
        note = "CuPy is not installed"
        if _nvidia_gpu_present():
            note += "; install the CuPy build matching the installed CUDA runtime"
        return False, note
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001 - optional runtime probe
        return False, str(exc)
    if count < 1:
        return False, "CuPy imported, but no CUDA device is visible"
    return True, None


def _mps_probe() -> tuple[bool, str | None]:
    if sys.platform != "darwin":
        return False, f"MPS requires macOS; current platform is {platform.system()}"
    if importlib.util.find_spec("Metal") is not None:
        return True, None
    try:
        import torch

        if bool(torch.backends.mps.is_available()):
            return True, None
        return False, "Torch is installed, but its MPS backend is unavailable"
    except Exception as exc:  # noqa: BLE001 - optional runtime probe
        return False, str(exc)


def detect() -> NativeDeviceName:
    """Return the available native GPU backend.

    CUDA takes precedence when both runtimes are visible. CPU is intentionally
    not a scientific fallback. Browser WebGPU must be selected explicitly by
    the browser-facing caller.
    """

    cuda_available, cuda_error = _cuda_probe()
    if cuda_available:
        return "cuda"
    mps_available, mps_error = _mps_probe()
    if mps_available:
        return "mps"
    raise RuntimeError(
        "No QuantEM GPU backend is available. "
        f"CUDA: {cuda_error}. MPS: {mps_error}."
    )


def resolve(name: str | None = "auto") -> DeviceName:
    """Validate and resolve a CUDA, MPS, or browser WebGPU backend name."""

    requested = "auto" if name is None else str(name).lower()
    if requested == "auto":
        return detect()
    if requested == "webgpu":
        return "webgpu"
    if requested == "cuda":
        available, error = _cuda_probe()
    elif requested == "mps":
        available, error = _mps_probe()
    else:
        raise ValueError(
            f"Unknown GPU backend {name!r}. Use 'auto', 'cuda', 'mps', or 'webgpu'."
        )
    if not available:
        raise RuntimeError(f"{requested.upper()} backend is unavailable: {error}")
    return requested
