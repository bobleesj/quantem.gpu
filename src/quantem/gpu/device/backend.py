"""Accelerator discovery and explicit backend resolution."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import Literal

DeviceName = Literal["cuda", "mps", "webgpu"]
NativeDeviceName = Literal["cuda", "mps"]


def profile(device: str | None = None) -> dict[str, str | list[str] | None]:
    """Return a notebook-friendly summary of the active compute environment.

    With no argument, choose CUDA device 0, MPS, or CPU automatically and never
    raise when an accelerator is unavailable. Pass an explicit device such as
    ``"cuda:1"`` to select and validate another visible CUDA GPU. In a
    notebook, keep the returned ``device`` value as the single source of truth
    for later calls. ``available_devices`` lists privacy-safe device IDs so a
    multi-GPU machine is visible without exposing hardware names. Hostnames,
    executable paths, and detailed hardware names are intentionally omitted.

    Parameters
    ----------
    device : str or None, default None
        Requested Torch device. Use ``None`` or ``"auto"`` for automatic
        selection, ``"cuda"`` or ``"cuda:0"`` for the first visible CUDA GPU,
        ``"cuda:N"`` for another visible GPU, ``"mps"`` for Apple Metal, or
        ``"cpu"`` for CPU execution. CUDA indices refer to the devices visible
        to the process.
    """
    cuda_available = False
    cuda_count = 0
    mps_available = False
    torch_module = None
    try:
        import torch

        torch_module = torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        mps_available = bool(torch.backends.mps.is_available())
        torch_version = str(torch.__version__)
    except Exception:  # noqa: BLE001 - diagnostics must remain non-blocking
        torch_version = None

    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"", "auto"} and cuda_available:
        backend = "cuda"
        resolved_device = "cuda:0"
    elif requested in {"", "auto"} and mps_available:
        backend = "mps"
        resolved_device = "mps"
    elif requested in {"", "auto", "cpu"}:
        backend = "cpu"
        resolved_device = "cpu"
    elif requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS device is unavailable; use profile() for automatic selection."
            )
        backend = "mps"
        resolved_device = "mps"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_available or torch_module is None:
            raise RuntimeError(
                "CUDA device is unavailable; use profile() for automatic selection."
            )
        if requested == "cuda":
            cuda_index = 0
        else:
            try:
                cuda_index = int(requested.removeprefix("cuda:"))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid CUDA device {device!r}; use 'cuda' or 'cuda:N', for example 'cuda:1'."
                ) from exc
        if cuda_index < 0 or cuda_index >= cuda_count:
            raise ValueError(
                f"CUDA device {device!r} is unavailable: {cuda_count} CUDA device(s) are visible. "
                f"Choose an index from 0 to {cuda_count - 1}."
            )
        backend = "cuda"
        resolved_device = f"cuda:{cuda_index}"
    else:
        raise ValueError(
            f"Unknown device {device!r}; use 'auto', 'cuda:N', 'mps', or 'cpu'."
        )

    if cuda_count:
        available_devices = [f"cuda:{index}" for index in range(cuda_count)]
    elif mps_available:
        available_devices = ["mps"]
    else:
        available_devices = ["cpu"]

    return {
        "platform": platform.platform(),
        "torch": torch_version,
        "backend": backend,
        "device": resolved_device,
        "available_devices": available_devices,
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
        f"No QuantEM GPU backend is available. CUDA: {cuda_error}. MPS: {mps_error}."
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
