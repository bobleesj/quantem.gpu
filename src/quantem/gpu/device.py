"""Device selection and reporting for QuantEM accelerated backends."""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from typing import Literal

Backend = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceReport:
    """Availability report for accelerated QuantEM backends.

    Parameters
    ----------
    selected
        Backend selected by the current policy.
    cuda_available
        Whether CuPy imports and reports at least one CUDA device.
    cuda_device_count
        Number of visible CUDA devices, or zero when CUDA is unavailable.
    cuda_error
        Probe error when CUDA is unavailable for a known reason.
    mps_available
        Whether an Apple MPS/Metal runtime is importable.
    mps_error
        Probe note or error for MPS.
    cpu_available
        CPU fallback availability. Always true for this package.
    """

    selected: Backend
    cuda_available: bool
    cuda_device_count: int
    cuda_error: str | None
    mps_available: bool
    mps_error: str | None
    cpu_available: bool = True


def _nvidia_gpu_present() -> bool:
    return sys.platform.startswith("linux") and os.path.exists("/dev/nvidia0")


def _cuda_probe() -> tuple[bool, int, str | None]:
    if importlib.util.find_spec("cupy") is None:
        note = "cupy is not installed"
        if _nvidia_gpu_present():
            note += "; install cupy-cuda12x/cupy-cuda13x or conda-forge cupy"
        return False, 0, note
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001 - CUDA probes raise runtime-specific errors
        return False, 0, str(exc)
    if count <= 0:
        return False, 0, "cupy imported but no CUDA devices are visible"
    return True, count, None


def _mps_probe() -> tuple[bool, str | None]:
    if sys.platform != "darwin":
        return False, f"MPS is only available on macOS; current platform is {platform.system()}"
    if importlib.util.find_spec("Metal") is not None:
        return True, None
    try:
        import torch

        if bool(torch.backends.mps.is_available()):
            return True, None
        return False, "torch MPS backend is installed but not available"
    except Exception as exc:  # noqa: BLE001 - optional backend probe
        return False, str(exc)


def select_device(preferred: str | None = "auto") -> Backend:
    """Select an accelerated backend.

    Parameters
    ----------
    preferred
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    Backend
        Selected backend name.

    Raises
    ------
    RuntimeError
        If an explicit accelerated backend is unavailable.
    ValueError
        If ``preferred`` is not a supported backend token.
    """

    requested = "auto" if preferred is None else str(preferred).lower()
    valid = {"auto", "cuda", "mps", "cpu"}
    if requested not in valid:
        raise ValueError(
            f"Unknown device backend {preferred!r}. Use 'auto', 'cuda', 'mps', or 'cpu'."
        )

    cuda_ok, _cuda_count, cuda_error = _cuda_probe()
    mps_ok, mps_error = _mps_probe()

    if requested == "cuda":
        if cuda_ok:
            return "cuda"
        raise RuntimeError(f"CUDA backend requested but unavailable: {cuda_error}")
    if requested == "mps":
        if mps_ok:
            return "mps"
        raise RuntimeError(f"MPS backend requested but unavailable: {mps_error}")
    if requested == "cpu":
        return "cpu"
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def device_report(preferred: str | None = "auto") -> DeviceReport:
    """Return backend availability and the selected backend."""

    cuda_ok, cuda_count, cuda_error = _cuda_probe()
    mps_ok, mps_error = _mps_probe()
    selected = select_device(preferred)
    return DeviceReport(
        selected=selected,
        cuda_available=cuda_ok,
        cuda_device_count=cuda_count,
        cuda_error=cuda_error,
        mps_available=mps_ok,
        mps_error=mps_error,
    )
