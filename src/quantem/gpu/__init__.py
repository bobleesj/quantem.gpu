"""Multi-backend accelerated STEM IO and compute for QuantEM."""
from __future__ import annotations

from importlib import import_module

from .device import DeviceReport, device_report, select_device
from .compute import compute_backend
from .detector import (
    adf,
    auto_probe,
    bf,
    detector_mask,
    detect_bf_radius,
    df,
    dp_mean,
    masked_sum,
    mean_dp,
    virtual,
    virtual_image,
)
from .dpc import DPCResult, center_of_mass, com, dpc, idpc

__version__ = "0.0.1"

_SSB_EXPORTS = {
    "DefocusSweepResult",
    "SSB",
    "SSBResult",
    "defocus_sweep",
    "ssb",
    "ssb_preview_mps",
    "ssb_series",
}

__all__ = [
    "DPCResult",
    "DefocusSweepResult",
    "DeviceReport",
    "SSB",
    "SSBResult",
    "adf",
    "auto_probe",
    "bf",
    "center_of_mass",
    "com",
    "compute_backend",
    "detector_mask",
    "detect_bf_radius",
    "df",
    "defocus_sweep",
    "device_report",
    "dpc",
    "dp_mean",
    "idpc",
    "masked_sum",
    "mean_dp",
    "select_device",
    "ssb",
    "ssb_preview_mps",
    "ssb_series",
    "virtual",
    "virtual_image",
    "__version__",
]


def __getattr__(name: str):
    """Load CUDA-only SSB exports lazily so CPU/MPS imports stay lightweight."""
    if name in _SSB_EXPORTS:
        module = import_module("quantem.gpu.ssb")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
