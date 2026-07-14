"""Multi-backend accelerated STEM IO and compute for QuantEM."""
from __future__ import annotations

from .device import DeviceReport, device_report, select_device
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
from .ssb import DefocusSweepResult, SSB, SSBResult, defocus_sweep, ssb, ssb_series

__version__ = "0.0.1"

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
    "ssb_series",
    "virtual",
    "virtual_image",
    "__version__",
]
