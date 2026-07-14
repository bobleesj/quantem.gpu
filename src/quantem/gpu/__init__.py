"""Multi-backend accelerated STEM IO and compute for QuantEM."""
from __future__ import annotations

from .device import DeviceReport, device_report, select_device

__version__ = "0.0.1"

__all__ = [
    "DeviceReport",
    "device_report",
    "select_device",
    "__version__",
]
