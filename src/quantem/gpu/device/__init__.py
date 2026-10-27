"""GPU device discovery."""

from quantem.gpu.device.select import (
    detect,
    profile,
    release_cached_memory,
    resolve,
    resolve_device,
    runtime_notice,
)
from quantem.gpu.device.telemetry import status

__all__ = [
    "detect",
    "profile",
    "release_cached_memory",
    "resolve",
    "resolve_device",
    "runtime_notice",
    "status",
]
