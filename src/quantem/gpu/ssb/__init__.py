"""Single-sideband ptychography compute API for QuantEM GPU backends."""
from __future__ import annotations

from importlib import import_module

_API_EXPORTS = {
    "defocus_sweep": ("quantem.gpu.ssb.api", "defocus_sweep"),
    "dpc": ("quantem.gpu.ssb.api", "dpc"),
    "ssb": ("quantem.gpu.ssb.api", "ssb"),
    "ssb_series": ("quantem.gpu.ssb.api", "ssb_series"),
    "SSBEngine": ("quantem.gpu.ssb.engine", "SSBEngine"),
    "SSB": ("quantem.gpu.ssb.reconstruction", "SSB"),
    "spatial_frequencies": ("quantem.gpu.ssb.reconstruction", "spatial_frequencies"),
    "BFRadiusSweepResult": ("quantem.gpu.ssb.results", "BFRadiusSweepResult"),
    "DPCResult": ("quantem.gpu.ssb.results", "DPCResult"),
    "DefocusSweepResult": ("quantem.gpu.ssb.results", "DefocusSweepResult"),
    "SSBResult": ("quantem.gpu.ssb.results", "SSBResult"),
    "MpsSSBPreviewResult": ("quantem.gpu.ssb.mps", "MpsSSBPreviewResult"),
    "ssb_preview_mps": ("quantem.gpu.ssb.mps", "ssb_preview"),
}

__all__ = [
    "BFRadiusSweepResult",
    "DPCResult",
    "DefocusSweepResult",
    "SSB",
    "SSBEngine",
    "SSBResult",
    "MpsSSBPreviewResult",
    "defocus_sweep",
    "dpc",
    "spatial_frequencies",
    "ssb",
    "ssb_preview_mps",
    "ssb_series",
]


def __getattr__(name: str):
    if name in _API_EXPORTS:
        module_name, attr = _API_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
