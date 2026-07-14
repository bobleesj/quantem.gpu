"""Accelerated IO APIs for QuantEM STEM data."""
from __future__ import annotations

from importlib import import_module

_HDF5_EXPORTS = {
    "H5Writer",
    "LoadResult",
    "MasterReadiness",
    "bin",
    "discover_masters",
    "find_emd_sibling",
    "get_metadata",
    "inspect_master_readiness",
    "is_master_ready",
    "load",
    "load_scan_region",
    "load_parallel",
    "disk_of",
    "group_by_disk",
    "read_emd_metadata",
    "read_pixel_mask",
    "save",
    "wait_for_saves",
}

_BACKEND_EXPORTS = {"detect_backend", "resolve_backend"}

_MPS_EXPORTS = {"MPSChunked4DSTEM", "clear_mps_cache", "load_mps_4dstem"}

__all__ = sorted(_HDF5_EXPORTS | _BACKEND_EXPORTS | _MPS_EXPORTS)


def __getattr__(name: str):
    if name in _HDF5_EXPORTS:
        module = import_module("quantem.gpu.io.hdf5")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _BACKEND_EXPORTS:
        module = import_module("quantem.gpu.io.backends")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _MPS_EXPORTS:
        module = import_module("quantem.gpu.io.backends.mps")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
