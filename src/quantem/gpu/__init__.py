"""Multi-backend accelerated STEM IO and compute for QuantEM."""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from .device import DeviceReport, device_report, select_device
from .calibration import (
    CalibrationMemoryPlan,
    CalibrationProducts,
    calibration_memory_plan,
    calibration_products_cache_path,
    load_calibration_products,
)
from .compute import (
    VirtualImageKernelSupport,
    compute_backend,
    virtual_image_kernel_support,
)
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

try:
    __version__ = version("quantem.gpu")
except PackageNotFoundError:
    __version__ = "0.0.1rc5"

_SSB_EXPORTS = {
    "DefocusSweepResult",
    "SSB",
    "SSBResult",
    "SSBTimeSeriesResult",
    "VirtualImageKernelSupport",
    "defocus_sweep",
    "ssb",
    "ssb_time_average",
    "ssb_time_series",
    "ssb_fit_mps",
    "ssb_preview_mps",
    "ssb_series",
    "bf_df_dpc",
    "Preview",
}
_IO_EXPORTS = {
    "load",
    "load_scan_indices",
    "load_scan_region",
    "random_scan_indices",
}
_PARALLAX_EXPORTS = {
    "BFImage": ("quantem.gpu.parallax_results", "BFImage"),
    "Parallax": ("quantem.gpu.parallax", "Parallax"),
    "ParallaxResult": ("quantem.gpu.parallax_results", "ParallaxResult"),
    "parallax": ("quantem.gpu.parallax", "parallax"),
}
_LAZY_MODULE_EXPORTS = {
    "movie": "quantem.gpu.movie",
    "webgpu": "quantem.gpu.webgpu",
}

__all__ = [
    "DPCResult",
    "DefocusSweepResult",
    "CalibrationMemoryPlan",
    "CalibrationProducts",
    "DeviceReport",
    "BFImage",
    "SSB",
    "SSBResult",
    "SSBTimeSeriesResult",
    "Preview",
    "Parallax",
    "ParallaxResult",
    "adf",
    "auto_probe",
    "bf",
    "bf_df_dpc",
    "calibration_products_cache_path",
    "calibration_memory_plan",
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
    "load",
    "load_calibration_products",
    "load_scan_indices",
    "load_scan_region",
    "masked_sum",
    "mean_dp",
    "movie",
    "parallax",
    "random_scan_indices",
    "select_device",
    "ssb",
    "ssb_time_average",
    "ssb_time_series",
    "ssb_fit_mps",
    "ssb_preview_mps",
    "ssb_series",
    "virtual",
    "virtual_image",
    "virtual_image_kernel_support",
    "webgpu",
    "__version__",
]


def __getattr__(name: str):
    """Load CUDA-only SSB exports lazily so CPU/MPS imports stay lightweight."""
    if name in _SSB_EXPORTS:
        module = import_module("quantem.gpu.ssb")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _IO_EXPORTS:
        module = import_module("quantem.gpu.io")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _PARALLAX_EXPORTS:
        module_name, attr = _PARALLAX_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    if name in _LAZY_MODULE_EXPORTS:
        module = import_module(_LAZY_MODULE_EXPORTS[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
