"""CUDA image-alignment helpers used by GPU reconstruction modules."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "cross_correlation_shift_batch_cp": (
        "quantem.gpu.image.dft_upsample",
        "cross_correlation_shift_batch_cp",
    ),
    "cross_correlation_shift_cp": (
        "quantem.gpu.image.dft_upsample",
        "cross_correlation_shift_cp",
    ),
    "dft_upsample_cp": ("quantem.gpu.image.dft_upsample", "dft_upsample_cp"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
