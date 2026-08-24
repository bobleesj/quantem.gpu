"""GPU-accelerated scientific workflows for QuantEM."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from .ssb import SSB, SSBResult

try:
    __version__ = version("quantem.gpu")
except PackageNotFoundError:
    __version__ = "0.0.1rc6"


_NAMESPACES = {
    "detector",
    "device",
    "dpc",
    "geometry",
    "io",
    "movie",
    "optics",
    "parallax",
    "screening",
}

__all__ = [
    "SSB",
    "SSBResult",
    "__version__",
    "detector",
    "device",
    "dpc",
    "geometry",
    "io",
    "movie",
    "optics",
    "parallax",
    "screening",
]


def __getattr__(name: str):
    """Load one public scientific namespace lazily."""

    if name in _NAMESPACES:
        module = import_module(f"quantem.gpu.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
