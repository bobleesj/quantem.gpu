"""GPU-accelerated scientific workflows for QuantEM."""

from importlib import import_module
from importlib.metadata import version

from quantem.gpu.device import cuda_runtime

# before anything imports CuPy's FFT/BLAS: see device.cuda_runtime (torch's CUDA 13 cuFFT otherwise shadows CuPy's CUDA 12 one)
cuda_runtime.preload_libraries()

from quantem.gpu.ssb import SSB, SSBResult

__version__ = version("quantem.gpu")


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
