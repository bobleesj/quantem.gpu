"""GPU-accelerated scientific workflows for QuantEM."""

from importlib import import_module
from importlib.metadata import version

from quantem.gpu.device import cuda_runtime

# before anything imports CuPy's FFT/BLAS: see device.cuda_runtime (torch's CUDA 13 cuFFT otherwise shadows CuPy's CUDA 12 one)
cuda_runtime.preload_libraries()

from quantem.gpu.direct_ptycho import DirectPtycho, SSB, SSBResult

__version__ = version("quantem.gpu")


__all__ = [
    "DirectPtycho",
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


# every public name but the SSB entry points is a subpackage, loaded on first use (movie needs the movie extra)
_NAMESPACES = set(__all__) - {"DirectPtycho", "SSB", "SSBResult", "__version__"}


def __getattr__(name: str):
    """Load one public scientific namespace lazily."""
    if name in _NAMESPACES:
        module = import_module(f"quantem.gpu.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
