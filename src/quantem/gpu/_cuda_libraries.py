"""Load CuPy's pip-wheel CUDA libraries by absolute path before CuPy needs them.

``cupy-cuda12x`` from pip links CUDA 12 libraries (``libcufft.so.11``, ``libcublas.so.12``, ...) that live in the
``nvidia-*-cu12`` wheels under ``site-packages/nvidia/<name>/lib``; CuPy finds them through ``cuda.pathfinder`` when it
imports its FFT/BLAS modules. In an environment whose PyTorch uses conda's CUDA 13, importing torch first loads CUDA 13's
cuFFT; pathfinder then sees "cufft already loaded" and skips the CUDA 12 file, and CuPy's FFT fails with
``ImportError: libcufft.so.11: cannot open shared object file`` (every notebook that imports quantem.widget, which imports
torch, before running SSB). The two majors have different sonames, so both can be loaded: loading the wheel files by path,
globally, at quantem.gpu import makes CuPy's modules resolve against them whatever the import order.

Only the wheels that are installed are touched; conda-packaged CuPy (no nvidia-* wheels) is unaffected.
"""
from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

# dependency order: nvJitLink before cuSPARSE / cuSOLVER, cuBLAS before cuSOLVER
_WHEELS = ("nvjitlink", "cuda_nvrtc", "cublas", "cufft", "curand", "cusparse", "cusolver")


def preload() -> list[str]:
    """dlopen (RTLD_GLOBAL) every shared library of the installed ``nvidia.<name>`` CUDA wheels; returns the loaded paths."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    loaded = []
    for root in spec.submodule_search_locations:
        for name in _WHEELS:
            for path in sorted((Path(root) / name / "lib").glob("lib*.so.*")):
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(str(path))
                except OSError:
                    continue
    return loaded
