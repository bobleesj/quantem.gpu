"""CUDA runtime plumbing every CUDA path shares: library preload, optional CuPy, pinned host buffers.

``preload_libraries`` loads CuPy's pip-wheel CUDA libraries by absolute path before CuPy needs them.
``cupy-cuda12x`` from pip links CUDA 12 libraries (``libcufft.so.11``, ``libcublas.so.12``, ...) that live in the
``nvidia-*-cu12`` wheels under ``site-packages/nvidia/<name>/lib``; CuPy finds them through ``cuda.pathfinder`` when it
imports its FFT/BLAS modules. In an environment whose PyTorch uses conda's CUDA 13, importing torch first loads CUDA 13's
cuFFT; pathfinder then sees "cufft already loaded" and skips the CUDA 12 file, and CuPy's FFT fails with
``ImportError: libcufft.so.11: cannot open shared object file`` (every notebook that imports quantem.widget, which imports
torch, before running SSB). The two majors have different sonames, so both can be loaded: loading the wheel files by path,
globally, at quantem.gpu import makes CuPy's modules resolve against them whatever the import order.
Only the wheels that are installed are touched; conda-packaged CuPy (no nvidia-* wheels) is unaffected.

``cp`` is CuPy, imported on first use: importing quantem.gpu never imports CuPy, so discovery, inspection and the
CPU and Metal paths run without it, and an installed but broken CUDA runtime fails only when a CUDA path runs.

The pinned-buffer functions own reusable page-locked host staging for asynchronous host-to-device transfers.
"""

import ctypes
import threading
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import numpy as np

# dependency order: nvJitLink before cuSPARSE / cuSOLVER, cuBLAS before cuSOLVER
_WHEELS = ("nvjitlink", "cuda_nvrtc", "cublas", "cufft", "curand", "cusparse", "cusolver")


def preload_libraries() -> list[str]:
    """dlopen (RTLD_GLOBAL) every shared library of the installed ``nvidia.<name>`` CUDA wheels; returns the loaded paths."""
    try:
        spec = find_spec("nvidia")
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


class _LazyCuPy:
    """CuPy's attributes, looked up in the imported ``cupy`` module at each use.

    Nothing is cached here, so the module ``sys.modules["cupy"]`` holds now (a
    test's stand-in, or CuPy with a patched function) is the one every call sees.
    """

    def __getattr__(self, name: str):
        return getattr(import_module("cupy"), name)


# Missing CuPy remains distinguishable from an installed but broken runtime.
# Runtime import failures propagate on use; they must not select a CPU fallback.
try:
    _available = find_spec("cupy") is not None
except ModuleNotFoundError:
    _available = False
cp = _LazyCuPy() if _available else None


_MODULES: dict[tuple, dict] = {}
_MODULES_LOCK = threading.Lock()


def cuda_module(source: str, names: tuple[str, ...], options: tuple[str, ...] = ()) -> dict:
    """Return the named kernels of CUDA ``source``, compiled once per device and context.

    Kernels compile on first use rather than at import, so importing the package
    needs no GPU. A compiled module belongs to the CUDA context it was loaded in,
    so the cache key holds the current device and context as well as the source
    and compiler options; switching devices compiles once more for the new one.
    """
    key = (*current_context(), source, names, options)
    with _MODULES_LOCK:
        functions = _MODULES.get(key)
        if functions is None:
            module = cp.RawModule(code=source, options=options)
            functions = _MODULES[key] = {name: module.get_function(name) for name in names}
    return functions


def current_context() -> tuple[int, int]:
    """Identify the current device and CUDA context, without changing devices."""
    if cp is None:
        raise ImportError(
            "CuPy is required for CUDA I/O. Install CuPy or select "
            "backend='cpu' or backend='mps'."
        )
    device = int(cp.cuda.runtime.getDevice())
    context = int(cp.cuda.driver.ctxGetCurrent())
    if not context:
        # First use may precede any array allocation. Initialize only the
        # already selected device; never replace an existing context.
        cp.cuda.runtime.free(0)
        context = int(cp.cuda.driver.ctxGetCurrent())
    if not context:
        raise RuntimeError(f"CUDA I/O could not initialize device {device}.")
    return device, context


_LIBC = None
_PINNED_BUFS: list[dict] = []
_PINNED_BUFS_LOCK = threading.Lock()
_PINNED_LARGE_BUFFER_THRESHOLD = 64 * 1024 * 1024
_PINNED_LARGE_BUFFER_GRANULARITY = 4 * 1024 * 1024


def _get_libc():
    """Lazy-load libc for posix_fadvise. None on non-Linux platforms."""
    global _LIBC
    if _LIBC is None:
        import ctypes
        import ctypes.util
        lib_name = ctypes.util.find_library("c")
        if lib_name is None:
            _LIBC = False
        else:
            try:
                libc = ctypes.CDLL(lib_name, use_errno=True)
            except OSError:
                _LIBC = False
            else:
                if not hasattr(libc, "posix_fadvise"):
                    _LIBC = False
                else:
                    _LIBC = libc
    return _LIBC if _LIBC is not False else None


def _pinned_registration_size(nbytes: int) -> int:
    """Return a bounded-capacity registration size for a requested buffer.

    Large sequential HDF5 batches from one source can differ by a fraction of
    a percent in compressed size. Registering their exact byte counts can make
    a later, slightly larger batch pay for a third page-locked buffer even
    though the pipeline has only two staging slots. Round large registrations
    to 4 MiB so nearby batches reuse those two slots. The extra pinned memory is
    bounded below 4 MiB per slot; small sparse selections keep exact sizing.
    """

    nbytes = int(nbytes)
    if nbytes <= 0:
        raise ValueError("Pinned buffer size must be positive")
    if nbytes < _PINNED_LARGE_BUFFER_THRESHOLD:
        return nbytes
    granularity = _PINNED_LARGE_BUFFER_GRANULARITY
    return ((nbytes + granularity - 1) // granularity) * granularity


def _alloc_pinned_fast(nbytes: int) -> np.ndarray:
    """Return a page-locked uint8 host buffer of length >= nbytes, view[:nbytes].

    Reuses a registered buffer from the free list when one fits (size within
    1.5x, so a 1024-scan buffer is not wasted on a 512-scan load); otherwise
    mmaps a page-aligned anonymous region and page-locks it once with CuPy's
    ``cudaHostRegister``. The page lock permits asynchronous downstream H2D
    transfer, and the page alignment lets direct I/O read into the buffer.
    Reusing a compatible registered region amortizes that one-time
    registration cost. Without CuPy (the Metal and CPU readers on a Mac)
    nothing can page-lock the buffer, so an ordinary array serves.
    """
    if cp is None:
        return np.empty(nbytes, dtype=np.uint8)
    with _PINNED_BUFS_LOCK:
        for entry in _PINNED_BUFS:
            if entry["free"] and nbytes <= entry["size"] <= int(nbytes * 1.5):
                entry["free"] = False
                return entry["arr"][:nbytes]
    import mmap
    registration_size = _pinned_registration_size(nbytes)
    region = mmap.mmap(-1, registration_size)  # anonymous: page-aligned base
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    cp.cuda.runtime.hostRegister(addr, registration_size, 0)
    arr = np.frombuffer(region, dtype=np.uint8)
    with _PINNED_BUFS_LOCK:
        _PINNED_BUFS.append(
            {
                "region": region,
                "addr": addr,
                "arr": arr,
                "size": registration_size,
                "free": False,
            }
        )
    return arr[:nbytes]


def _release_pinned(view: np.ndarray, *, prune: bool = True) -> None:
    """Mark a buffer from :func:`_alloc_pinned_fast` reusable.

    Keeps one reusable buffer per non-overlapping size class. A larger free
    buffer supersedes a smaller one when it is no more than 1.5x larger, which
    is the same fit rule used by :func:`_alloc_pinned_fast`. ``view`` is the
    sliced array; its ``.base`` is the full registered array we cached.

    Group-pipelined loads pass ``prune=False`` while disk preparation and GPU
    decode overlap. They prune once after the pipeline drains, so the next
    group can reuse every in-flight staging slot instead of repeatedly
    unregistering and registering a nearby-sized buffer.
    """
    base = view.base if view.base is not None else view
    with _PINNED_BUFS_LOCK:
        for entry in _PINNED_BUFS:
            if entry["arr"] is base:
                entry["free"] = True
                if prune:
                    _prune_pinned_free_locked()
                return


def _prune_pinned_free(*, retain_per_size_class: int = 1) -> None:
    """Discard redundant idle staging buffers after a pipeline drains."""
    with _PINNED_BUFS_LOCK:
        _prune_pinned_free_locked(retain_per_size_class=retain_per_size_class)


def _prune_pinned_free_locked(*, retain_per_size_class: int = 1) -> None:
    """Prune redundant free buffers while ``_PINNED_BUFS_LOCK`` is held."""
    retain_per_size_class = max(1, int(retain_per_size_class))
    retained: list[dict] = []
    redundant: list[dict] = []
    for entry in sorted(
        (item for item in _PINNED_BUFS if item["free"]),
        key=lambda item: item["size"],
        reverse=True,
    ):
        covering = sum(
            keeper["size"] <= int(entry["size"] * 1.5)
            for keeper in retained
        )
        if covering >= retain_per_size_class:
            redundant.append(entry)
        else:
            retained.append(entry)
    for entry in redundant:
        _unregister_pinned_entry(entry)
        _PINNED_BUFS.remove(entry)
        entry.clear()


def _unregister_pinned_entry(entry: dict) -> None:
    """Unregister one idle mmap-backed staging buffer before it is dropped."""
    cp.cuda.runtime.hostUnregister(entry["addr"])
