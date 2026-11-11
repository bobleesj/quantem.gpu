"""Accelerator discovery and the one ``device=`` resolver: CUDA or Apple MPS, never the CPU."""

import gc
import importlib.util
import os
import platform
import shutil
import sys

import torch

INSTALL_CUDA = 'pip install "quantem.gpu[cuda]"'
INSTALL_WIDGET_CUDA = 'pip install "quantem.widget[cuda]"'
INSTALL_MPS = 'pip install "quantem.gpu[mps]"'
INSTALL_WIDGET_MPS = 'pip install "quantem.widget[mps]"'
# The missing-runtime notice prints once per process.
_printed: set[str] = set()


def runtime_notice() -> str | None:
    """Return one line naming the missing GPU runtime and its install command, or None.

    An NVIDIA GPU (``nvidia-smi`` on the path or a loaded NVIDIA driver) is
    unused without CuPy, and an Apple-silicon GPU without PyObjC Metal and
    MLX; the ``[cuda]`` and ``[mps]`` extras install them. Without this line
    the package would refuse a machine that has a GPU without saying why.
    """
    if sys.platform == "darwin":
        if platform.machine() != "arm64":
            return None
        missing = [
            package
            for module, package in (("Metal", "pyobjc-framework-Metal"), ("mlx", "mlx"))
            if importlib.util.find_spec(module) is None
        ]
        if not missing:
            return None
        gpu, runtime = "this Mac's Apple GPU", " and ".join(missing)
        install, widget = INSTALL_MPS, INSTALL_WIDGET_MPS
    else:
        if shutil.which("nvidia-smi") is None and not os.path.exists("/proc/driver/nvidia"):
            return None
        if importlib.util.find_spec("cupy") is not None:
            return None
        gpu, runtime = "an NVIDIA GPU", "CuPy"
        install, widget = INSTALL_CUDA, INSTALL_WIDGET_CUDA
    return (
        f"quantem.gpu: {gpu} is present but {runtime} is not installed, so the GPU is not used. "
        f"Install it with: {install} (widget: {widget})"
    )


def _print_once(line: str | None) -> None:
    """Print ``line`` the first time it occurs in this process."""
    if line is not None and line not in _printed:
        _printed.add(line)
        print(line)


def release_cached_memory(verbose: bool = False) -> float:
    """Return the GPU memory this process caches but no longer uses to the driver; returns the gigabytes released.

    Call it after dropping every reference to the arrays (``del`` them, ``close()`` the acquisition): live arrays
    are never touched. The result is ``(held before - held after) / 1e9``, where held is the memory this process
    has taken from the driver: on CUDA the bytes Torch's allocator has reserved plus the bytes of CuPy's pool,
    summed over the GPUs the process has opened, and on Apple silicon the Metal driver's total for the process.
    Both include cached blocks, so the result is what went back. Torch's MPS tensor counter excludes them and
    read 0 GB while gigabytes went back. ``cudaMemGetInfo`` counts every process on a GPU: over calls that
    returned 2.1 GB it read between -13.2 and 2.2 GB while other jobs were allocating, so it is printed, not
    returned.

    Why each step is needed: garbage is collected first because a reference cycle (a widget and its observers)
    keeps arrays alive after ``del``. Torch's caching allocator, CuPy's memory pool and MLX keep freed blocks for
    reuse, so a dropped acquisition still counts as used and the next load can run out of memory until each
    cache is emptied. CuPy's pool counts and frees the blocks of its current GPU only, so each opened GPU is
    visited, and its cached FFT plans are dropped first because a plan keeps its work area in the pool. CuPy's
    pinned pool holds page-locked host buffers of the transfers: host memory, returned here and not part of the
    result. A GPU the process has not opened is not visited: it holds none of its blocks, and visiting it would
    open it (550 MiB of driver context, or an error when that GPU cannot be opened).

    What stays: arrays something still refers to, including the locals of a failed call that the traceback of
    the last error keeps (``sys.last_exc`` in a notebook), and the FFT plans of other threads. On CUDA it needs
    Torch built with CUDA; beside a CPU-only Torch, CuPy's pool is left alone.

    Parameters
    ----------
    verbose : bool, default False
        Print one line per opened GPU: the gigabytes this process released, and the GPU's memory in use and free
        as the driver reports them for all processes. On Apple silicon the line has no free value.

    Returns
    -------
    float
        Gigabytes (1e9 bytes) this process returned to the driver; 0.0 without a GPU.

    Examples
    --------
    >>> from quantem.gpu import device
    >>> data.close()
    >>> del data
    >>> device.release_cached_memory(verbose=True)
    GPU 0: released 2.1 GB, 32.0 GB in use, 69.9 GB free
    2.147483648
    """
    if torch.cuda.is_available():
        from quantem.gpu.device.cuda_runtime import cp

        gc.collect()
        # torch has no public call that says whether this process holds a context on a GPU
        devices = [index for index in range(torch.cuda.device_count()) if torch._C._cuda_hasPrimaryContext(index)]
        reserved_before = {index: torch.cuda.memory_reserved(index) for index in devices}
        torch.cuda.empty_cache()  # one call covers every GPU; CuPy's caches below need a visit per GPU
        released = {index: before - torch.cuda.memory_reserved(index) for index, before in reserved_before.items()}
        if cp is not None:
            pool = cp.get_default_memory_pool()
            for index in devices:
                with cp.cuda.Device(index):
                    pool_before = pool.total_bytes()
                    plans = cp.fft.config.get_plan_cache()
                    if plans.get_curr_size():  # clear() runs a garbage collection of its own, 80 ms even when empty
                        plans.clear()
                    pool.free_all_blocks()
                    released[index] += pool_before - pool.total_bytes()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        if verbose:
            for index in devices:
                free, total = torch.cuda.mem_get_info(index)
                print(
                    f"GPU {index}: released {released[index] / 1e9:.1f} GB, "
                    f"{(total - free) / 1e9:.1f} GB in use, {free / 1e9:.1f} GB free"
                )
        return sum(released.values()) / 1e9
    if torch.backends.mps.is_available():
        before = torch.mps.driver_allocated_memory()
        gc.collect()  # after the reading: a collected Metal resident hands its buffers straight back to the driver
        torch.mps.empty_cache()
        # the Metal encode and direct ptychography paths compute with MLX, which keeps its own cache; never imported, it holds nothing
        mlx_core = sys.modules.get("mlx.core")
        if mlx_core is not None:
            mlx_core.clear_cache()
        in_use = torch.mps.driver_allocated_memory()
        if verbose:
            print(f"MPS: released {(before - in_use) / 1e9:.1f} GB, {in_use / 1e9:.1f} GB in use")
        return (before - in_use) / 1e9
    if verbose:
        print("No CUDA or MPS device in Torch: nothing to release.")
    return 0.0


def resolve(device: int | str | None = "auto") -> str:
    """Resolve ``device=`` to the GPU a call runs on: ``"cuda"``, ``"cuda:N"`` or ``"mps"``, never the CPU.

    Every ``device=`` and ``backend=`` argument of the package goes through this one function, so one spelling
    names the same GPU everywhere. ``"auto"`` (or ``None``) takes CUDA when CuPy sees a device, else Apple Metal,
    and raises when neither works: the CPU is not a scientific backend, so it is never chosen, silently or on
    request. ``"cuda"`` keeps no index, meaning the current CUDA device; ``"cuda:N"`` or the integer ``N`` names
    visible CUDA device N (on a Mac without CUDA, ``0`` is the one Apple GPU); ``"mps"`` is the Apple GPU. The
    result is a Torch device string: ``torch.device(resolved).type`` is the backend and ``.index`` the CUDA device,
    or None. When a GPU is present but its runtime is not installed, one line names the ``pip`` command that
    enables it (see ``runtime_notice``).

    Examples
    --------
    >>> from quantem.gpu import device
    >>> device.resolve("auto")
    'cuda'
    >>> device.resolve(1)
    'cuda:1'
    """
    if isinstance(device, int) and not isinstance(device, bool):
        # device=0 or 1 names a visible GPU: CUDA device N, or on a Mac the one Apple GPU
        device = "mps" if device == 0 and not _cuda_probe()[0] and _mps_probe()[0] else f"cuda:{device}"
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "cpu":
        raise ValueError(
            "quantem.gpu runs on CUDA or Apple MPS only; device='cpu' is not supported. "
            "Omit device to use this machine's GPU."
        )
    if requested not in ("", "auto", "cuda", "mps") and not requested.startswith("cuda:"):
        raise ValueError(f"Unknown device {device!r}; use 'auto', a GPU number (0, 1), 'cuda', 'cuda:N' or 'mps'.")
    _print_once(runtime_notice())
    if requested == "mps":
        # an explicit Apple GPU never probes CuPy, so choosing MPS starts no CUDA context
        available, error = _mps_probe()
        if not available:
            raise RuntimeError(f"MPS backend is unavailable: {error}")
        return "mps"
    cuda_count, cuda_error = _cuda_probe()
    if requested in ("", "auto"):
        if cuda_count:
            return "cuda"
        mps_available, mps_error = _mps_probe()
        if mps_available:
            return "mps"
        raise RuntimeError(f"No QuantEM GPU backend is available. CUDA: {cuda_error}. MPS: {mps_error}.")
    if not cuda_count:
        raise RuntimeError(f"CUDA backend is unavailable: {cuda_error}")
    if requested == "cuda":
        return "cuda"
    try:
        index = int(requested.removeprefix("cuda:"))
    except ValueError as exc:
        raise ValueError(f"Invalid CUDA device {device!r}; use 'cuda' or 'cuda:N', for example 'cuda:1'.") from exc
    if not 0 <= index < cuda_count:
        raise ValueError(
            f"CUDA device {device!r} is unavailable: {cuda_count} CUDA device(s) are visible. "
            f"Choose an index from 0 to {cuda_count - 1}."
        )
    return f"cuda:{index}"


def _cuda_probe() -> tuple[int, str | None]:
    """Return how many CUDA devices CuPy sees (0 when CUDA is unusable), and why none when there are none.

    ``resolve`` needs the reason to explain an unavailable backend instead of failing on the first CuPy call,
    and the count to reject a device index that is not visible.
    """
    try:
        cupy_spec = importlib.util.find_spec("cupy")
    except ModuleNotFoundError:
        cupy_spec = None
    if cupy_spec is None:
        return 0, f"CuPy is not installed; install it with {INSTALL_CUDA}"
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
    except (ImportError, OSError, RuntimeError) as exc:
        # a CuPy build without a usable driver or device is reported, not raised
        return 0, str(exc)
    if count < 1:
        return 0, "CuPy imported, but no CUDA device is visible"
    return count, None


def _mps_probe() -> tuple[bool, str | None]:
    """Return whether Apple Metal is usable, through PyObjC Metal or Torch MPS, and why not."""
    if sys.platform != "darwin":
        return False, f"MPS requires macOS; current platform is {platform.system()}"
    if importlib.util.find_spec("Metal") is not None:
        return True, None
    try:
        if bool(torch.backends.mps.is_available()):
            return True, None
        return False, "Torch is installed, but its MPS backend is unavailable"
    except (OSError, RuntimeError) as exc:
        return False, str(exc)
