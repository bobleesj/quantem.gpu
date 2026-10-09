"""Accelerator discovery and explicit backend resolution, for devices and for I/O (which also accepts the CPU reference)."""

import importlib.util
import os
import platform
import shutil
import sys
from typing import Literal, TypeAlias

DeviceName = Literal["cuda", "mps", "webgpu"]
NativeDeviceName = Literal["cuda", "mps"]

INSTALL_CUDA = 'pip install "quantem.gpu[cuda]"'
INSTALL_WIDGET_CUDA = 'pip install "quantem.widget[cuda]"'
INSTALL_MPS = 'pip install "quantem.gpu[mps]"'
INSTALL_WIDGET_MPS = 'pip install "quantem.widget[mps]"'
# Each line prints once per process: the notice once, each auto choice once.
_printed: set[str] = set()


def resolve_device(device: str | None = "auto") -> str:
    """Resolve ``device=`` to a Torch device string, the same way in every caller that accepts CPU.

    ``"auto"`` (or ``None``) picks the best available device, CUDA, then
    Apple MPS, then CPU, and prints one line naming the choice, once per
    process, so CPU is never selected silently. ``"cuda"``, ``"cuda:N"``,
    ``"mps"`` and ``"cpu"`` select that device and raise when it is not
    available. When a GPU is present but its runtime is not installed, one
    line names the ``pip`` command that enables it (see ``runtime_notice``).

    The GPU-only loaders keep ``resolve_backend``, whose ``"auto"`` never
    selects the CPU; this resolver is for computations that also run on CPU.

    Parameters
    ----------
    device : str or None, default "auto"
        ``"auto"``, ``"cuda"``, ``"cuda:N"``, ``"mps"`` or ``"cpu"``.

    Returns
    -------
    str
        ``"cuda:N"``, ``"mps"`` or ``"cpu"``.
    """
    _print_once(runtime_notice())
    resolved = str(profile(device)["device"])
    if device is None or str(device).strip().lower() in {"", "auto"}:
        _print_once(f'quantem.gpu: device="auto" selected {resolved}.')
    return resolved


def runtime_notice() -> str | None:
    """Return one line naming the missing GPU runtime and its install command, or None.

    An NVIDIA GPU (``nvidia-smi`` on the path or a loaded NVIDIA driver) is
    unused without CuPy, and an Apple-silicon GPU without PyObjC Metal and
    MLX; the ``[cuda]`` and ``[mps]`` extras install them. Without this line
    the package would quietly run on the CPU, or refuse, on a machine that has
    a GPU.
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


def release_cached_memory() -> None:
    """Return unused backend allocator blocks without releasing live arrays."""
    from quantem.gpu.device.cuda_runtime import cp

    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


def profile(device: str | None = None) -> dict[str, str | list[str] | None]:
    """Return a notebook-friendly summary of the active compute environment.

    With no argument, choose CUDA device 0, MPS, or CPU automatically and never
    raise when an accelerator is unavailable. Pass an explicit device such as
    ``"cuda:1"`` to select and validate another visible CUDA GPU. In a
    notebook, keep the returned ``device`` value as the single source of truth
    for later calls. ``available_devices`` lists privacy-safe device IDs so a
    multi-GPU machine is visible without exposing hardware names. Hostnames,
    executable paths, and detailed hardware names are intentionally omitted.

    Parameters
    ----------
    device : str or None, default None
        Requested Torch device. Use ``None`` or ``"auto"`` for automatic
        selection, ``"cuda"`` or ``"cuda:0"`` for the first visible CUDA GPU,
        ``"cuda:N"`` for another visible GPU, ``"mps"`` for Apple Metal, or
        ``"cpu"`` for CPU execution. CUDA indices refer to the devices visible
        to the process.
    """
    cuda_available = False
    cuda_count = 0
    mps_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        mps_available = bool(torch.backends.mps.is_available())
        torch_version = str(torch.__version__)
    except (ImportError, OSError, RuntimeError):
        # a missing or broken Torch install is reported as torch=None, never raised
        torch_version = None

    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"", "auto"} and cuda_available:
        backend = "cuda"
        resolved_device = "cuda:0"
    elif requested in {"", "auto"} and mps_available:
        backend = "mps"
        resolved_device = "mps"
    elif requested in {"", "auto", "cpu"}:
        backend = "cpu"
        resolved_device = "cpu"
    elif requested == "mps":
        if not mps_available:
            raise RuntimeError(
                'MPS device is unavailable; use device="auto" for automatic selection.'
            )
        backend = "mps"
        resolved_device = "mps"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(
                'CUDA device is unavailable; use device="auto" for automatic selection.'
            )
        if requested == "cuda":
            cuda_index = 0
        else:
            try:
                cuda_index = int(requested.removeprefix("cuda:"))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid CUDA device {device!r}; use 'cuda' or 'cuda:N', for example 'cuda:1'."
                ) from exc
        if cuda_index < 0 or cuda_index >= cuda_count:
            raise ValueError(
                f"CUDA device {device!r} is unavailable: {cuda_count} CUDA device(s) are visible. "
                f"Choose an index from 0 to {cuda_count - 1}."
            )
        backend = "cuda"
        resolved_device = f"cuda:{cuda_index}"
    else:
        raise ValueError(
            f"Unknown device {device!r}; use 'auto', 'cuda:N', 'mps', or 'cpu'."
        )

    if cuda_count:
        available_devices = [f"cuda:{index}" for index in range(cuda_count)]
    elif mps_available:
        available_devices = ["mps"]
    else:
        available_devices = ["cpu"]

    return {
        "platform": platform.platform(),
        "torch": torch_version,
        "backend": backend,
        "device": resolved_device,
        "available_devices": available_devices,
    }


def _cuda_probe() -> tuple[bool, str | None]:
    """Return whether CUDA is usable through CuPy, and why not when it is not.

    ``detect`` and ``resolve`` need the reason to explain an unavailable
    backend instead of failing on the first CuPy call.
    """
    try:
        cupy_spec = importlib.util.find_spec("cupy")
    except ModuleNotFoundError:
        cupy_spec = None
    if cupy_spec is None:
        return False, f"CuPy is not installed; install it with {INSTALL_CUDA}"
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
    except (ImportError, OSError, RuntimeError) as exc:
        # a CuPy build without a usable driver or device is reported, not raised
        return False, str(exc)
    if count < 1:
        return False, "CuPy imported, but no CUDA device is visible"
    return True, None


def _mps_probe() -> tuple[bool, str | None]:
    """Return whether Apple Metal is usable, through PyObjC Metal or Torch MPS, and why not."""
    if sys.platform != "darwin":
        return False, f"MPS requires macOS; current platform is {platform.system()}"
    if importlib.util.find_spec("Metal") is not None:
        return True, None
    try:
        import torch

        if bool(torch.backends.mps.is_available()):
            return True, None
        return False, "Torch is installed, but its MPS backend is unavailable"
    except (ImportError, OSError, RuntimeError) as exc:
        return False, str(exc)


def detect() -> NativeDeviceName:
    """Return the available native GPU backend.

    CUDA takes precedence when both runtimes are visible. CPU is intentionally
    not a scientific fallback. Browser WebGPU must be selected explicitly by
    the browser-facing caller.
    """
    _print_once(runtime_notice())
    cuda_available, cuda_error = _cuda_probe()
    if cuda_available:
        return "cuda"
    mps_available, mps_error = _mps_probe()
    if mps_available:
        return "mps"
    raise RuntimeError(
        f"No QuantEM GPU backend is available. CUDA: {cuda_error}. MPS: {mps_error}. "
        'Pass device="cpu" where a CPU path exists.'
    )


def resolve(name: str | None = "auto") -> DeviceName:
    """Validate and resolve a CUDA, MPS, or browser WebGPU backend name."""
    requested = "auto" if name is None else str(name).lower()
    if requested == "auto":
        return detect()
    _print_once(runtime_notice())
    if requested == "webgpu":
        return "webgpu"
    if requested == "cuda":
        available, error = _cuda_probe()
    elif requested == "mps":
        available, error = _mps_probe()
    else:
        raise ValueError(
            f"Unknown GPU backend {name!r}. Use 'auto', 'cuda', 'mps', or 'webgpu'."
        )
    if not available:
        raise RuntimeError(f"{requested.upper()} backend is unavailable: {error}")
    return requested


BackendName: TypeAlias = Literal["cuda", "mps", "cpu"]
_VALID: tuple[BackendName, ...] = ("cuda", "mps", "cpu")


def resolve_backend(backend: str | None) -> BackendName:
    """Validate an I/O backend, which may also be the CPU reference, or resolve ``None``/``auto``.

    ``auto`` selects an accelerator and never falls back to the CPU.
    """
    if backend in (None, "auto"):
        return detect()
    if backend not in _VALID:
        allowed = ", ".join(repr(name) for name in _VALID)
        raise ValueError(
            f"Unknown I/O backend {backend!r}. Use 'auto' or one of {allowed}."
        )
    if backend == "cpu":
        return backend
    return resolve(backend)
