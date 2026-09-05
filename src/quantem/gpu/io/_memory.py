"""Private reusable host staging and pinned-buffer ownership."""

from __future__ import annotations

import threading

import numpy as np

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
    mmaps a page-aligned anonymous region and cudaHostRegisters it once. The
    page lock permits asynchronous downstream H2D transfer. Reusing a
    compatible registered region amortizes that one-time registration cost.
    """
    with _PINNED_BUFS_LOCK:
        for entry in _PINNED_BUFS:
            if entry["free"] and nbytes <= entry["size"] <= int(nbytes * 1.5):
                entry["free"] = False
                return entry["arr"][:nbytes]
    import ctypes
    import mmap
    try:
        from cuda.bindings import runtime as cudart
    except ModuleNotFoundError:
        return np.empty(nbytes, dtype=np.uint8)
    registration_size = _pinned_registration_size(nbytes)
    region = mmap.mmap(-1, registration_size)  # anonymous → page-aligned base
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    err = cudart.cudaHostRegister(addr, registration_size, 0)
    if int(err[0]) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {int(err[0])}")
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
        released = None
        for entry in _PINNED_BUFS:
            if entry["arr"] is base:
                entry["free"] = True
                released = entry
                break
        if released is None:
            return

        if not prune:
            return

        _prune_pinned_free_locked()


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
        if _unregister_pinned_entry(entry):
            _PINNED_BUFS.remove(entry)
            entry.clear()


def _unregister_pinned_entry(entry: dict) -> bool:
    """Unregister one idle mmap-backed staging buffer."""
    try:
        from cuda.bindings import runtime as cudart
    except ModuleNotFoundError:
        return False
    error = cudart.cudaHostUnregister(entry["addr"])
    return int(error[0]) == 0
