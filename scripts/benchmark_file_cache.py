"""Per-file cache controls for disposable benchmark outputs on Linux and macOS.

Eviction is advisory. Confirm its effect with residency and physical-read
counters; neither this helper nor a first load alone proves cold storage.
"""

import ctypes
import math
import mmap
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _mapping(path: Path, protection: int, flags: int):
    """Map a nonempty file without touching its pages, then always unmap it."""
    library = ctypes.CDLL(None, use_errno=True)
    library.mmap.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int64,
    ]
    library.mmap.restype = ctypes.c_void_p
    library.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.munmap.restype = ctypes.c_int
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size <= 0:
            raise ValueError("Cache measurements require a nonempty file.")
        address = library.mmap(None, size, protection, flags, handle.fileno(), 0)
        if address == ctypes.c_void_p(-1).value:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(path))
        try:
            yield library, address, size
        finally:
            library.munmap(address, size)


def cache_state(path: Path) -> dict:
    """Return a file-page residency snapshot without reading the file body."""
    if sys.platform not in ("darwin", "linux"):
        return {"available": False, "reason": "mincore is supported on Linux and macOS only"}
    with _mapping(path, 0, mmap.MAP_PRIVATE) as (library, address, size):
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        pages = math.ceil(size / page_bytes)
        vector = (ctypes.c_ubyte * pages)()
        library.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        library.mincore.restype = ctypes.c_int
        if library.mincore(address, size, vector):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(path))
        resident = int(np.count_nonzero(np.ctypeslib.as_array(vector) & 1))
    return {
        "available": True, "bytes": size, "page_bytes": page_bytes,
        "pages": pages, "resident_pages": resident,
        "resident_percent": 100 * resident / pages,
    }


def evict_file_cache(path: Path) -> str:
    """Request eviction of one disposable output; never flush the global cache."""
    if sys.platform == "linux":
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return "posix_fadvise(POSIX_FADV_DONTNEED)"
    if sys.platform == "darwin":
        with _mapping(path, mmap.PROT_READ, mmap.MAP_SHARED) as (library, address, size):
            library.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            library.msync.restype = ctypes.c_int
            # Darwin sys/mman.h: MS_SYNC=0x10 and MS_INVALIDATE=0x02.
            if library.msync(address, size, 0x12):
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code), str(path))
        return "msync(MS_SYNC|MS_INVALIDATE), read-only shared mapping"
    raise NotImplementedError("Controlled file-cache eviction requires Linux or macOS.")
