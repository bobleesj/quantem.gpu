"""Collect HDF5 chunk locations without a Python callback per frame."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import h5py
import numpy as np
from h5py._objects import phil
from numba import carray, cfunc, types


@lru_cache(maxsize=1)
def _native_iterator():
    """Resolve symbols from h5py's own library, never a second HDF5 install."""
    try:
        library = ctypes.CDLL(h5py.h5d.__file__)
        iterate = library.H5Dchunk_iter
    except (OSError, AttributeError):
        return None
    iterate.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
    ]
    iterate.restype = ctypes.c_int
    signature = types.intc(
        types.CPointer(types.uint64), types.uintc,
        types.uint64, types.uint64, types.voidptr,
    )
    return library, iterate, cfunc(signature, cache=True)(_collect_chunk)


def _collect_chunk(offset, filter_mask, address, size, context):
    """Append to a capacity-checked array owned by the synchronous caller."""
    header = carray(context, (2,), dtype=np.uint64)
    capacity, count = header[0], header[1]
    if count >= capacity:
        return -1
    output = carray(context, (2 + 2 * capacity,), dtype=np.uint64)
    output[2 + 2 * count] = address
    output[3 + 2 * count] = size
    header[1] = count + 1
    return 0


def _chunk_locations_python(dataset: h5py.Dataset) -> np.ndarray:
    """Retain the h5py reference when direct library symbols are unavailable."""
    locations = []
    dataset.id.chunk_iter(
        lambda info: locations.append((info.byte_offset, info.size))
    )
    return np.asarray(locations, dtype=np.uint64).reshape(-1, 2)


def _chunk_locations(dataset: h5py.Dataset) -> np.ndarray:
    """Return allocated chunk offsets and sizes in HDF5 iteration order."""
    # Use h5py's lock because its linked HDF5 library need not be thread-safe.
    # It also serializes lazy callback compilation across simultaneous loads.
    with phil:
        native = _native_iterator()
        if native is None:
            return _chunk_locations_python(dataset)
        # Size for acquired chunks, not the potentially huge declared extent
        # of an unfinished scan. Enumeration performs no evidence reads.
        capacity = dataset.id.get_num_chunks()
        output = np.empty(2 + 2 * capacity, dtype=np.uint64)
        output[:2] = capacity, 0
        _, iterate, callback = native
        status = iterate(dataset.id.id, 0, callback.address, output.ctypes.data)
        if status != 0:
            raise OSError(
                f"Cannot enumerate HDF5 chunks in {dataset.name!r}; "
                "check that the source is complete and readable."
            )
        count = int(output[1])
        return output[2:2 + 2 * count].reshape(-1, 2)
