"""Private, explicit bridge to a pre-existing compact CUDA source.

This module neither loads data nor initializes CUDA. The supplied executor must
serialize every task on the source owner's existing context/thread, including
other native clients. It must drain a failed task before reporting failure.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from types import MappingProxyType

import numpy as np

_FAULTED_SOURCES: list[CudaResidentSource] = []


def _is_immutable_batch(
    value: object, shape: tuple[int, ...], dtype: np.dtype,
) -> bool:
    """Recognize a complete ndarray owned entirely by one immutable bytes object."""
    if (type(value) is not np.ndarray or value.shape != shape
            or value.dtype != dtype or not value.flags.c_contiguous
            or value.flags.writeable):
        return False
    owner = value
    while type(owner) is np.ndarray:
        if owner.flags.writeable:
            return False
        owner = owner.base
    return type(owner) is bytes and len(owner) == value.nbytes


class CudaResidentSource:
    """Borrow an existing exact uint16 source through its owner-thread executor.

    Parameters
    ----------
    owner
        Strong reference to the existing scientific allocation owner.
    shape
        Logical ``(acquisition, scan_row, scan_col, detector_row, detector_col)``.
    device
        Actual source device, for example ``'cuda:0'`` within its process.
    generation
        Immutable source-generation token, distinct from display request IDs.
    storage_format, source_codec, sparse_codec
        Explicit scientific format and dense/sparse codec version identifiers
        attested by the existing owner. These are distinct from bridge version.
    resident_bytes
        Actual compact source/index bytes retained by the supplied owner.
    valid_pixels
        Original detector-validity mask; raw diffraction counts stay untouched.
    execute
        Synchronous owner-thread dispatcher. Run the supplied zero-argument
        task on the existing CUDA context, serialize native users, and return
        its result. Never run scientific tasks on a new widget CUDA thread.
    integrate, patterns
        Callbacks ``integrate(owner, mask)`` and ``patterns(owner, flat_scan)``.
        Each must calculate the full acquisition batch in one backend call.
        Returned device arrays remain borrowed until the task's blocking host
        copy completes; neither callback may release or replace source owners.
        Complete host arrays backed entirely by immutable bytes are retained
        directly; other host arrays are snapshotted before the task completes.
    """

    ndim = 5
    dtype = np.dtype("<u2")
    representation = "compact"
    _is_gpu_frames = True
    _quantem_cuda_resident_version = 1

    def __init__(
        self, owner: object, *, shape: tuple[int, int, int, int, int],
        device: str, generation: str, storage_format: str, source_codec: str,
        sparse_codec: str, resident_bytes: int,
        valid_pixels: np.ndarray, execute: Callable[[Callable[[], object]], object],
        integrate: Callable[[object, np.ndarray], object],
        patterns: Callable[[object, int], object],
    ) -> None:
        if len(shape) != 5 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("Supply all five positive logical resident dimensions.")
        if shape[-2] * shape[-1] * 65535 > np.iinfo(np.uint32).max:
            raise ValueError("This uint32 batch bridge cannot bound exact detector sums for this shape.")
        if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("Supply the existing owner's explicit CUDA device, e.g. 'cuda:0'.")
        if not isinstance(generation, str) or not generation:
            raise ValueError("Supply a nonempty source-generation token.")
        storage_metadata = dict(
            storage_format=storage_format, source_codec=source_codec,
            sparse_codec=sparse_codec,
        )
        if any(not isinstance(value, str) or not value.strip()
               for value in storage_metadata.values()):
            raise ValueError("Supply the actual scientific storage format and both codec versions.")
        if type(resident_bytes) is not int or resident_bytes <= 0:
            raise ValueError("Supply the actual positive compact resident byte count.")
        valid = np.asarray(valid_pixels)
        if valid.shape != tuple(shape[-2:]) or valid.dtype != np.bool_:
            raise ValueError("valid_pixels must be the original boolean detector-validity array.")
        if not all(callable(value) for value in (execute, integrate, patterns)):
            raise TypeError("Supply the existing owner executor and both all-acquisition callbacks.")
        self.shape, self.device, self.generation = tuple(shape), device, generation
        self.storage_metadata = MappingProxyType(storage_metadata)
        self.nbytes = resident_bytes
        self.valid_pixels = np.frombuffer(valid.tobytes(), dtype=np.bool_).reshape(valid.shape)
        self._owner, self._execute = owner, execute
        self._integrate, self._patterns = integrate, patterns
        self._lock = threading.RLock()
        self._failure: BaseException | None = None
        self._views: dict[int, _ResidentAcquisition] = {}

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Compact data cannot be materialized implicitly; use an exact detector session or source archive.")

    def numel(self) -> int:
        """Logical native count, without materializing any source values."""
        return int(np.prod(self.shape, dtype=np.int64))

    def element_size(self) -> int:
        return self.dtype.itemsize

    def __getitem__(self, index: int) -> _ResidentAcquisition:
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < self.shape[0]:
            raise IndexError("Select one valid acquisition index; full compact materialization is unsupported.")
        index = int(index)
        with self._lock:
            if index not in self._views:
                self._views[index] = _ResidentAcquisition(self, index)
            return self._views[index]

    def mask(self, value) -> np.ndarray:
        """Validate a binary aperture and retain the original validity semantics."""
        value = np.asarray(value)
        if value.shape != self.shape[-2:] or not np.all((value == 0) | (value == 1)):
            raise ValueError("Use a full-resolution binary detector mask; weighted detectors are unsupported.")
        return np.ascontiguousarray(value.astype(np.bool_) & self.valid_pixels, dtype=np.uint8)

    def _calculate(self, operation: str, argument, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("The resident source executor failed; recover it before issuing more requests.") from self._failure

            def task():
                callback = self._integrate if operation == "integrate" else self._patterns
                value = callback(self._owner, argument)
                if (tuple(value.shape) != shape or np.dtype(value.dtype) != dtype
                        or not value.flags.c_contiguous):
                    raise ValueError(f"Resident {operation} must return the complete {shape} {dtype.str} batch.")
                if not isinstance(value, np.ndarray) and f"cuda:{value.device.id}" != self.device:
                    raise ValueError("The callback returned a batch from a different CUDA owner device.")
                # An immutable transport response already owns all of its bytes.
                if _is_immutable_batch(value, shape, dtype):
                    return value
                # Snapshot mutable host output while the owner is serialized.
                # CuPy get(blocking=True) fences the copy before output reuse.
                if isinstance(value, np.ndarray):
                    host = np.array(value, copy=True, order="C", subok=False)
                else:
                    host = value.get(blocking=True)
                if (not isinstance(host, np.ndarray) or host.shape != shape
                        or host.dtype != dtype or not host.flags.c_contiguous):
                    raise ValueError(f"Resident {operation} must return the complete {shape} {dtype.str} batch.")
                # Immutable bytes own this result independently of GPU output reuse.
                return np.frombuffer(host.tobytes(), dtype=dtype).reshape(shape)

            try:
                result = self._execute(task)
                if not _is_immutable_batch(result, shape, dtype):
                    raise ValueError("The owner executor did not return the completed immutable batch.")
                return result
            except BaseException as error:
                # Keep the exception traceback, callback locals and source owner
                # alive even if a dispatcher could not drain a CUDA failure.
                self._failure = error
                if not any(source is self for source in _FAULTED_SOURCES):
                    _FAULTED_SOURCES.append(self)
                raise

    def masked_sum_batch_exact(self, mask) -> np.ndarray:
        """Compute all exact uint32 virtual images in one backend invocation."""
        return self._calculate("integrate", self.mask(mask), self.shape[:3], np.dtype("<u4"))

    def frame_batch(self, index: int) -> np.ndarray:
        """Decode the same scan position across all acquisitions as exact uint16."""
        if (not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
                or not 0 <= index < self.shape[1] * self.shape[2]):
            raise IndexError("Use a flat scan index inside the complete native scan.")
        return self._calculate("patterns", int(index), (self.shape[0], *self.shape[-2:]), self.dtype)


class _ResidentAcquisition:
    """Logical acquisition view; no payload slicing or expansion occurs."""
    _is_gpu_frames = True
    _quantem_cuda_resident_version = 1
    ndim = 3

    def __init__(self, source: CudaResidentSource, index: int):
        self.source, self.index = source, index
        self.shape = (source.shape[1] * source.shape[2], *source.shape[-2:])
        self.dtype, self.device = source.dtype, source.device

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Use detector.prepare(view); compact acquisition expansion is unsupported.")

    def __getitem__(self, index: int) -> np.ndarray:
        return self.source.frame_batch(index)[self.index]


def is_cuda_resident(value) -> bool:
    """Recognize only the explicit owner/view, before generic chunk dispatch."""
    return isinstance(value, (CudaResidentSource, _ResidentAcquisition))


class _ResidentBackend:
    """Prepared detector adapter without importing any accelerator runtime."""
    capabilities = ("exact_batch",)

    def __init__(self, data):
        self.source = data.source if isinstance(data, _ResidentAcquisition) else data
        self.acquisition = data.index if isinstance(data, _ResidentAcquisition) else None
        self.scan_shape = self.source.shape[1:3]
        self.det_shape = self.source.shape[-2:]
        self.n_frames = int(np.prod(self.scan_shape))
        self.device = self.source.device

    def masked_sum_batch_exact(self, mask):
        return self.source.masked_sum_batch_exact(mask)

    def frame_batch(self, index):
        return self.source.frame_batch(index)

    def _selected(self):
        if self.acquisition is None:
            raise ValueError("Use the all-acquisition batch method or prepare(source[acquisition]) for one view.")
        return self.acquisition

    def frame(self, index):
        selected = self._selected()
        return self.frame_batch(index)[selected]

    def masked_sum_exact(self, mask):
        selected = self._selected()
        return self.masked_sum_batch_exact(mask)[selected].astype(np.uint64)

    def masked_sum(self, mask):
        return self.masked_sum_exact(mask).astype(np.float32)

    def mean_dp(self):
        raise NotImplementedError("Full-scan mean DP is not implemented for this compact owner; use frame_batch for exact point patterns.")

    def reduce_frames(self, indices, reduce="mean"):
        raise NotImplementedError("Multi-position DP reductions are not implemented for this compact owner; source counts remain resident.")

    def center_of_mass(self, mask=None):
        raise NotImplementedError("CoM is not implemented for this compact owner.")
