"""Which acquisitions are held on which GPU of the pool, and admission of new ones.

Each acquisition is loaded once, encoded, wholly onto one GPU; every bin and
crop a client asks for is computed from that one resident. Splitting one
interactive acquisition across GPUs would add synchronization to every
request, so separate acquisitions are spread across the pool instead.
Admission picks the GPU, evicts the least recently used residents until the
load fits, and retries after an out-of-memory error.
"""

import math
import threading
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from quantem.gpu import detector, io
from quantem.gpu.detector import DetectorSession
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.remote.catalog import Catalog, file_signature, format_size
from quantem.gpu.remote.plan import BrowsePlan

# Share of each GPU's memory the resident cache may hold.
CACHE_FRACTION = 0.80
# Allocator headroom kept free beyond a load's expected peak.
LOAD_HEADROOM_BYTES = 1 << 30
MAX_ENTRIES_PER_GPU = 8


@dataclass(frozen=True)
class Gpu:
    """One CUDA device of the pool and the share of its memory the cache may hold.

    ``device`` is the ``cupy.cuda.Device`` that selects the GPU in a worker
    thread.
    """

    index: int
    device: object
    name: str
    total_memory_bytes: int
    cache_budget_bytes: int


@dataclass(eq=False)
class ResidentAcquisition:
    """One acquisition held encoded on one GPU, read by every plan of it.

    ``pins`` counts requests computing on it; eviction waits for zero.
    ``derived`` caches what a plan derives from the resident, the fitted
    bright-field geometry and the centre-of-mass maps, so repeated images skip
    that work.
    """

    key: str
    gpu: int
    loaded: Dataset4dstemGPU
    session: DetectorSession
    resident_bytes: int
    source_signature: tuple[tuple[str, int, int], ...]
    pins: int = 0
    closed: bool = False
    derived: dict[tuple[BrowsePlan, str], object] = field(default_factory=dict)


def open_gpus(requested: tuple[int, ...] | None) -> tuple[list[Gpu], str | None]:
    """Open every requested CUDA device, or every visible one for None, and size its cache.

    A device that fails to initialize is left out of the pool and its error
    reported, so one bad GPU does not stop the service; with no working
    device the pool is empty and the error says why.

    Returns
    -------
    tuple
        The pool, and the errors of the devices left out (None when every
        device opened).
    """
    if cp is None:
        return [], "ImportError: CuPy is not installed; install quantem.gpu[cuda,remote]"
    try:
        indices = requested or tuple(range(int(cp.cuda.runtime.getDeviceCount())))
    except RuntimeError as exc:
        return [], f"{type(exc).__name__}: {exc}"
    gpus: list[Gpu] = []
    errors: list[str] = []
    for index in indices:
        try:
            device = cp.cuda.Device(index)
            with device:
                properties = cp.cuda.runtime.getDeviceProperties(index)
                # CuPy's first reduction keeps a reference to its input;
                # spend it on a throwaway array so evicted data can free.
                cp.zeros((1,), dtype=cp.uint8).sum()
                cp.get_default_memory_pool().free_all_blocks()
        except (RuntimeError, MemoryError) as exc:
            errors.append(f"GPU {index}: {type(exc).__name__}: {exc}")
            continue
        total = int(properties["totalGlobalMem"])
        gpus.append(
            Gpu(
                index,
                device,
                properties["name"].decode("utf-8", errors="replace"),
                total,
                int(total * CACHE_FRACTION),
            )
        )
    if not gpus:
        return [], "RuntimeError: " + ("; ".join(errors) or "no CUDA devices were found")
    return gpus, "; ".join(errors) or None


class Residency:
    """Hold acquisitions on the GPUs of one pool and admit new ones within each GPU's budget.

    Parameters
    ----------
    catalog : Catalog
        Inspects the acquisitions to size them before loading.
    gpus : list of Gpu
        The pool, from :func:`open_gpus`.
    """

    def __init__(self, catalog: Catalog, gpus: list[Gpu]) -> None:
        self.catalog = catalog
        self.gpus = {gpu.index: gpu for gpu in gpus}
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._load_lock = threading.Lock()
        # Numba's parallel header parser creates one native worker pool per
        # calling thread. Keep every serialized load on the same reusable
        # thread so long-running HTTP services do not accumulate another pool
        # whenever Starlette chooses a new worker.
        self._load_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="quantem-cuda-load",
        )
        self._entries: OrderedDict[str, ResidentAcquisition] = OrderedDict()
        self._active_key: str | None = None
        self._compute_locks: dict[str, threading.Lock] = {}
        self._compute_locks_lock = threading.Lock()

    def devices(self) -> list[dict[str, object]]:
        """Describe every GPU of the pool with the live capacity admission uses."""
        with self._lock:
            return [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_bytes": gpu.total_memory_bytes,
                    "resident_entries": self._resident_entries(gpu.index),
                    **self._capacity(gpu.index),
                }
                for gpu in self.gpus.values()
            ]

    def entry(self, path: Path, *, reserve: bool = False) -> ResidentAcquisition:
        """Return the acquisition's resident, loading it on the best GPU when absent.

        ``reserve`` pins the entry until :meth:`release`.
        """
        with self._lock:
            cached = self._reuse(path, reserve=reserve)
            if cached is not None:
                return cached
        resident_bytes, peak_bytes = self._expected_bytes(path)
        largest_budget = max((gpu.cache_budget_bytes for gpu in self.gpus.values()), default=0)
        if peak_bytes > largest_budget:
            raise HTTPException(
                413,
                f"Loading this acquisition needs about {format_size(peak_bytes)} "
                "of GPU memory, but the largest per-GPU data-cache budget is "
                f"{format_size(largest_budget)}. Serve it from a GPU with more memory.",
            )
        with self._load_lock:
            with self._lock:
                cached = self._reuse(path, reserve=reserve)
                if cached is not None:
                    return cached
                preserve_key = self._active_key
                if not self._candidate_gpus(resident_bytes, peak_bytes):
                    preserve_key = None
                    if not self._candidate_gpus(resident_bytes, peak_bytes, preserve_active=False):
                        raise HTTPException(
                            413,
                            "No configured CUDA GPU can admit this acquisition. Close "
                            "other GPU work or serve it from a GPU with more memory.",
                        )
            loaded = self._admit(path, resident_bytes, peak_bytes, preserve_key)
            if loaded is None:
                raise HTTPException(
                    413,
                    "The acquisition did not fit on any configured CUDA GPU. Close "
                    "other GPU work or serve it from a GPU with more memory.",
                )
            with self._lock:
                self._evict_for(loaded.gpu, loaded.resident_bytes, preserve=preserve_key)
                self._entries[loaded.key] = loaded
                self._entries.move_to_end(loaded.key)
                if reserve:
                    loaded.pins += 1
            return loaded

    def resident(self, path: Path) -> ResidentAcquisition:
        """Return the acquisition's resident without loading it, or refuse with 409."""
        with self._lock:
            entry = self._entries.get(str(path))
        if entry is None:
            raise HTTPException(
                409,
                "This acquisition is not resident; load a virtual image first.",
            )
        return entry

    def release(self, entry: ResidentAcquisition) -> None:
        """Release one reservation taken by :meth:`entry` or :meth:`pinned`."""
        with self._lock:
            if entry.pins <= 0:
                raise RuntimeError("resident entry reservation was released twice")
            entry.pins -= 1
            self._changed.notify_all()

    @contextmanager
    def pinned(self, entry: ResidentAcquisition) -> Iterator[ResidentAcquisition]:
        """Pin one resident acquisition and select its GPU while a result is computed.

        Calculations on one acquisition run one at a time; the pin keeps
        eviction from closing it underneath them.
        """
        with self._lock:
            if self._entries.get(entry.key) is not entry:
                raise HTTPException(
                    409,
                    "This acquisition is no longer resident; load a virtual image first.",
                )
            lock = self._entry_lock(entry.key)
            self._entries.move_to_end(entry.key)
            entry.pins += 1
        lock.acquire()
        try:
            if entry.closed:
                raise HTTPException(
                    409,
                    "This acquisition is no longer resident; load a virtual image first.",
                )
            with self.gpus[entry.gpu].device:
                yield entry
        finally:
            lock.release()
            self.release(entry)

    def activate(self, key: str) -> None:
        """Mark a still-resident acquisition as the one eviction keeps longest."""
        with self._lock:
            if key in self._entries:
                self._active_key = key
                self._entries.move_to_end(key)

    def close(self) -> None:
        """Stop loading and release every resident acquisition and pooled device block."""
        self._load_executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._close_entry(entry)
        for gpu in self.gpus:
            self._flush(gpu)

    # --- Admission

    def _expected_bytes(self, path: Path) -> tuple[int, int]:
        """Return the (resident, peak) device bytes expected for loading one acquisition.

        Admission and eviction use these to pick a GPU and to refuse a load
        that cannot fit, instead of failing halfway through on the device.
        The request's plan has already checked that the master reports its
        shape. The encoded size is known only once the counts are encoded.
        Detector counts take fewer bytes in the resident code than in the
        stored bitshuffle-LZ4 files (0.73 of the stored bytes on a 256 x 256 x
        192 x 192 Arina scan) and about the dense size when the files are not
        compressed, so the smaller of the two is reserved; the measured size
        replaces it once loaded. The loader also needs 128 MiB plus 64 bytes
        per detector pixel for each of at least 512 staged scan positions.
        """
        inspection = self.catalog.inspect(path)
        positions = math.prod(int(value) for value in inspection.scan_shape)
        pixels = math.prod(int(value) for value in inspection.detector_shape)
        # uint32 Arina counts are encoded as uint16 once every count fits.
        itemsize = 1 if np.dtype(inspection.dtype) == np.dtype(np.uint8) else 2
        source_files = (inspection.source_signature or {}).get("files", [])
        stored_bytes = sum(int(item.get("size", 0)) for item in source_files)
        dense_bytes = positions * pixels * itemsize
        resident_bytes = min(stored_bytes, dense_bytes) if stored_bytes else dense_bytes
        staging_bytes = (128 << 20) + min(512, positions) * pixels * 64
        return resident_bytes, resident_bytes + staging_bytes

    def _capacity(self, gpu: int, *, preserve_active: bool = True) -> dict[str, int | None]:
        """Return the live capacity of one GPU; reporting and admission share it so they cannot drift.

        Call with the lock held. Peak capacity counts evictable residents as
        free memory; resident capacity keeps room for the active acquisition.
        """
        budget = self.gpus[gpu].cache_budget_bytes
        resident = self._resident_bytes(gpu)
        active = self._entries.get(self._active_key) if preserve_active else None
        active_resident = active.resident_bytes if active is not None and active.gpu == gpu else 0
        evictable = max(0, resident - active_resident)
        free = self._free_bytes(gpu)
        return {
            "cache_budget_bytes": budget,
            "free_bytes": free,
            "resident_bytes": resident,
            "active_resident_bytes": active_resident,
            "evictable_bytes": evictable,
            "available_peak_bytes": budget if free is None else min(budget, max(0, free + evictable)),
            "available_resident_bytes": max(0, budget - active_resident),
        }

    def _candidate_gpus(
        self,
        resident_bytes: int,
        peak_bytes: int,
        *,
        preserve_active: bool = True,
    ) -> list[int]:
        """Return the GPUs that can admit an acquisition, the least loaded first. Call with the lock held."""
        candidates: list[tuple[int, int, float, int]] = []
        for gpu in self.gpus:
            capacity = self._capacity(gpu, preserve_active=preserve_active)
            if (
                peak_bytes > capacity["available_peak_bytes"]
                or resident_bytes > capacity["available_resident_bytes"]
            ):
                continue
            pressure = capacity["resident_bytes"] / capacity["cache_budget_bytes"]
            candidates.append(
                (self._resident_entries(gpu), -capacity["available_peak_bytes"], pressure, gpu)
            )
        candidates.sort()
        return [candidate[3] for candidate in candidates]

    def _admit(
        self,
        path: Path,
        resident_bytes: int,
        peak_bytes: int,
        preserve_key: str | None,
    ) -> ResidentAcquisition | None:
        """Load onto the first candidate GPU that fits, or return None.

        The first pass keeps the active acquisition resident; only when no GPU
        fits that way does a second pass let eviction take it too. On each GPU,
        an out-of-memory error evicts one more acquisition and retries.
        """
        # Leave a little allocator headroom beyond the expected peak.
        required_free_bytes = peak_bytes + min(LOAD_HEADROOM_BYTES, max(0, peak_bytes // 20))
        for preserved in [preserve_key, None] if preserve_key is not None else [None]:
            with self._lock:
                candidates = self._candidate_gpus(
                    resident_bytes,
                    peak_bytes,
                    preserve_active=preserved is not None,
                )
            for gpu in candidates:
                while True:
                    with self._lock:
                        self._evict_for(
                            gpu,
                            resident_bytes,
                            preserve=preserved,
                            required_free_bytes=required_free_bytes,
                        )
                    self._flush(gpu)
                    try:
                        return self._load_executor.submit(self._load, path, gpu).result()
                    except MemoryError as exc:
                        # CuPy's OutOfMemoryError is a MemoryError. Its
                        # traceback retains _load locals, including partly
                        # loaded CUDA arrays. Drop it before flushing the pool
                        # or the retry can never use the memory that just
                        # failed to allocate.
                        exc.__traceback__ = None
                        self._flush(gpu)
                        with self._lock:
                            if self._evict_one(gpu, preserve=preserved) is None:
                                break
        return None

    def _load(self, path: Path, gpu: int) -> ResidentAcquisition:
        """Load one acquisition encoded onto one GPU and prepare its detector session."""
        # Taken before reading so a file rewritten during the load reads as stale.
        signature = file_signature(path)
        with self.gpus[gpu].device:
            loaded = io.load(path, backend="cuda", device=gpu, verbose=False)
            session = detector.prepare(loaded)
        return ResidentAcquisition(
            key=str(path),
            gpu=gpu,
            loaded=loaded,
            session=session,
            resident_bytes=int(loaded.resident_bytes),
            source_signature=signature,
        )

    def _reuse(self, path: Path, *, reserve: bool) -> ResidentAcquisition | None:
        """Return the cached resident of ``path`` as the most recently used, or None.

        Call with the lock held. A resident whose files changed on disk is
        refused rather than reloaded, so products computed before and after a
        request always come from the same counts.
        """
        cached = self._entries.get(str(path))
        if cached is None:
            return None
        if cached.source_signature != file_signature(path):
            raise HTTPException(
                409,
                "The acquisition files changed on disk after they were loaded. "
                "Restart the service to load the new files.",
            )
        self._entries.move_to_end(cached.key)
        if reserve:
            cached.pins += 1
        return cached

    # --- Eviction

    def _evict_for(
        self,
        gpu: int,
        incoming_bytes: int,
        *,
        preserve: str | None = None,
        required_free_bytes: int = 0,
    ) -> None:
        """Evict least recently used entries until the incoming acquisition fits.

        Call with the lock held. Pinned entries are being computed on; wait for
        them rather than evict.
        """
        budget = self.gpus[gpu].cache_budget_bytes
        projected_free = self._free_bytes(gpu) if required_free_bytes > 0 else None
        while self._entries and (
            self._resident_entries(gpu) >= MAX_ENTRIES_PER_GPU
            or self._resident_bytes(gpu) + incoming_bytes > budget
            or (projected_free is not None and projected_free < required_free_bytes)
        ):
            freed = self._evict_one(gpu, preserve=preserve)
            if freed is None:
                pinned_victim = any(
                    key != preserve and entry.gpu == gpu and entry.pins > 0
                    for key, entry in self._entries.items()
                )
                if pinned_victim:
                    self._changed.wait(timeout=1.0)
                    if required_free_bytes > 0:
                        projected_free = self._free_bytes(gpu)
                    continue
                break
            if projected_free is not None:
                projected_free += freed

    def _evict_one(self, gpu: int, *, preserve: str | None = None) -> int | None:
        """Evict the oldest unpinned entry on one GPU and return its byte size. Call with the lock held."""
        victim = next(
            (
                key
                for key, entry in self._entries.items()
                if key != preserve and entry.gpu == gpu and entry.pins == 0
            ),
            None,
        )
        if victim is None:
            return None
        victim_entry = self._entries.pop(victim)
        if victim == self._active_key:
            self._active_key = None
        self._close_entry(victim_entry)
        return victim_entry.resident_bytes

    def _close_entry(self, entry: ResidentAcquisition) -> None:
        """Release one acquisition's device memory once no calculation is using it.

        The detector session borrows the encoded chunks, so the session and the
        loaded acquisition both let go here; otherwise any reference left to
        either one would keep the whole resident on the GPU.
        """
        with self._entry_lock(entry.key), self.gpus[entry.gpu].device:
            entry.session.close()
            entry.loaded.close()
            entry.closed = True
            entry.derived.clear()
        with self._compute_locks_lock:
            self._compute_locks.pop(entry.key, None)

    # --- Primitives

    def _entry_lock(self, key: str) -> threading.Lock:
        """Return the lock that serializes calculations on one resident acquisition."""
        with self._compute_locks_lock:
            return self._compute_locks.setdefault(key, threading.Lock())

    def _resident_bytes(self, gpu: int) -> int:
        return sum(entry.resident_bytes for entry in self._entries.values() if entry.gpu == gpu)

    def _resident_entries(self, gpu: int) -> int:
        return sum(entry.gpu == gpu for entry in self._entries.values())

    def _free_bytes(self, gpu: int) -> int | None:
        """Return the GPU's free bytes, or None when the CUDA runtime cannot report them."""
        try:
            with self.gpus[gpu].device:
                free, _ = cp.cuda.runtime.memGetInfo()
        except RuntimeError:
            return None
        return int(free)

    def _flush(self, gpu: int) -> None:
        """Return cached pool blocks so the next load sees all free device memory."""
        with self.gpus[gpu].device:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
