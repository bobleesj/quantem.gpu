"""Shared, dependency-light helpers for repository benchmark entry points."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import resource
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _nearest_rank_summary(values: list[float]) -> dict[str, float | int]:
    """Return run count, p50, p95, and maximum using nearest ranks."""

    if not values:
        raise ValueError("At least one timing value is required; got an empty list.")
    ordered = sorted(float(value) for value in values)

    def percentile(probability: float) -> float:
        rank = max(1, int(np.ceil(probability * len(ordered))))
        return ordered[min(rank - 1, len(ordered) - 1)]

    return {
        "samples": len(ordered),
        "p50_seconds": percentile(0.50),
        "p95_seconds": percentile(0.95),
        "max_seconds": ordered[-1],
    }


def _git_state() -> dict[str, Any]:
    """Return the current full revision and dirty state."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "diff_sha256": None}
    dirty = bool(diff or untracked)
    digest = None
    if dirty:
        hasher = hashlib.sha256()
        hasher.update(diff)
        for relative in untracked.decode().splitlines():
            hasher.update(relative.encode())
            path = ROOT / relative
            if path.is_file():
                hasher.update(path.read_bytes())
        digest = hasher.hexdigest()
    return {"revision": revision, "dirty": dirty, "diff_sha256": digest}


def _host_info() -> dict[str, Any]:
    """Return portable host and runtime fields without local paths."""

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }


def _process_peak_rss_bytes() -> int | None:
    """Return the process high-water resident set in bytes."""

    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError):
        return None
    if sys.platform == "darwin":
        return value
    return value * 1024


def _process_current_rss_bytes() -> int | None:
    """Return the process resident set at the sampling instant."""

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, AttributeError, OSError):
        return None


def _swap_used_bytes() -> int | None:
    """Return whole-system swap use when psutil is available."""

    try:
        import psutil

        return int(psutil.swap_memory().used)
    except (ImportError, AttributeError, OSError):
        return None


def _memory_snapshot(backend: str) -> dict[str, int | None]:
    """Return backend allocator, driver, card, process, and swap fields."""

    snapshot: dict[str, int | None] = {
        "allocator_current_bytes": None,
        "allocator_reserved_bytes": None,
        "driver_allocated_bytes": None,
        "total_card_used_bytes": None,
        "process_current_rss_bytes": _process_current_rss_bytes(),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "system_swap_used_bytes": _swap_used_bytes(),
    }
    if backend == "cuda":
        try:
            import cupy as cp

            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            snapshot.update(
                {
                    "allocator_current_bytes": int(pool.used_bytes()),
                    "allocator_reserved_bytes": int(pool.total_bytes()),
                    "total_card_used_bytes": int(total_bytes - free_bytes),
                }
            )
        except (AttributeError, ImportError, RuntimeError):
            pass
    elif backend == "mps":
        try:
            import torch

            snapshot.update(
                {
                    "allocator_current_bytes": int(
                        torch.mps.current_allocated_memory()
                    ),
                    "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
                }
            )
        except (AttributeError, ImportError, RuntimeError):
            pass
    return snapshot


class MemorySampler:
    """Sample accelerator and process memory while one timed operation runs."""

    def __init__(self, backend: str, interval_ms: float = 10.0) -> None:
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be positive; got {interval_ms}")
        self.backend = backend
        self.interval_seconds = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, int | None]] = []

    def start(self) -> None:
        """Start sampling; one sampler instance may be started once."""

        if self._thread is not None:
            raise RuntimeError("memory sampler has already been started")
        self._samples.append(_memory_snapshot(self.backend))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._samples.append(_memory_snapshot(self.backend))

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return the maximum observed numeric fields."""

        if self._thread is None:
            raise RuntimeError("memory sampler has not been started")
        self._stop.set()
        self._thread.join()
        self._samples.append(_memory_snapshot(self.backend))
        numeric_fields = set().union(*(sample.keys() for sample in self._samples))
        peak = {
            field: max(
                value
                for sample in self._samples
                if (value := sample.get(field)) is not None
            )
            for field in sorted(numeric_fields)
            if any(sample.get(field) is not None for sample in self._samples)
        }
        return {
            "interval_ms": self.interval_seconds * 1000.0,
            "sample_count": len(self._samples),
            "peak": peak,
        }


def _sync_backend(backend: str) -> None:
    """Wait for queued work on the selected backend."""

    if backend == "cuda":
        try:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
        except (ImportError, RuntimeError):
            pass
    elif backend == "mps":
        try:
            import torch

            torch.mps.synchronize()
        except (ImportError, RuntimeError):
            pass


def _clear_backend(backend: str) -> None:
    """Release benchmark-owned transient references and allocator caches."""

    gc.collect()
    if backend == "cuda":
        try:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except (ImportError, RuntimeError):
            pass
    elif backend == "mps":
        try:
            from quantem.gpu.io import clear_mps_cache

            clear_mps_cache()
        except (ImportError, RuntimeError):
            pass
    gc.collect()


def _release_array(array: Any) -> str | None:
    """Release an explicitly managed benchmark output when supported.

    Native MPS chunk containers own Metal buffers outside Python's garbage
    collector and expose ``free()`` for deterministic release. Ordinary NumPy,
    CuPy, and Torch arrays have no such method and remain allocator-managed.
    """

    free = getattr(array, "free", None)
    if not callable(free):
        return None
    free()
    return "free"


def _array_chunks(array: Any) -> list[np.ndarray]:
    """Return host views for one array or chunk-backed result."""

    chunks = getattr(array, "chunks", None)
    values = list(chunks) if chunks is not None else [array]
    host_chunks: list[np.ndarray] = []
    for value in values:
        module = type(value).__module__.split(".", 1)[0]
        if module == "cupy":
            import cupy as cp

            host_chunks.append(cp.asnumpy(value))
        else:
            host_chunks.append(np.asarray(value))
    return host_chunks


LOGICAL_PIXEL_HASH_SCHEMA = "quantem.gpu.4dstem-logical-pixels/v1"
LOGICAL_PIXEL_AXIS_ORDER = (
    "scan_row",
    "scan_column",
    "detector_row",
    "detector_column",
)


def _array_sha256(array: Any) -> str:
    """Hash logical pixels in C order using canonical little-endian bytes.

    Chunk boundaries are excluded from the digest. Callers separately validate
    scan/detector shape and dtype so storage layouts cannot be conflated.
    """

    digest = hashlib.sha256()
    for chunk in _array_chunks(array):
        dtype = np.dtype(chunk.dtype)
        little_endian = dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(chunk.astype(little_endian, copy=False))
        digest.update(canonical.view(np.uint8))
    return digest.hexdigest()


def _array_description(array: Any) -> dict[str, Any]:
    """Return logical shape, dtype, and byte count."""

    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    nbytes = getattr(array, "nbytes", None)
    chunks = getattr(array, "chunks", None)
    if nbytes is None and chunks is not None:
        nbytes = sum(int(getattr(chunk, "nbytes", 0)) for chunk in chunks)
    return {
        "shape": None if shape is None else [int(value) for value in shape],
        "dtype": None if dtype is None else str(np.dtype(dtype)),
        "logical_resident_bytes": None if nbytes is None else int(nbytes),
    }


def _stable_json_sha256(value: Any) -> str:
    """Return a SHA-256 for a path-free JSON value."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
