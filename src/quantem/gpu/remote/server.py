"""Loopback-only CUDA service for native 4D-STEM viewers.

The service owns no browser UI and no reconstruction orchestration. It exposes
the smallest versioned contract needed by a native client: catalog discovery,
acquisition readiness, exact virtual-detector products, custom detector masks,
and selected diffraction patterns. Raw detector data remains on the compute
host and all full-volume work stays in :mod:`quantem.gpu`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .maped_api import MAPEDProtocolError, MAPEDProtocolService
from .ssb_api import (
    SSBPayloadNotReady,
    SSBPayloadUnavailable,
    SSBProtocolError,
    SSBProtocolService,
)

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised by clean-install smoke
    raise ImportError(
        "Remote viewing requires FastAPI. Install "
        "quantem.gpu with the remote extra: "
        "pip install 'quantem.gpu[cuda,remote]'"
    ) from exc


PROTOCOL_NAME = "quantem-gpu-browse"
PROTOCOL_VERSION = 1
_SCAN_BINS = {1, 2, 4, 8, 16}
_CUDA_CACHE_FRACTION = 0.80
_CUDA_LOAD_HEADROOM_BYTES = 1 << 30
_MAX_MASTER_ENTRIES = 8
_MAX_IMAGE_ENTRIES = 64
_MASTER_RESCAN_INTERVAL_SECONDS = 10.0

logger = logging.getLogger("quantem.gpu.remote")


def _format_size(n_bytes: int) -> str:
    for label, scale in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.1f} {label}"
    return f"{n_bytes} B"


def _wire_image(array: object) -> tuple[bytes, str]:
    """Encode one image without changing exact detector counts."""
    image = np.asarray(array)
    if image.ndim != 2:
        raise HTTPException(500, f"Remote compute produced a non-2-D image: {image.shape}.")
    if np.issubdtype(image.dtype, np.integer):
        if np.issubdtype(image.dtype, np.signedinteger) and image.size and image.min() < 0:
            raise HTTPException(500, "Remote compute produced negative detector counts.")
        maximum = int(image.max()) if image.size else 0
        if maximum > int(np.iinfo(np.uint32).max):
            raise HTTPException(
                409,
                "Exact detector counts exceed uint32 display capacity. "
                "Use a smaller scan bin or detector aperture.",
            )
        wire = np.ascontiguousarray(image, dtype="<u4")
        return wire.tobytes(), "<u4"
    wire = np.ascontiguousarray(image, dtype="<f4")
    return wire.tobytes(), "<f4"


def _image_response(
    array: object,
    *,
    cache_control: str = "max-age=300",
    value_divisor: int = 1,
) -> Response:
    image = np.asarray(array)
    payload, dtype = _wire_image(image)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Width": str(image.shape[1]),
            "X-Height": str(image.shape[0]),
            "X-Dtype": dtype,
            "X-Value-Divisor": str(max(1, int(value_divisor))),
            "Cache-Control": cache_control,
        },
    )


def _scan_region(
    row_start: int | None,
    row_stop: int | None,
    column_start: int | None,
    column_stop: int | None,
) -> tuple[int, int, int, int] | None:
    values = (row_start, row_stop, column_start, column_stop)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise HTTPException(
            400,
            "scan crop requires row_start, row_stop, column_start, and column_stop",
        )
    region = tuple(int(value) for value in values)
    if region[0] < 0 or region[2] < 0 or region[1] <= region[0] or region[3] <= region[2]:
        raise HTTPException(
            400,
            "scan crop must be a nonempty half-open (row, column) region",
        )
    return region


def _scan_roi_indices(
    scan_shape: tuple[int, int],
    *,
    shape: str,
    center_row: float,
    center_column: float,
    radius: float,
) -> np.ndarray:
    """Return exact flat scan indices for one circle or square ROI."""
    if shape not in {"circle", "square"}:
        raise HTTPException(400, "scan ROI shape must be circle or square")
    if not math.isfinite(radius) or radius <= 0:
        raise HTTPException(400, "scan ROI radius must be greater than zero")
    if not math.isfinite(center_row) or not math.isfinite(center_column):
        raise HTTPException(400, "scan ROI center must be finite")
    row_start = max(0, math.ceil(center_row - radius))
    row_stop = min(scan_shape[0] - 1, math.floor(center_row + radius))
    column_start = max(0, math.ceil(center_column - radius))
    column_stop = min(scan_shape[1] - 1, math.floor(center_column + radius))
    if row_start > row_stop or column_start > column_stop:
        raise HTTPException(400, "scan ROI does not include any scan positions")

    rows = np.arange(row_start, row_stop + 1, dtype=np.int64)[:, None]
    columns = np.arange(column_start, column_stop + 1, dtype=np.int64)[None, :]
    if shape == "circle":
        row_offset = rows - float(center_row)
        column_offset = columns - float(center_column)
        mask = row_offset * row_offset + column_offset * column_offset <= radius * radius
        selected_rows, selected_columns = np.nonzero(mask)
        indices = (
            (selected_rows + row_start) * scan_shape[1]
            + selected_columns
            + column_start
        )
    else:
        indices = (rows * scan_shape[1] + columns).reshape(-1)
    indices = indices.astype(np.int32, copy=False)
    if indices.size == 0:
        raise HTTPException(400, "scan ROI does not include any scan positions")
    return indices


def _file_signature(master: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [master]
    stem = master.name.removesuffix("_master.h5")
    paths.extend(sorted(master.parent.glob(f"{stem}_data_*.h5")))
    try:
        import h5py

        with h5py.File(master, "r") as handle:
            data_group = handle.get("entry/data")
            if data_group is not None:
                for name in data_group:
                    link = data_group.get(name, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        linked = Path(link.filename)
                        if not linked.is_absolute():
                            linked = master.parent / linked
                        paths.append(linked.absolute())
    except (OSError, KeyError, TypeError, ValueError):
        pass
    signature: list[tuple[str, int, int]] = []
    for path in dict.fromkeys(paths):
        try:
            stat = path.stat()
        except OSError:
            continue
        signature.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(signature)


def _session_identity(root: Path, session_dir: Path) -> tuple[str, str]:
    try:
        relative = session_dir.relative_to(root)
    except ValueError:
        relative = Path(session_dir.name)
    if len(relative.parts) >= 2:
        return relative.parts[-2], relative.parts[-1]
    if len(relative.parts) == 1:
        return root.name, relative.parts[0]
    return session_dir.parent.name, session_dir.name


def _session_path(root: Path, session_dir: Path) -> str:
    """Return one stable POSIX folder path relative to the served root."""
    try:
        relative = session_dir.relative_to(root)
    except ValueError:
        relative = Path(session_dir.name)
    return relative.as_posix() or "."


class BrowseService:
    """Own a CUDA device pool, one raw-data root, and bounded viewer caches.

    Each resident 4D volume stays wholly on one GPU. Separate datasets are
    placed across the configured pool, preserving exact data and avoiding the
    synchronization cost and complexity of splitting one interactive dataset.
    """

    def __init__(
        self,
        data_folder: str | os.PathLike[str],
        *,
        gpu: int = 0,
        gpus: Sequence[int] | str | None = None,
        initialize_cuda: bool = True,
    ) -> None:
        self.data_folder = Path(data_folder).expanduser().absolute()
        if isinstance(gpus, str):
            if gpus != "auto":
                raise ValueError("gpus must be 'auto' or a sequence of CUDA indices")
            self.requested_gpus: tuple[int, ...] | None = None
        else:
            requested = tuple(dict.fromkeys(int(value) for value in (gpus or (gpu,))))
            if not requested or any(value < 0 for value in requested):
                raise ValueError("CUDA device indices must be zero or greater")
            self.requested_gpus = requested
        self.gpu = self.requested_gpus[0] if self.requested_gpus else 0
        self.gpus = list(self.requested_gpus or ())
        self.backend: str | None = None
        self.device_name: str | None = None
        self.device_error: str | None = None
        self.device: Any = None
        self.cache_budget_bytes = 0
        self.aggregate_cache_budget_bytes = 0
        self._devices: dict[int, Any] = {}
        self._device_names: dict[int, str] = {}
        self._total_memory_bytes: dict[int, int] = {}
        self._cache_budgets: dict[int, int] = {}

        self._catalog_lock = threading.Lock()
        self._catalog_refresh_lock = threading.Lock()
        self._catalog_generation = 0
        self._catalog: dict[str, Any] | None = None
        self._session_paths: dict[str, Path] = {}
        self._inspection_lock = threading.Lock()
        self._inspection_cache: dict[str, tuple[tuple, Any]] = {}
        self._master_discovery_lock = threading.Lock()
        self._master_paths_lock = threading.Lock()
        self._known_master_paths: tuple[Path, ...] | None = None
        self._master_scan_in_flight = False
        self._last_master_scan_completed = 0.0

        self._master_lock = threading.Lock()
        self._resident_changed = threading.Condition(self._master_lock)
        self._load_lock = threading.Lock()
        # Numba's parallel header parser creates one native worker pool per
        # calling thread.  Keep every serialized load on the same reusable
        # thread so long-running HTTP services do not accumulate another pool
        # whenever Starlette chooses a new worker.
        self._load_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="quantem-cuda-load",
        )
        self._master_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self._active_master_key: tuple | None = None
        self._image_lock = threading.Lock()
        self._image_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._compute_locks: dict[tuple, threading.Lock] = {}
        self._compute_locks_lock = threading.Lock()
        if initialize_cuda:
            self._initialize_cuda()

    def _initialize_cuda(self) -> None:
        try:
            import cupy as cp

            requested = self.requested_gpus
            if requested is None:
                requested = tuple(range(int(cp.cuda.runtime.getDeviceCount())))
            errors: list[str] = []
            for gpu in requested:
                try:
                    device = cp.cuda.Device(gpu)
                    with device:
                        properties = cp.cuda.runtime.getDeviceProperties(gpu)
                        name = properties.get("name")
                        if isinstance(name, bytes):
                            name = name.decode("utf-8", errors="replace")
                        total = int(properties.get("totalGlobalMem") or 0)
                        if total <= 0:
                            _, total = cp.cuda.runtime.memGetInfo()
                        cp.zeros((1,), dtype=cp.uint8).sum()
                        cp.get_default_memory_pool().free_all_blocks()
                    self._devices[gpu] = device
                    self._device_names[gpu] = str(name) if name else f"CUDA GPU {gpu}"
                    self._total_memory_bytes[gpu] = total
                    self._cache_budgets[gpu] = int(total * _CUDA_CACHE_FRACTION)
                except (RuntimeError, MemoryError) as exc:
                    errors.append(f"GPU {gpu}: {type(exc).__name__}: {exc}")
            self.gpus = list(self._devices)
            if not self.gpus:
                detail = "; ".join(errors) or "no CUDA devices were found"
                raise RuntimeError(detail)
            self.gpu = self.gpus[0]
            self.device = self._devices[self.gpu]
            self.device_name = self._device_names[self.gpu]
            self.cache_budget_bytes = max(self._cache_budgets.values())
            self.aggregate_cache_budget_bytes = sum(self._cache_budgets.values())
            self.device_error = "; ".join(errors) or None
            self.backend = "cuda"
        except (ImportError, RuntimeError, MemoryError) as exc:
            self.backend = None
            self.device = None
            self.device_error = f"{type(exc).__name__}: {exc}"

    def capabilities(self) -> dict[str, Any]:
        with self._master_lock:
            devices = []
            for gpu in self._available_gpus():
                capacity = self._admission_capacity(gpu)
                devices.append(
                    {
                        "index": gpu,
                        "name": self._device_name(gpu),
                        "total_memory_bytes": self._total_memory_bytes.get(gpu),
                        "cache_budget_bytes": capacity["cache_budget_bytes"],
                        "free_bytes": capacity["free_bytes"],
                        "resident_bytes": capacity["resident_bytes"],
                        "resident_entries": self._resident_entries(gpu),
                        "active_resident_bytes": capacity["active_resident_bytes"],
                        "evictable_bytes": capacity["evictable_bytes"],
                        "available_peak_bytes": capacity["available_peak_bytes"],
                        "available_resident_bytes": capacity[
                            "available_resident_bytes"
                        ],
                    }
                )
        aggregate_budget = self.aggregate_cache_budget_bytes or sum(
            device["cache_budget_bytes"] for device in devices
        )
        return {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "backend": self.backend,
            "device_name": self.device_name,
            "device_error": self.device_error,
            "browse_gpu": self.gpu if self.backend == "cuda" else None,
            "browse_gpus": [device["index"] for device in devices],
            "cache_fraction": _CUDA_CACHE_FRACTION,
            "cache_budget_bytes": self.cache_budget_bytes,
            "aggregate_cache_budget_bytes": aggregate_budget,
            "largest_device_memory_bytes": max(
                self._total_memory_bytes.values(), default=None
            ),
            "devices": devices,
            "data_folders": [str(self.data_folder)],
            "features": {
                "catalog_refresh": True,
                "selected_diffraction": True,
                "virtual_detectors": True,
                "custom_detector": True,
                "scan_roi_diffraction": True,
                "acquisition_events": True,
                "exact_integer_images": True,
                "multi_gpu_residency": len(devices) > 1,
                "ssb": SSBProtocolService.advertised_capability(),
            },
        }

    def _discover_master_paths(self) -> list[Path]:
        if not self.data_folder.is_dir():
            return []
        return sorted(
            path
            for path in self.data_folder.rglob("*_master.h5")
            if not path.name.startswith("._")
            and not any(
                part.startswith((".", "_"))
                for part in path.relative_to(self.data_folder).parts
            )
        )

    def _scan_master_paths(self) -> list[Path]:
        with self._master_discovery_lock:
            paths = self._discover_master_paths()
        with self._master_paths_lock:
            self._known_master_paths = tuple(paths)
            self._last_master_scan_completed = time.monotonic()
            self._master_scan_in_flight = False
        return paths

    def _finish_background_master_scan(self) -> None:
        try:
            self._scan_master_paths()
        except Exception:  # pragma: no cover - protects the long-lived service
            logger.exception("background master discovery failed")
            with self._master_paths_lock:
                self._last_master_scan_completed = time.monotonic()
                self._master_scan_in_flight = False

    def _watched_master_paths(self) -> list[Path]:
        with self._master_paths_lock:
            known = self._known_master_paths
            should_scan = (
                known is not None
                and not self._master_scan_in_flight
                and time.monotonic() - self._last_master_scan_completed
                >= _MASTER_RESCAN_INTERVAL_SECONDS
            )
            if should_scan:
                self._master_scan_in_flight = True
        if known is None:
            return self._scan_master_paths()
        if should_scan:
            threading.Thread(
                target=self._finish_background_master_scan,
                name="quantem-gpu-master-discovery",
                daemon=True,
            ).start()
        return list(known)

    def _inspect(self, master: Path):
        signature = _file_signature(master)
        key = str(master)
        with self._inspection_lock:
            cached = self._inspection_cache.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]
        from quantem.gpu.io import inspect

        inspection = inspect(master)
        with self._inspection_lock:
            self._inspection_cache[key] = (signature, inspection)
        return inspection

    def _watch_inspection(self, master: Path):
        """Reuse unchanged inspections while continuing to poll active writers."""
        key = str(master)
        with self._inspection_lock:
            cached = self._inspection_cache.get(key)
        if cached is None:
            return self._inspect(master)

        signature, inspection = cached
        if not inspection.ready:
            source_files = (inspection.source_signature or {}).get("files", [])
            if source_files and self._source_files_unchanged(source_files):
                return inspection
            return self._inspect(master)

        master_signature = next(
            (item for item in signature if item[0] == key),
            None,
        )
        try:
            stat = master.stat()
        except OSError:
            return self._inspect(master)
        if master_signature is None or master_signature[1:] != (
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ):
            return self._inspect(master)
        return inspection

    @staticmethod
    def _source_files_unchanged(source_files: list[dict[str, Any]]) -> bool:
        """Check an incomplete acquisition with metadata-only file stats."""
        for expected in source_files:
            path = expected.get("path")
            if not path:
                return False
            try:
                stat = Path(path).stat()
            except OSError:
                if expected.get("missing", False):
                    continue
                return False
            if expected.get("missing", False) or expected.get("unreadable", False):
                return False
            if (
                int(expected.get("size", -1)) != int(stat.st_size)
                or int(expected.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
            ):
                return False
        return True

    def _master_size(self, master: Path) -> int:
        return sum(item[1] for item in _file_signature(master))

    @staticmethod
    def _sampling(metadata: dict[str, Any], *names: str) -> float | None:
        for name in names:
            value = metadata.get(name)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def refresh_catalog(self) -> dict[str, Any]:
        with self._catalog_lock:
            requested_generation = self._catalog_generation
        with self._catalog_refresh_lock:
            with self._catalog_lock:
                if (
                    self._catalog is not None
                    and self._catalog_generation != requested_generation
                ):
                    return self._catalog
            return self._refresh_catalog()

    def _refresh_catalog(self) -> dict[str, Any]:
        grouped: dict[str, tuple[str, str, list[Path]]] = {}
        session_paths: dict[str, Path] = {}
        legacy_paths: dict[str, Path | None] = {}
        for master in self._scan_master_paths():
            source, date = _session_identity(self.data_folder, master.parent)
            path = _session_path(self.data_folder, master.parent)
            if path not in grouped:
                grouped[path] = (source, date, [])
            grouped[path][2].append(master)
            session_paths[path] = master.parent
            legacy_path = f"{source}/{date}"
            if legacy_path not in legacy_paths:
                legacy_paths[legacy_path] = master.parent
            elif legacy_paths[legacy_path] != master.parent:
                legacy_paths[legacy_path] = None

        for path, directory in legacy_paths.items():
            if directory is not None:
                session_paths.setdefault(path, directory)

        sessions: list[dict[str, Any]] = []
        for path, (source, date, masters) in sorted(grouped.items()):
            files: list[dict[str, Any]] = []
            for master in sorted(masters):
                try:
                    inspection = self._watch_inspection(master)
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("could not inspect %s: %s", master, exc)
                    continue
                scan_shape = inspection.scan_shape or (0, 0)
                detector_shape = inspection.detector_shape or (0, 0)
                source_files = (inspection.source_signature or {}).get("files", [])
                size = sum(int(item.get("size", 0)) for item in source_files)
                if size <= 0:
                    size = self._master_size(master)
                metadata = inspection.metadata or {}
                scan_sampling = self._sampling(
                    metadata,
                    "scan_sampling_A",
                    "scan_sampling_angstrom",
                )
                detector_sampling = self._sampling(
                    metadata,
                    "k_pixel_size_mrad",
                    "detector_sampling_mrad",
                )
                files.append(
                    {
                        "name": master.name,
                        "shape": [*scan_shape, *detector_shape],
                        "dtype": inspection.dtype,
                        "cal": "ok" if scan_sampling is not None else "un",
                        "size": _format_size(size),
                        "size_bytes": size,
                        "loadable": bool(inspection.ready),
                        "load_status": {
                            "loadable": bool(inspection.ready),
                            "reason": inspection.reason,
                            "action": inspection.action,
                        },
                        "scan_sampling_A": scan_sampling,
                        "k_pixel_size_mrad": detector_sampling,
                        "has_ssb": False,
                    }
                )
            if files:
                sessions.append(
                    {"path": path, "source": source, "date": date, "files": files}
                )

        payload = {
            "data_folder": str(self.data_folder),
            "data_folders": [str(self.data_folder)],
            "sessions": sessions,
            "complete": True,
        }
        with self._catalog_lock:
            self._catalog = payload
            self._session_paths = session_paths
            self._catalog_generation += 1
        return payload

    def sessions(self, *, refresh: bool = False) -> dict[str, Any]:
        with self._catalog_lock:
            cached = self._catalog
        if cached is None or refresh:
            return self.refresh_catalog()
        return cached

    def acquisitions(self) -> dict[str, Any]:
        pending: list[dict[str, Any]] = []
        ready: list[dict[str, Any]] = []
        ready_signatures: list[tuple[str, tuple]] = []
        for master in self._watched_master_paths():
            try:
                inspection = self._watch_inspection(master)
            except (OSError, KeyError, TypeError, ValueError):
                continue
            source_files = (inspection.source_signature or {}).get("files", [])
            expected_chunks = max(0, len(source_files) - 1)
            present_chunks = sum(
                1
                for item in source_files
                if item.get("path") != str(master) and not item.get("missing", False)
            )
            latest_mtime = max(
                (int(item.get("mtime_ns", 0)) for item in source_files),
                default=0,
            )
            event = {
                "kind": "ready" if inspection.ready else "writing",
                "path": str(master),
                "stem": master.name.removesuffix("_master.h5"),
                "chunks_seen": present_chunks,
                "chunks_expected": expected_chunks,
                "bytes_total": sum(int(item.get("size", 0)) for item in source_files),
                "ts": datetime.fromtimestamp(
                    latest_mtime / 1_000_000_000 if latest_mtime else 0,
                    tz=UTC,
                ).isoformat(),
            }
            if inspection.ready:
                ready.append(event)
                ready_signatures.append(
                    (
                        str(master),
                        tuple(
                            (
                                str(item.get("path", "")),
                                int(item.get("size", 0)),
                                int(item.get("mtime_ns", 0)),
                            )
                            for item in source_files
                        ),
                    )
                )
            else:
                pending.append(event)
        token = hashlib.blake2b(repr(ready_signatures).encode(), digest_size=12).hexdigest()
        return {
            "enabled": True,
            "pending": pending,
            "in_flight": [],
            "history": ready,
            "ready_token": token,
        }

    def resolve_master(self, session: str, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise HTTPException(400, "file must be one master filename")
        with self._catalog_lock:
            directory = self._session_paths.get(session)
        if directory is None:
            self.refresh_catalog()
            with self._catalog_lock:
                directory = self._session_paths.get(session)
        if directory is None:
            raise HTTPException(404, f"unknown session: {session}")
        path = directory / filename
        if not path.is_file() or path.parent != directory:
            raise HTTPException(404, f"master not found: {filename}")
        return path

    @staticmethod
    def _plan_key(
        path: Path,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
    ) -> tuple:
        return (str(path), int(det_bin), int(scan_bin), scan_region)

    def _normalize_scan_region(
        self,
        path: Path,
        scan_region: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        """Treat an explicit full scan as the loader's uncropped path."""
        if scan_region is None:
            return None
        inspection = self._inspect(path)
        if inspection.scan_shape is None:
            return scan_region
        rows, columns = (int(value) for value in inspection.scan_shape)
        return None if scan_region == (0, rows, 0, columns) else scan_region

    def _plan_bytes(
        self,
        path: Path,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
    ) -> tuple[int, int]:
        inspection = self._inspect(path)
        if inspection.scan_shape is None or inspection.detector_shape is None:
            raise HTTPException(422, "The master does not report a usable 4D shape.")
        scan_rows, scan_columns = (int(value) for value in inspection.scan_shape)
        if scan_region is not None:
            r0, r1, c0, c1 = scan_region
            if r1 > scan_rows or c1 > scan_columns:
                raise HTTPException(400, "scan crop exceeds the source scan shape")
            scan_rows, scan_columns = r1 - r0, c1 - c0
        detector_rows, detector_columns = (int(value) for value in inspection.detector_shape)
        detector_rows = math.ceil(detector_rows / det_bin)
        detector_columns = math.ceil(detector_columns / det_bin)
        source_positions = scan_rows * scan_columns
        try:
            source_itemsize = int(np.dtype(inspection.dtype).itemsize)
        except TypeError:
            source_itemsize = 2
        # The loader normally narrows over-allocated uint32 Arina counts to
        # uint16 in the decode pass. Admission still reserves the native
        # itemsize so a real count above 65535 can remain exact without an
        # unexpected out-of-memory failure.
        decoded_itemsize = source_itemsize if det_bin == 1 else 4
        decoded_bytes = source_positions * detector_rows * detector_columns * decoded_itemsize
        output_rows = math.ceil(scan_rows / scan_bin)
        output_columns = math.ceil(scan_columns / scan_bin)
        resident_itemsize = decoded_itemsize if scan_bin == 1 else 4
        resident_bytes = (
            output_rows
            * output_columns
            * detector_rows
            * detector_columns
            * resident_itemsize
        )
        peak_bytes = max(decoded_bytes, resident_bytes)
        if scan_bin > 1:
            peak_bytes = decoded_bytes + resident_bytes
        return resident_bytes, peak_bytes

    @staticmethod
    def _entry_bytes(entry: dict[str, Any]) -> int:
        data = entry.get("data")
        return int(getattr(data, "nbytes", 0) or 0)

    def _available_gpus(self) -> list[int]:
        if self._devices:
            return list(self._devices)
        if self.backend == "cuda":
            return [self.gpu]
        return []

    def _device_for(self, gpu: int):
        if gpu in self._devices:
            return self._devices[gpu]
        if gpu == self.gpu:
            return self.device
        return None

    def _device_name(self, gpu: int) -> str:
        if gpu in self._device_names:
            return self._device_names[gpu]
        if gpu == self.gpu and self.device_name:
            return self.device_name
        return f"CUDA GPU {gpu}"

    def _budget_for(self, gpu: int) -> int:
        if gpu in self._cache_budgets:
            return self._cache_budgets[gpu]
        if gpu == self.gpu:
            return self.cache_budget_bytes
        return 0

    def _resident_bytes(self, gpu: int) -> int:
        return sum(
            self._entry_bytes(entry)
            for entry in self._master_cache.values()
            if int(entry.get("gpu", self.gpu)) == gpu
        )

    def _resident_entries(self, gpu: int) -> int:
        return sum(
            int(entry.get("gpu", self.gpu)) == gpu
            for entry in self._master_cache.values()
        )

    def _free_bytes(self, gpu: int) -> int | None:
        try:
            import cupy as cp

            with self._cuda_context(gpu):
                free, _ = cp.cuda.runtime.memGetInfo()
            return int(free)
        except (ImportError, RuntimeError, AttributeError):
            return None

    def _admission_capacity(
        self,
        gpu: int,
        *,
        preserve_active: bool = True,
    ) -> dict[str, int | None]:
        """Return the live capacity used by both reporting and admission."""
        resident = self._resident_bytes(gpu)
        active = (
            self._master_cache.get(self._active_master_key)
            if preserve_active
            else None
        )
        active_resident = (
            self._entry_bytes(active)
            if active is not None and int(active.get("gpu", self.gpu)) == gpu
            else 0
        )
        evictable = max(0, resident - active_resident)
        budget = self._budget_for(gpu)
        free = self._free_bytes(gpu)
        available_peak: int | None = budget if budget > 0 else None
        if free is not None:
            physical_capacity = max(0, free + evictable)
            available_peak = (
                min(available_peak, physical_capacity)
                if available_peak is not None
                else physical_capacity
            )
        available_resident = max(0, budget - active_resident) if budget > 0 else None
        return {
            "cache_budget_bytes": budget,
            "free_bytes": free,
            "resident_bytes": resident,
            "active_resident_bytes": active_resident,
            "evictable_bytes": evictable,
            "available_peak_bytes": available_peak,
            "available_resident_bytes": available_resident,
        }

    def mark_active(self, key: tuple) -> None:
        """Protect the selected resident dataset from cache eviction."""
        with self._master_lock:
            self._active_master_key = key

    def _cuda_context(self, gpu: int | None = None):
        """Select an entry's CUDA device in the calling worker thread."""
        selected = self.gpu if gpu is None else int(gpu)
        device = self._device_for(selected)
        return device if device is not None else nullcontext()

    def _close_entry(self, entry: dict[str, Any]) -> None:
        cache_key = entry.get("cache_key")
        gpu = int(entry.get("gpu", self.gpu))
        lock = self._entry_lock(cache_key) if cache_key is not None else nullcontext()
        with lock:
            with self._cuda_context(gpu):
                compute = entry.get("compute")
                if compute is not None:
                    compute.close()
                data = entry.get("data")
                if data is not None and hasattr(data, "free"):
                    try:
                        data.free()
                    except (AttributeError, RuntimeError):
                        pass
            entry.clear()
        if cache_key is not None:
            with self._compute_locks_lock:
                self._compute_locks.pop(cache_key, None)

    def _flush_cuda_pool(self, gpu: int) -> None:
        try:
            import cupy as cp

            with self._cuda_context(gpu):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except (ImportError, RuntimeError, MemoryError, AttributeError):
            pass

    def _evict_for(
        self,
        gpu: int,
        incoming_bytes: int,
        *,
        preserve: tuple | None = None,
        required_free_bytes: int = 0,
    ) -> None:
        budget = self._budget_for(gpu)
        projected_free = self._free_bytes(gpu) if required_free_bytes > 0 else None
        while self._master_cache and (
            self._resident_entries(gpu) >= _MAX_MASTER_ENTRIES
            or (budget > 0 and self._resident_bytes(gpu) + incoming_bytes > budget)
            or (
                projected_free is not None
                and projected_free < required_free_bytes
            )
        ):
            freed = self._evict_one(gpu, preserve=preserve)
            if freed is None:
                pinned_victim = any(
                    key != preserve
                    and int(entry.get("gpu", self.gpu)) == gpu
                    and int(entry.get("resident_pins", 0)) > 0
                    for key, entry in self._master_cache.items()
                )
                if pinned_victim:
                    self._resident_changed.wait(timeout=1.0)
                    if required_free_bytes > 0:
                        projected_free = self._free_bytes(gpu)
                    continue
                break
            if projected_free is not None:
                projected_free += freed

    def _evict_one(self, gpu: int, *, preserve: tuple | None = None) -> int | None:
        """Evict the oldest cache entry on one GPU and return its byte size."""
        victim = next(
            (
                key
                for key, entry in self._master_cache.items()
                if key != preserve
                and int(entry.get("gpu", self.gpu)) == gpu
                and int(entry.get("resident_pins", 0)) == 0
            ),
            None,
        )
        if victim is None:
            return None
        victim_entry = self._master_cache.pop(victim)
        freed = self._entry_bytes(victim_entry)
        if victim == self._active_master_key:
            self._active_master_key = None
        self._close_entry(victim_entry)
        return freed

    @staticmethod
    def _load_free_requirement(peak_bytes: int) -> int:
        """Reserve small transient CUDA allocator headroom without shrinking cache."""
        headroom = min(_CUDA_LOAD_HEADROOM_BYTES, max(0, peak_bytes // 20))
        return peak_bytes + headroom

    def _candidate_gpus(
        self,
        resident_bytes: int,
        peak_bytes: int,
        *,
        preserve_active: bool = True,
    ) -> list[int]:
        candidates: list[tuple[int, int, float, int]] = []
        for gpu in self._available_gpus():
            capacity = self._admission_capacity(
                gpu,
                preserve_active=preserve_active,
            )
            available_peak = capacity["available_peak_bytes"]
            available_resident = capacity["available_resident_bytes"]
            if available_peak is not None and peak_bytes > available_peak:
                continue
            if available_resident is not None and resident_bytes > available_resident:
                continue
            budget = int(capacity["cache_budget_bytes"] or 0)
            resident = int(capacity["resident_bytes"] or 0)
            pressure = resident / budget if budget > 0 else 0.0
            available = available_peak or 0
            candidates.append((self._resident_entries(gpu), -available, pressure, gpu))
        candidates.sort()
        return [candidate[3] for candidate in candidates]

    def _load_entry(
        self,
        path: Path,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
        gpu: int,
    ) -> dict[str, Any]:
        if self.backend != "cuda" or self._device_for(gpu) is None:
            raise HTTPException(503, f"CUDA unavailable: {self.device_error}")
        from quantem.gpu import detector
        from quantem.gpu.io import load

        with self._cuda_context(gpu):
            loaded = load(
                path,
                backend="cuda",
                device=gpu,
                det_bin=det_bin,
                scan_region=scan_region,
                verbose=False,
            )
            data = loaded.data
            if scan_bin > 1:
                from quantem.gpu.io.load import bin as gpu_bin

                data = gpu_bin(
                    data,
                    factor=scan_bin,
                    axes="scan",
                    reduction="sum",
                    edge="partial",
                )
            compute = detector.prepare(data)
            mean_dp = np.asarray(compute.mean_dp(), dtype=np.float32)
        return {
            "gpu": gpu,
            "data": data,
            "compute": compute,
            "scan_shape": tuple(int(value) for value in data.shape[:2]),
            "detector_shape": tuple(int(value) for value in data.shape[-2:]),
            "mean_dp": mean_dp,
            "bf_geometry": None,
            "com_row": None,
            "com_column": None,
        }

    def entry(
        self,
        session: str,
        filename: str,
        *,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
        reserve: bool = False,
    ) -> tuple[tuple, dict[str, Any]]:
        det_bin = max(1, int(det_bin))
        scan_bin = int(scan_bin)
        if scan_bin not in _SCAN_BINS:
            raise HTTPException(400, "scan_bin must be 1, 2, 4, 8, or 16")
        if self.backend != "cuda" or not self._available_gpus():
            raise HTTPException(503, f"CUDA unavailable: {self.device_error}")
        path = self.resolve_master(session, filename)
        scan_region = self._normalize_scan_region(path, scan_region)
        key = self._plan_key(path, det_bin, scan_bin, scan_region)
        with self._master_lock:
            cached = self._master_cache.get(key)
            if cached is not None:
                self._master_cache.move_to_end(key)
                if reserve:
                    cached["resident_pins"] = int(cached.get("resident_pins", 0)) + 1
                return key, cached

        resident_bytes, peak_bytes = self._plan_bytes(path, det_bin, scan_bin, scan_region)
        largest_budget = max(
            (self._budget_for(gpu) for gpu in self._available_gpus()),
            default=0,
        )
        if largest_budget > 0 and peak_bytes > largest_budget:
            raise HTTPException(
                413,
                f"Requested exact load needs a {_format_size(peak_bytes)} transition peak "
                f"but the largest per-GPU data-cache budget is {_format_size(largest_budget)}. "
                "Crop the scan region or increase scan/detector binning.",
            )
        with self._load_lock:
            with self._master_lock:
                cached = self._master_cache.get(key)
                if cached is not None:
                    self._master_cache.move_to_end(key)
                    if reserve:
                        cached["resident_pins"] = int(cached.get("resident_pins", 0)) + 1
                    return key, cached
                stale_keys = [
                    candidate
                    for candidate in self._master_cache
                    if candidate[0] == str(path)
                ]
                for stale_key in stale_keys:
                    stale = self._master_cache.get(stale_key)
                    while stale is not None and int(stale.get("resident_pins", 0)) > 0:
                        self._resident_changed.wait(timeout=1.0)
                        stale = self._master_cache.get(stale_key)
                    if stale is not None:
                        self._master_cache.pop(stale_key)
                        if stale_key == self._active_master_key:
                            self._active_master_key = None
                        self._close_entry(stale)
                candidates = self._candidate_gpus(resident_bytes, peak_bytes)
                preserve_key = self._active_master_key
                if not candidates:
                    candidates = self._candidate_gpus(
                        resident_bytes,
                        peak_bytes,
                        preserve_active=False,
                    )
                    preserve_key = None
            if not candidates:
                raise HTTPException(
                    413,
                    "No configured CUDA GPU can admit the exact load. Crop the scan "
                    "region or increase scan/detector binning.",
                )
            loaded: dict[str, Any] | None = None
            preservation_passes = [preserve_key]
            if preserve_key is not None:
                preservation_passes.append(None)
            for preserved in preservation_passes:
                with self._master_lock:
                    candidates = self._candidate_gpus(
                        resident_bytes,
                        peak_bytes,
                        preserve_active=preserved is not None,
                    )
                for gpu in candidates:
                    while True:
                        with self._master_lock:
                            self._evict_for(
                                gpu,
                                resident_bytes,
                                preserve=preserved,
                                required_free_bytes=self._load_free_requirement(peak_bytes),
                            )
                        self._flush_cuda_pool(gpu)
                        try:
                            loaded = self._load_executor.submit(
                                self._load_entry,
                                path,
                                det_bin,
                                scan_bin,
                                scan_region,
                                gpu,
                            ).result()
                            break
                        except Exception as exc:
                            try:
                                import cupy as cp

                                out_of_memory = isinstance(
                                    exc,
                                    (MemoryError, cp.cuda.memory.OutOfMemoryError),
                                )
                            except ImportError:
                                out_of_memory = isinstance(exc, MemoryError)
                            if not out_of_memory:
                                raise
                            # An exception traceback retains _load_entry locals,
                            # including decoded and binned CUDA arrays. Drop it
                            # before flushing the pool or the retry can never use
                            # the memory that just failed to allocate.
                            exc.__traceback__ = None
                            self._flush_cuda_pool(gpu)
                            with self._master_lock:
                                freed = self._evict_one(gpu, preserve=preserved)
                            if freed is None:
                                break
                    if loaded is not None:
                        break
                if loaded is not None:
                    break
            if loaded is None:
                raise HTTPException(
                    413,
                    "The exact load did not fit on any configured CUDA GPU. Crop the "
                    "scan region or increase scan/detector binning.",
                )
            loaded["cache_key"] = key
            with self._master_lock:
                self._evict_for(
                    int(loaded["gpu"]),
                    self._entry_bytes(loaded),
                    preserve=preserve_key,
                )
                self._master_cache[key] = loaded
                self._master_cache.move_to_end(key)
                if reserve:
                    loaded["resident_pins"] = int(loaded.get("resident_pins", 0)) + 1
            return key, loaded

    def _bf_geometry(self, entry: dict[str, Any]) -> tuple[float, float, float]:
        geometry = entry.get("bf_geometry")
        if geometry is None:
            from quantem.gpu import detector

            center, radius = detector.auto_probe(entry["mean_dp"])
            geometry = (float(center[0]), float(center[1]), float(radius))
            entry["bf_geometry"] = geometry
        return geometry

    def _entry_lock(self, key: tuple) -> threading.Lock:
        with self._compute_locks_lock:
            return self._compute_locks.setdefault(key, threading.Lock())

    def _release_entry(self, entry: dict[str, Any]) -> None:
        """Release one short-lived reservation acquired by :meth:`entry`."""
        with self._master_lock:
            pins = int(entry.get("resident_pins", 0))
            if pins <= 0:
                raise RuntimeError("resident entry reservation was released twice")
            entry["resident_pins"] = pins - 1
            self._resident_changed.notify_all()

    @contextmanager
    def _resident_entry(self, key: tuple, expected: dict[str, Any] | None = None):
        """Pin one resident entry while an interactive result is computed."""
        with self._master_lock:
            entry = self._master_cache.get(key)
            if entry is None or (expected is not None and entry is not expected):
                raise HTTPException(
                    409,
                    "The requested crop/bin is no longer resident; load a virtual "
                    "image first.",
                )
            lock = self._entry_lock(key)
            self._master_cache.move_to_end(key)
            entry["resident_pins"] = int(entry.get("resident_pins", 0)) + 1
        lock.acquire()
        try:
            if not entry:
                raise HTTPException(
                    409,
                    "The requested crop/bin is no longer resident; load a virtual "
                    "image first.",
                )
            yield entry
        finally:
            lock.release()
            self._release_entry(entry)

    def activate(self, key: tuple) -> None:
        """Mark a still-resident plan as the active eviction preference."""
        with self._master_lock:
            if key in self._master_cache:
                self._active_master_key = key
                self._master_cache.move_to_end(key)

    def virtual_image(
        self,
        key: tuple,
        entry: dict[str, Any],
        *,
        mode: str,
        inner: float,
        outer: float,
        center_row: float | None = None,
        center_column: float | None = None,
    ) -> np.ndarray:
        with self._resident_entry(key, entry) as resident:
            mode = str(mode)
            fit_row, fit_column, radius = self._bf_geometry(resident)
            row = fit_row if center_row is None else float(center_row)
            column = fit_column if center_column is None else float(center_column)
            inner_pixels = max(0.0, float(inner) * radius)
            outer_pixels = max(inner_pixels + 1.0, float(outer) * radius)
            detector_rows, detector_columns = resident["detector_shape"]
            rows, columns = np.ogrid[:detector_rows, :detector_columns]
            distance_squared = (rows - row) ** 2 + (columns - column) ** 2
            if mode == "BF":
                mask = distance_squared <= outer_pixels**2
            elif mode in {"ABF", "ADF", "HAADF", "DF"}:
                mask = (distance_squared >= inner_pixels**2) & (
                    distance_squared <= outer_pixels**2
                )
            elif mode in {"CoMx", "CoMy", "CoMmag", "DPC", "iCoM"}:
                with self._cuda_context(int(resident.get("gpu", self.gpu))):
                    if (
                        resident.get("com_row") is None
                        or resident.get("com_column") is None
                    ):
                        com_row, com_column = resident["compute"].center_of_mass()
                        resident["com_row"] = np.asarray(com_row, dtype=np.float32)
                        resident["com_column"] = np.asarray(com_column, dtype=np.float32)
                com_row = resident["com_row"].reshape(resident["scan_shape"])
                com_column = resident["com_column"].reshape(resident["scan_shape"])
                if mode == "CoMy":
                    return com_row
                if mode == "CoMx":
                    return com_column
                if mode in {"CoMmag", "DPC"}:
                    return np.hypot(com_row, com_column).astype(np.float32, copy=False)
                from quantem.gpu import dpc

                return np.asarray(dpc.integrate(com_row, com_column), dtype=np.float32)
            else:
                raise HTTPException(400, f"unknown virtual-image mode: {mode!r}")
            with self._cuda_context(int(resident.get("gpu", self.gpu))):
                result = resident["compute"].masked_sum_exact(
                    np.asarray(mask, dtype=bool)
                )
            return np.asarray(result, dtype=np.uint64).reshape(resident["scan_shape"])

    def custom_detector(
        self,
        key: tuple,
        entry: dict[str, Any],
        *,
        center_row: float,
        center_column: float,
        inner_radius: float,
        outer_radius: float,
        shape: str = "annulus",
    ) -> np.ndarray:
        with self._resident_entry(key, entry) as resident:
            detector_rows, detector_columns = resident["detector_shape"]
            rows, columns = np.ogrid[:detector_rows, :detector_columns]
            if shape == "square":
                mask = (
                    (np.abs(rows - center_row) <= outer_radius)
                    & (np.abs(columns - center_column) <= outer_radius)
                )
            else:
                distance_squared = (rows - center_row) ** 2 + (
                    columns - center_column
                ) ** 2
                mask = distance_squared <= outer_radius**2
                if shape == "annulus":
                    mask &= distance_squared >= inner_radius**2
            with self._cuda_context(int(resident.get("gpu", self.gpu))):
                result = resident["compute"].masked_sum_exact(
                    np.asarray(mask, dtype=bool)
                )
            return np.asarray(result, dtype=np.uint64).reshape(resident["scan_shape"])

    def scan_region_diffraction(
        self,
        session: str,
        filename: str,
        *,
        shape: str,
        center_row: float,
        center_column: float,
        radius: float,
        reduce: str,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
        expected: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return the exact detector-count sum over one resident scan ROI."""
        path = self.resolve_master(session, filename)
        scan_region = self._normalize_scan_region(path, scan_region)
        key = self._plan_key(path, det_bin, scan_bin, scan_region)
        with self._resident_entry(key, expected) as entry:
            indices = _scan_roi_indices(
                entry["scan_shape"],
                shape=shape,
                center_row=center_row,
                center_column=center_column,
                radius=radius,
            )
            compute = entry["compute"]
            reducer_name = (
                "reduce_frames_max" if reduce == "max" else "reduce_frames_exact"
            )
            reducer = getattr(compute, reducer_name, None)
            if reducer is None:
                raise HTTPException(
                    501, "This CUDA data type has no exact scan-ROI reducer."
                )
            with self._cuda_context(int(entry.get("gpu", self.gpu))):
                result = reducer(indices)
            return (
                np.asarray(result, dtype=np.uint64).reshape(entry["detector_shape"]),
                int(indices.size),
            )

    def selected_diffraction(
        self,
        session: str,
        filename: str,
        *,
        scan_row: int,
        scan_column: int,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
        expected: dict[str, Any] | None = None,
    ) -> np.ndarray:
        path = self.resolve_master(session, filename)
        scan_region = self._normalize_scan_region(path, scan_region)
        key = self._plan_key(path, det_bin, scan_bin, scan_region)
        with self._resident_entry(key, expected) as entry:
            scan_rows, scan_columns = entry["scan_shape"]
            row = max(0, min(scan_rows - 1, int(scan_row)))
            column = max(0, min(scan_columns - 1, int(scan_column)))
            with self._cuda_context(int(entry.get("gpu", self.gpu))):
                frame = entry["data"][row, column]
                if hasattr(frame, "get"):
                    frame = frame.get()
                elif hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                return np.asarray(frame)

    def cached_image(self, key: tuple) -> np.ndarray | None:
        with self._image_lock:
            image = self._image_cache.get(key)
            if image is not None:
                self._image_cache.move_to_end(key)
            return image

    def store_image(self, key: tuple, image: np.ndarray) -> None:
        with self._image_lock:
            self._image_cache[key] = image
            self._image_cache.move_to_end(key)
            while len(self._image_cache) > _MAX_IMAGE_ENTRIES:
                self._image_cache.popitem(last=False)

    def close(self) -> None:
        self._load_executor.shutdown(wait=True, cancel_futures=True)
        with self._master_lock:
            entries = list(self._master_cache.values())
            self._master_cache.clear()
        for entry in entries:
            self._close_entry(entry)
        for gpu in self._available_gpus():
            self._flush_cuda_pool(gpu)


def create_app(
    data_folder: str | os.PathLike[str],
    *,
    gpu: int = 0,
    gpus: Sequence[int] | str | None = None,
    service: BrowseService | None = None,
    ssb_service: SSBProtocolService | None = None,
    maped_service: MAPEDProtocolService | None = None,
    implementation_revision: str | None = None,
) -> FastAPI:
    """Create the loopback remote-viewer application."""
    browse = service or BrowseService(data_folder, gpu=gpu, gpus=gpus)
    resolved_revision = (
        implementation_revision
        or getattr(ssb_service, "implementation_revision", None)
        or os.environ.get("QUANTEM_GPU_IMPLEMENTATION_REVISION", "unrecorded")
    )
    ssb = ssb_service or SSBProtocolService(
        data_folder,
        available_gpus=browse._available_gpus,
        device_name=browse._device_name,
        implementation_revision=resolved_revision,
    )
    maped = maped_service or MAPEDProtocolService(
        data_folder,
        available_gpus=browse._available_gpus,
        device_name=browse._device_name,
        implementation_revision=resolved_revision,
    )
    if (
        resolved_revision != "unrecorded"
        and maped.implementation_revision != resolved_revision
    ):
        raise ValueError(
            "MAPED implementation revision does not match the served revision."
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await asyncio.to_thread(ssb.close)
            await asyncio.to_thread(browse.close)

    app = FastAPI(
        title="QuantEM GPU Remote Browse",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.browse_service = browse
    app.state.ssb_service = ssb
    app.state.maped_service = maped

    def maped_http_error(exc: MAPEDProtocolError) -> HTTPException:
        return HTTPException(
            exc.status_code,
            detail={
                "code": exc.code,
                "message": str(exc),
                "recoverySuggestion": exc.recovery,
                "stage": exc.stage,
            },
        )

    def resident_result(
        session: str,
        filename: str,
        *,
        det_bin: int,
        scan_bin: int,
        scan_region: tuple[int, int, int, int] | None,
        operation: Callable[[tuple, dict[str, Any]], Any],
    ) -> tuple[tuple, Any]:
        """Keep one exact plan resident through its requested calculation."""
        for attempt in range(3):
            key, entry = browse.entry(
                session,
                filename,
                det_bin=det_bin,
                scan_bin=scan_bin,
                scan_region=scan_region,
                reserve=True,
            )
            try:
                result = operation(key, entry)
            except HTTPException as exc:
                if exc.status_code != 409 or attempt == 2:
                    raise
            else:
                browse.activate(key)
                return key, result
            finally:
                browse._release_entry(entry)
        raise AssertionError("resident retry loop did not return or raise")

    @app.get("/api/browse/capabilities")
    async def capabilities() -> dict[str, Any]:
        result = browse.capabilities()
        if ssb.backend_kind == "local_mps":
            capability = ssb.capability()
            device = capability["device"]
            result = {
                "protocol": PROTOCOL_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "backend": "mps" if capability["ready"] else None,
                "device_name": device["deviceName"],
                "device_error": capability["unavailableReason"],
                "devices": [device] if capability["ready"] else [],
                "data_folders": [str(browse.data_folder)],
                "features": {"ssb": capability},
            }
        else:
            result["features"]["ssb"] = ssb.capability()
            result["features"]["maped"] = maped.advertised_capability()
        return result

    @app.post("/api/maped/inventory")
    async def maped_inventory(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(maped.inventory, request)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

    @app.post("/api/maped/snapshot")
    async def maped_snapshot(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(maped.snapshot, request)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

    @app.post("/api/maped/previews")
    async def maped_previews(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(maped.previews, request)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

    @app.post("/api/maped/diffraction")
    async def maped_diffraction(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(maped.selected_diffraction, request)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

    @app.get("/api/maped/payloads/{payload_id}")
    async def maped_payload(payload_id: str) -> Response:
        try:
            payload, descriptor = maped.payload(payload_id)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "X-Width": str(descriptor["shape"]["columns"]),
                "X-Height": str(descriptor["shape"]["rows"]),
                "X-Dtype": descriptor["dtype"],
                "X-Byte-Count": str(descriptor["byteCount"]),
                "X-SHA256": descriptor["sha256"],
                "X-Generation": str(descriptor["generation"]),
                "X-Source-Identity-SHA256": descriptor["sourceIdentity"][
                    "sourceIdentitySHA256"
                ],
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/maped/cache/validate")
    async def maped_validate_cache(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(maped.validate_cache, request)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

    @app.post("/api/maped/runs", status_code=202)
    async def maped_start_run(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return maped.start_run(request)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

    @app.get("/api/maped/runs/{run_id}/events")
    async def maped_run_events(
        run_id: str, afterSequence: int = 0
    ) -> StreamingResponse:
        try:
            maped.run_event_snapshot(run_id)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

        def stream():
            for event in maped.run_events(run_id, after_sequence=afterSequence):
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.delete("/api/maped/runs/{run_id}", status_code=202)
    async def maped_cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return maped.cancel_run(run_id)
        except MAPEDProtocolError as exc:
            raise maped_http_error(exc) from exc

    @app.get("/api/ssb/source-identity")
    async def ssb_source_identity(master_path: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(ssb.source_identity, master_path)
        except SSBProtocolError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/ssb/prepare")
    async def ssb_prepare(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(ssb.prepare, request)
        except (SSBProtocolError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/ssb/reconstruct")
    async def ssb_reconstruct(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(ssb.reconstruct, request)
        except SSBProtocolError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/ssb/interactive/sessions", status_code=201)
    async def ssb_open_interactive_session(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(ssb.open_interactive_session, request)
        except (SSBProtocolError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/ssb/interactive/sessions/{session_id}")
    async def ssb_close_interactive_session(session_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(ssb.close_interactive_session, session_id)
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/ssb/interactive/sessions/{session_id}")
    async def ssb_interactive_session(session_id: str) -> dict[str, Any]:
        try:
            return ssb.interactive_session_snapshot(session_id)
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/ssb/interactive/jobs", status_code=202)
    async def ssb_submit_interactive(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return ssb.submit_interactive(request)
        except (SSBProtocolError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/ssb/interactive/fits", status_code=202)
    async def ssb_submit_interactive_fit(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return ssb.submit_interactive_fit(request)
        except (SSBProtocolError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/ssb/interactive/jobs/{job_id}")
    async def ssb_interactive_job(job_id: str, generation: int) -> dict[str, Any]:
        try:
            return ssb.interactive_job_snapshot(job_id, generation)
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/ssb/interactive/jobs/{job_id}", status_code=202)
    async def ssb_cancel_interactive(job_id: str, generation: int) -> dict[str, Any]:
        try:
            return ssb.cancel_interactive_job(job_id, generation)
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/ssb/interactive/jobs/{job_id}/phase")
    async def ssb_interactive_phase(job_id: str, generation: int) -> Response:
        try:
            payload, descriptor = ssb.interactive_payload(job_id, generation)
        except SSBPayloadNotReady as exc:
            raise HTTPException(425, str(exc)) from exc
        except SSBPayloadUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "X-Width": str(descriptor["shape"]["columns"]),
                "X-Height": str(descriptor["shape"]["rows"]),
                "X-Dtype": descriptor["dtype"],
                "X-Byte-Count": str(descriptor["byteCount"]),
                "X-SHA256": descriptor["sha256"],
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/ssb/jobs", status_code=202)
    async def ssb_submit(request: dict[str, Any]) -> dict[str, Any]:
        try:
            return ssb.submit(request)
        except (SSBProtocolError, ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/ssb/jobs/{job_id}")
    async def ssb_job(job_id: str, generation: int) -> dict[str, Any]:
        try:
            return ssb.job_snapshot(job_id, generation)
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/ssb/jobs/{job_id}", status_code=202)
    async def ssb_cancel(job_id: str, generation: int) -> dict[str, Any]:
        try:
            return ssb.cancel_job(job_id, generation)
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/ssb/jobs/{job_id}/phase")
    async def ssb_phase(job_id: str, generation: int) -> Response:
        try:
            payload, descriptor = ssb.payload(job_id, generation)
        except SSBPayloadNotReady as exc:
            raise HTTPException(425, str(exc)) from exc
        except SSBPayloadUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc
        except (SSBProtocolError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "X-Width": str(descriptor["shape"]["columns"]),
                "X-Height": str(descriptor["shape"]["rows"]),
                "X-Dtype": descriptor["dtype"],
                "X-Byte-Count": str(descriptor["byteCount"]),
                "X-SHA256": descriptor["sha256"],
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/browse/sessions")
    async def sessions(refresh: bool = False) -> dict[str, Any]:
        return await asyncio.to_thread(browse.sessions, refresh=refresh)

    @app.get("/api/browse/acquisitions")
    async def acquisitions() -> dict[str, Any]:
        return await asyncio.to_thread(browse.acquisitions)

    @app.get("/api/browse/cbed")
    async def cbed(
        session: str,
        file: str,
        sx: int = 0,
        sy: int = 0,
        det_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
        ensure_resident: bool = False,
    ) -> Response:
        region = _scan_region(row_start, row_stop, column_start, column_stop)
        if ensure_resident:
            _, image = await asyncio.to_thread(
                resident_result,
                session,
                file,
                det_bin=max(1, det_bin),
                scan_bin=scan_bin,
                scan_region=region,
                operation=lambda _key, entry: browse.selected_diffraction(
                    session,
                    file,
                    scan_row=sx,
                    scan_column=sy,
                    det_bin=max(1, det_bin),
                    scan_bin=scan_bin,
                    scan_region=region,
                    expected=entry,
                ),
            )
        else:
            image = await asyncio.to_thread(
                browse.selected_diffraction,
                session,
                file,
                scan_row=sx,
                scan_column=sy,
                det_bin=max(1, det_bin),
                scan_bin=scan_bin,
                scan_region=region,
            )
        return _image_response(image, cache_control="max-age=60")

    @app.get("/api/browse/realspace")
    async def realspace(
        session: str,
        file: str,
        mode: str = "BF",
        inner: float = 0.0,
        outer: float = 1.0,
        cx: float | None = None,
        cy: float | None = None,
        det_bin: int = 1,
        dtype: str = "uint16",
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
        ensure_resident: bool = False,
    ) -> Response:
        del dtype
        region = _scan_region(row_start, row_stop, column_start, column_stop)
        image_key = (
            session,
            file,
            max(1, det_bin),
            scan_bin,
            region,
            mode,
            round(inner, 5),
            round(outer, 5),
            None if cy is None else round(cy, 3),
            None if cx is None else round(cx, 3),
        )
        cached = browse.cached_image(image_key)
        if cached is not None and not ensure_resident:
            return _image_response(cached)

        def compute(key: tuple, entry: dict[str, Any]) -> np.ndarray:
            if cached is not None:
                return cached
            return browse.virtual_image(
                key,
                entry,
                mode=mode,
                inner=inner,
                outer=outer,
                center_row=cy,
                center_column=cx,
            )

        key, cached = await asyncio.to_thread(
            resident_result,
            session,
            file,
            det_bin=max(1, det_bin),
            scan_bin=scan_bin,
            scan_region=region,
            operation=compute,
        )
        if browse.cached_image(image_key) is None:
            browse.store_image(image_key, cached)
        browse.mark_active(key)
        return _image_response(cached)

    @app.get("/api/browse/realspace-shape")
    async def realspace_shape(
        session: str,
        file: str,
        shape: str = "annulus",
        cx: float = 0.0,
        cy: float = 0.0,
        inner: float = 0.0,
        outer: float = 0.0,
        det_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
    ) -> Response:
        if shape not in {"circle", "square", "annulus"}:
            raise HTTPException(400, "detector shape must be circle, square, or annulus")
        if not all(math.isfinite(value) for value in (cx, cy, inner, outer)):
            raise HTTPException(400, "detector center and radii must be finite")
        if outer <= 0 or (shape == "annulus" and (inner < 0 or outer <= inner)):
            raise HTTPException(400, "detector radii must satisfy 0 <= inner < outer")
        region = _scan_region(row_start, row_stop, column_start, column_stop)
        key, image = await asyncio.to_thread(
            resident_result,
            session,
            file,
            det_bin=max(1, det_bin),
            scan_bin=scan_bin,
            scan_region=region,
            operation=lambda key, entry: browse.custom_detector(
                key,
                entry,
                center_row=cy,
                center_column=cx,
                inner_radius=inner,
                outer_radius=outer,
                shape=shape,
            ),
        )
        browse.mark_active(key)
        return _image_response(image)

    @app.get("/api/browse/cbed-region")
    async def cbed_region(
        session: str,
        file: str,
        shape: str = "circle",
        cx: float = 0.0,
        cy: float = 0.0,
        radius: float = 1.0,
        reduce: str = "mean",
        det_bin: int = 1,
        scan_bin: int = 1,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
        ensure_resident: bool = False,
    ) -> Response:
        if reduce not in {"mean", "sum", "max"}:
            raise HTTPException(400, "scan ROI reduction must be mean, sum, or max")
        region = _scan_region(row_start, row_stop, column_start, column_stop)
        if ensure_resident:
            _, (image, count) = await asyncio.to_thread(
                resident_result,
                session,
                file,
                det_bin=max(1, det_bin),
                scan_bin=scan_bin,
                scan_region=region,
                operation=lambda _key, entry: browse.scan_region_diffraction(
                    session,
                    file,
                    shape=shape,
                    center_row=cy,
                    center_column=cx,
                    radius=radius,
                    reduce=reduce,
                    det_bin=max(1, det_bin),
                    scan_bin=scan_bin,
                    scan_region=region,
                    expected=entry,
                ),
            )
        else:
            image, count = await asyncio.to_thread(
                browse.scan_region_diffraction,
                session,
                file,
                shape=shape,
                center_row=cy,
                center_column=cx,
                radius=radius,
                reduce=reduce,
                det_bin=max(1, det_bin),
                scan_bin=scan_bin,
                scan_region=region,
            )
        return _image_response(
            image,
            cache_control="no-store",
            value_divisor=count if reduce == "mean" else 1,
        )

    return app


__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "BrowseService",
    "create_app",
]
