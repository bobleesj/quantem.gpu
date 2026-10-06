"""The served folder's acquisitions: session folders, masters and their readiness.

Nothing here touches a GPU. The catalog groups every ``*_master.h5`` under the
served folder into session folders, inspects each master once per change of
its files, and reports which acquisitions are still being written.
"""

import hashlib
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
from fastapi import HTTPException

from quantem.gpu import io
from quantem.gpu.io.inspect import Inspection

# Walking a large data folder on every poll would stall the client.
RESCAN_INTERVAL_SECONDS = 10.0

logger = logging.getLogger("quantem.gpu.remote")


class Catalog:
    """Discover, inspect and describe the acquisitions of one data folder.

    Parameters
    ----------
    data_folder : Path
        Resolved root of the served acquisitions. Folders whose names start
        with ``.`` or ``_`` are not served.

    Examples
    --------
    >>> catalog = Catalog(Path("/data").resolve())
    >>> catalog.sessions()["sessions"][0]["files"][0]["name"]
    'sample_master.h5'
    """

    def __init__(self, data_folder: Path) -> None:
        self.data_folder = data_folder
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._generation = 0
        self._catalog: dict[str, object] | None = None
        self._session_paths: dict[str, Path] = {}
        self._inspection_lock = threading.Lock()
        self._inspections: dict[str, tuple[tuple, Inspection]] = {}
        self._discovery_lock = threading.Lock()
        self._masters_lock = threading.Lock()
        self._known_masters: tuple[Path, ...] | None = None
        self._scan_in_flight = False
        self._last_scan_completed = 0.0

    def sessions(self, *, refresh: bool = False) -> dict[str, object]:
        """Return the cached catalog, building it on first use or when asked."""
        with self._lock:
            cached = self._catalog
        if cached is None or refresh:
            return self.refresh()
        return cached

    def refresh(self) -> dict[str, object]:
        """Rebuild the catalog, or reuse a rebuild that finished while this waited."""
        with self._lock:
            requested_generation = self._generation
        with self._refresh_lock:
            with self._lock:
                if self._catalog is not None and self._generation != requested_generation:
                    return self._catalog
            return self._rebuild()

    def acquisitions(self) -> dict[str, object]:
        """Report acquisitions still being written and those ready to load.

        ``ready_token`` changes whenever a ready acquisition changes, so a
        client can poll cheaply and refresh only on a new token.
        """
        pending: list[dict[str, object]] = []
        ready: list[dict[str, object]] = []
        ready_signatures: list[tuple[str, tuple]] = []
        for master in self._watched_masters():
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
        """Map a catalogued session path and master name to a file; clients send no paths."""
        if not filename or Path(filename).name != filename:
            raise HTTPException(400, "file must be one master filename")
        with self._lock:
            directory = self._session_paths.get(session)
        if directory is None:
            self.refresh()
            with self._lock:
                directory = self._session_paths.get(session)
        if directory is None:
            raise HTTPException(404, f"unknown session: {session}")
        path = directory / filename
        if not path.is_file() or path.parent != directory:
            raise HTTPException(404, f"master not found: {filename}")
        return path

    def inspect(self, master: Path) -> Inspection:
        """Inspect a master, reusing the result while its files are unchanged."""
        signature = file_signature(master)
        key = str(master)
        with self._inspection_lock:
            cached = self._inspections.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]
        inspection = io.inspect(master)
        with self._inspection_lock:
            self._inspections[key] = (signature, inspection)
        return inspection

    # --- Discovery and inspection

    def _rebuild(self) -> dict[str, object]:
        """Group masters into session folders and describe each file for the client.

        A session is the folder holding masters; its ``(source, date)`` labels
        are the last two parts of its path below the served root.
        """
        grouped: dict[str, tuple[str, str, list[Path]]] = {}
        session_paths: dict[str, Path] = {}
        for master in self._scan_masters():
            relative = master.parent.relative_to(self.data_folder)
            source, date = (self.data_folder.parent.name, self.data_folder.name, *relative.parts)[-2:]
            path = relative.as_posix()
            grouped.setdefault(path, (source, date, []))[2].append(master)
            session_paths[path] = master.parent

        sessions: list[dict[str, object]] = []
        for path, (source, date, masters) in sorted(grouped.items()):
            files: list[dict[str, object]] = []
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
                    size = sum(item[1] for item in file_signature(master))
                metadata = inspection.metadata or {}
                scan_sampling = _sampling(metadata, "scan_sampling_A")
                files.append(
                    {
                        "name": master.name,
                        "shape": [*scan_shape, *detector_shape],
                        "dtype": inspection.dtype,
                        "cal": "ok" if scan_sampling is not None else "un",
                        "size": format_size(size),
                        "size_bytes": size,
                        "loadable": bool(inspection.ready),
                        "load_status": {
                            "loadable": bool(inspection.ready),
                            "reason": inspection.reason,
                            "action": inspection.action,
                        },
                        "scan_sampling_A": scan_sampling,
                        "k_pixel_size_mrad": _sampling(metadata, "k_pixel_size_mrad"),
                        "has_ssb": False,
                    }
                )
            if files:
                sessions.append({"path": path, "source": source, "date": date, "files": files})

        payload = {
            "data_folder": str(self.data_folder),
            "data_folders": [str(self.data_folder)],
            "sessions": sessions,
            "complete": True,
        }
        with self._lock:
            self._catalog = payload
            self._session_paths = session_paths
            self._generation += 1
        return payload

    def _scan_masters(self) -> list[Path]:
        """Walk the data folder once, skipping hidden and underscore-prefixed folders, and remember the masters."""
        with self._discovery_lock:
            masters = (
                sorted(
                    path
                    for path in self.data_folder.rglob("*_master.h5")
                    if not path.name.startswith("._")
                    and not any(
                        part.startswith((".", "_"))
                        for part in path.relative_to(self.data_folder).parts
                    )
                )
                if self.data_folder.is_dir()
                else []
            )
        with self._masters_lock:
            self._known_masters = tuple(masters)
            self._last_scan_completed = time.monotonic()
            self._scan_in_flight = False
        return masters

    def _watched_masters(self) -> list[Path]:
        """Return the known masters at once, rescanning in the background when stale.

        A rescan runs at most every ``RESCAN_INTERVAL_SECONDS`` and only one at
        a time, so polling never waits for a walk of the data folder.
        """
        with self._masters_lock:
            known = self._known_masters
            should_scan = (
                known is not None
                and not self._scan_in_flight
                and time.monotonic() - self._last_scan_completed >= RESCAN_INTERVAL_SECONDS
            )
            if should_scan:
                self._scan_in_flight = True
        if known is None:
            return self._scan_masters()
        if should_scan:
            threading.Thread(
                target=self._finish_background_scan,
                name="quantem-gpu-master-discovery",
                daemon=True,
            ).start()
        return list(known)

    def _finish_background_scan(self) -> None:
        """Run one rescan in its own thread and clear the in-flight flag if it fails."""
        try:
            self._scan_masters()
        except OSError:
            # A failed rescan must not end discovery for the life of the
            # service; log it and let the next poll scan again.
            logger.exception("background master discovery failed")
            with self._masters_lock:
                self._last_scan_completed = time.monotonic()
                self._scan_in_flight = False

    def _watch_inspection(self, master: Path) -> Inspection:
        """Reuse an inspection while its files are unchanged, polling acquisitions still being written.

        A ready acquisition is checked by the master's size and modification
        time alone; one still being written by every member's, so a new shard
        triggers a fresh inspection.
        """
        key = str(master)
        with self._inspection_lock:
            cached = self._inspections.get(key)
        if cached is None:
            return self.inspect(master)
        signature, inspection = cached
        if not inspection.ready:
            source_files = (inspection.source_signature or {}).get("files", [])
            if source_files and _members_unchanged(source_files):
                return inspection
            return self.inspect(master)
        master_signature = next((item for item in signature if item[0] == key), None)
        try:
            stat = master.stat()
        except OSError:
            return self.inspect(master)
        if master_signature is None or master_signature[1:] != (
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ):
            return self.inspect(master)
        return inspection


# --- Primitives


def file_signature(master: Path) -> tuple[tuple[str, int, int], ...]:
    """Return (path, size, mtime) for the master and every member it reads.

    Cached inspections and resident acquisitions compare this signature to
    notice an acquisition that is still being written or was replaced.
    """
    paths = [master]
    stem = master.name.removesuffix("_master.h5")
    paths.extend(sorted(master.parent.glob(f"{stem}_data_*.h5")))
    try:
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
        # A master still being written may not open yet; its globbed members suffice.
        pass
    signature: list[tuple[str, int, int]] = []
    for path in dict.fromkeys(paths):
        try:
            stat = path.stat()
        except OSError:
            continue
        signature.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(signature)


def format_size(n_bytes: int) -> str:
    """Format a byte count with one decimal in the largest fitting binary unit."""
    for label, scale in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.1f} {label}"
    return f"{n_bytes} B"


def _members_unchanged(source_files: list[dict[str, object]]) -> bool:
    """Check an incomplete acquisition's members with metadata-only file stats."""
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


def _sampling(metadata: dict[str, object], name: str) -> float | None:
    """Return one numeric calibration from inspection metadata, or None when absent."""
    value = metadata.get(name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
