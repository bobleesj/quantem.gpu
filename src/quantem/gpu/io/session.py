"""A streamed session: its scans as one sequence of patterns by arrival index, with the microscope's state.

A receiver that streams scans numbers every pattern it keeps, from 0, across the session's scans. Each saved scan's
header carries ``session``, ``first_index`` and ``begin_epoch_s``; the folder may also hold ``index.jsonl`` (one
checked line per kept scan, written by the receiver) and ``session-log.jsonl`` (the microscope's readings, written
when one changed and once a minute). ``session(folder)`` reads those and joins them by time.

>>> from quantem.gpu import io
>>> session = io.session("scans/2026-11-20")            # doctest: +SKIP
>>> session.scans[3]                                    # doctest: +SKIP
>>> session.state(300_000)["defocus_nm"]                # doctest: +SKIP
{'value': -12.0, 'age_s': 3.2, 'between': None, 'indices': None}
>>> session.patterns(0, 512)                            # doctest: +SKIP
>>> session.report()                                    # doctest: +SKIP    one row per scan, with its readings
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import math
from os import PathLike
from pathlib import Path
from statistics import median
import zlib

INDEX_FILE = "index.jsonl"
LOG_FILE = "session-log.jsonl"
_NOT_READINGS = frozenset({"session", "event", "time_utc", "source", "changed"})
_NOT_SET = _NOT_READINGS | {"series_id", "beam_on_epoch_s"}


def _index_rows(path: Path) -> list[dict]:
    """The checked lines of an index.jsonl; a cut-off last line is dropped, a damaged earlier one raises."""
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    found = []
    for number, line in enumerate(lines):
        try:
            record = json.loads(line)
            crc = record.pop("crc32")
            if crc != zlib.crc32(json.dumps(record, allow_nan=False, sort_keys=True).encode()):
                raise ValueError("checksum")
        except (ValueError, KeyError, AttributeError) as error:
            if number == len(lines) - 1:
                break
            raise ValueError(f"{path}: line {number + 1} is damaged ({error}).") from error
        found.append(record)
    return found


def _log_rows(source) -> list[dict]:
    """Every event of one or more session-log.jsonl files (a file, a folder holding one, or several of either)."""
    sources = [source] if isinstance(source, (str, PathLike)) else list(source)
    found = []
    for item in sources:
        path = Path(item)
        path = path / LOG_FILE if path.is_dir() else path
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    found.append(json.loads(line))
    return found


def _epoch_s(time_utc: str) -> float:
    """Seconds since the epoch of an ISO time."""
    return datetime.fromisoformat(time_utc).timestamp()


def _row_time(row: dict) -> float:
    """When a reading was written, in seconds since the epoch."""
    return row["_epoch_s"]


def _first_index(row: dict) -> int:
    """A scan's first index: the order of a session's scans."""
    return row["first_index"]


class _Scans(Sequence):
    """A session's scans in arrival order; one is loaded when it is asked for."""

    def __init__(self, owner: "Session"):
        self._owner = owner

    def __len__(self) -> int:
        return len(self._owner.index)

    def __getitem__(self, number):
        if isinstance(number, slice):
            return [self[item] for item in range(*number.indices(len(self)))]
        return self._owner._open(self._owner.index[number])

    def __repr__(self) -> str:
        return f"<{len(self)} scans of session {self._owner.id}>"


class Session:
    """One session's scans by arrival index. Made by ``io.session(folder)``.

    ``index`` is one row per kept scan (``first_index``, ``received``, ``series``, ``scan_pattern``, ``path`` or
    None when its file is not in the folder). ``clock_offset_s`` is the receiver's clock minus the microscope PC's,
    from the scans whose start both logged, and ``clock_spread_s`` how far those scans disagree.
    """

    def __init__(self, folder: Path, identifier: str, index: list[dict], log: list[dict], *, read_period_s: float,
                 device=None):
        self.folder, self.id, self.index, self.device = folder, identifier, index, device
        self.read_period_s = float(read_period_s)
        # a scan's own line holds what was set for that scan alone, exact at its first pattern: a reading at beam-on
        set_at_start = [{**{name: value for name, value in row.items() if name not in _NOT_SET},
                         "_epoch_s": row["beam_on_epoch_s"], "_exact": True}
                        for row in log if row.get("event") == "scan_beam_on" and set(row) - _NOT_SET - {"_epoch_s"}]
        self.readings = sorted([row for row in log if row.get("event") == "reading"] + set_at_start, key=_row_time)
        self._reading_times = [row["_epoch_s"] for row in self.readings]
        beam_on = {row.get("series_id"): row["beam_on_epoch_s"] for row in log
                   if row.get("event") in ("scan_startup", "scan_beam_on") and row.get("beam_on_epoch_s") is not None}
        offsets = [row["begin_epoch_s"] - beam_on[row.get("series_id")] for row in index
                   if row.get("begin_epoch_s") is not None and row.get("series_id") in beam_on]
        self.clock_offset_s = median(offsets) if offsets else None
        self.clock_spread_s = max(offsets) - min(offsets) if offsets else None
        for row in index:
            # on the microscope PC's clock, where the readings are: its own start when logged, else the receiver's
            # start moved by the offset the other scans gave
            if row.get("series_id") in beam_on:
                row["start_epoch_s"], row["clock"] = beam_on[row["series_id"]], "logged"
            elif row.get("begin_epoch_s") is not None and self.clock_offset_s is not None:
                row["start_epoch_s"], row["clock"] = row["begin_epoch_s"] - self.clock_offset_s, "offset"
            else:
                row["start_epoch_s"], row["clock"] = row.get("begin_epoch_s"), "receiver"
        self.scans = _Scans(self)
        self._opened: tuple[int, object] | None = None
        self.count = max((row["first_index"] + row["received"] for row in index), default=0)

    def __repr__(self) -> str:
        here = sum(row["path"] is not None for row in self.index)
        clock = ("no clock pair" if self.clock_offset_s is None
                 else f"clock offset {self.clock_offset_s:+.3f} s (spread {self.clock_spread_s:.3f} s)")
        return (f"Session {self.id}: {len(self.index)} scans, {self.count:,} patterns, {here} of {len(self.index)} "
                f"files in {self.folder}, {len(self.readings)} readings, {clock}")

    # --- index -> scan, time ---------------------------------------------------------------------------------

    def _row(self, index: int) -> dict:
        """The scan a pattern index belongs to."""
        index = int(index)
        firsts = [row["first_index"] for row in self.index]
        place = bisect_right(firsts, index) - 1
        if place < 0 or index >= self.index[place]["first_index"] + self.index[place]["received"]:
            raise IndexError(f"No pattern {index}: this session holds 0 to {self.count - 1}"
                             + ("" if place < 0 or index >= self.count else ", and no kept scan covers that index") + ".")
        return self.index[place]

    def time(self, index: int) -> float | None:
        """When pattern ``index`` was taken, seconds since the epoch on the microscope PC's clock (the receiver's
        when no scan logged both): its scan's start plus its place times the dwell. None when the scan has no start."""
        row = self._row(index)
        if row["start_epoch_s"] is None or row.get("dwell_s") is None:
            return row["start_epoch_s"]
        return row["start_epoch_s"] + (int(index) - row["first_index"]) * float(row["dwell_s"])

    def _index_at(self, row: dict, epoch_s: float) -> int:
        """The pattern of this scan being taken at a time, kept inside the scan."""
        place = math.floor((epoch_s - row["start_epoch_s"]) / float(row["dwell_s"]))
        return row["first_index"] + min(max(place, 0), row["received"] - 1)

    # --- state ------------------------------------------------------------------------------------------------

    def state(self, index: int) -> dict:
        """The microscope's readings at pattern ``index``: ``{name: {"value", "age_s", "between", "indices"}}``.

        ``value`` is the last reading written at or before the pattern and ``age_s`` how old it was. A reading that
        changed is known only to one read period (``read_period_s``) and the clocks' spread: a pattern inside that
        window has ``between=(old, new)`` and ``indices=(first, last)``, the patterns of its scan the change may lie
        among. Without a session log the values are the scan's own header (read once when it was started), with
        ``age_s`` None. What the acquisition set for one scan alone (a pair's rotation, a tilt of a series) is in that
        scan's own log line and holds from its first pattern, with ``age_s`` counted from its beam-on.
        """
        row = self._row(index)
        when = self.time(index)
        if not self.readings or when is None:
            optics = row.get("optics") or {}
            return {name: {"value": value, "age_s": None, "between": None, "indices": None}
                    for name, value in optics.items()}
        spread = 0.0 if row["clock"] == "logged" else float(self.clock_spread_s or 0.0)
        place = bisect_right(self._reading_times, when)
        found: dict[str, dict] = {}
        for reading in reversed(self.readings[:place]):         # the newest reading of each name
            for name, value in reading.items():
                if name not in _NOT_READINGS and not name.startswith("_") and name not in found:
                    found[name] = {"value": value, "age_s": when - reading["_epoch_s"], "between": None,
                                   "indices": None}
        dwell = row.get("dwell_s")
        for number in range(max(place - 1, 0), len(self.readings)):
            reading = self.readings[number]
            if reading.get("_exact"):
                continue                            # set for one scan before its first pattern: no window
            opens = reading["_epoch_s"] - self.read_period_s - spread
            if opens > when:
                break
            if when > reading["_epoch_s"] + spread:
                continue
            for name in reading.get("changed") or ():
                if name not in reading:
                    continue
                before = self._before(number, name)
                entry = found.setdefault(name, {"value": before, "age_s": None, "between": None, "indices": None})
                entry["between"] = (before, reading[name])
                if dwell:
                    entry["indices"] = (self._index_at(row, opens), self._index_at(row, reading["_epoch_s"] + spread))
        return found

    def _before(self, number: int, name: str):
        """The value a name had in the newest reading before reading ``number``, or None."""
        for reading in reversed(self.readings[:number]):
            if name in reading:
                return reading[name]
        return None

    # --- counts -----------------------------------------------------------------------------------------------

    def _open(self, row: dict):
        """The scan of one index row, loaded (the last one stays open)."""
        from quantem.gpu.io.load import load

        if row["path"] is None:
            raise FileNotFoundError(f"Scan {row.get('series')} (patterns {row['first_index']} to "
                                    f"{row['first_index'] + row['received'] - 1}) is not in {self.folder}.")
        if self._opened is None or self._opened[0] != row["first_index"]:
            scan = load(row["path"], device=self.device, verbose=False)
            if self.readings:                       # the file's own header is left as it was written
                scan.metadata["state"] = self._scan_state(row)
            self._opened = (row["first_index"], scan)
        return self._opened[1]

    def _scan_state(self, row: dict) -> dict:
        """A scan's state at its first pattern, for its ``metadata["state"]``: the readings, how old each was, and
        the ones that were changing as it began."""
        state = self.state(row["first_index"])
        when = self.time(row["first_index"])
        return {"index": row["first_index"], "clock": row["clock"],
                "time_utc": None if when is None else datetime.fromtimestamp(when, timezone.utc).isoformat(),
                "readings": {name: entry["value"] for name, entry in state.items()},
                "age_s": {name: entry["age_s"] for name, entry in state.items()},
                "changing": {name: list(entry["between"]) for name, entry in state.items() if entry["between"]}}

    def report(self, names: Sequence[str] = ("defocus_nm", "beam_tilt_mrad", "scan_rotation_deg", "stage_um")):
        """One row per scan: its place in the session and the chosen readings at its first pattern, as a pandas
        table. A reading that was changing as the scan began is shown as ``(old, new)``.

        >>> session.report()                                    # doctest: +SKIP
        >>> session.report(names=("defocus_nm", "piezo_nm"))    # doctest: +SKIP
        """
        import pandas as pd

        table = []
        for number, row in enumerate(self.index):
            state = self.state(row["first_index"])
            when = self.time(row["first_index"])
            line = {"scan": number, "series": row.get("series"), "first_index": row["first_index"],
                    "patterns": row["received"], "scan_pattern": row.get("scan_pattern"),
                    "time_utc": None if when is None else datetime.fromtimestamp(when, timezone.utc).isoformat(),
                    "clock": row["clock"], "file": None if row["path"] is None else row["path"].name}
            for name in names:
                entry = state.get(name)
                line[name] = None if entry is None else (entry["between"] if entry["between"] else entry["value"])
            table.append(line)
        return pd.DataFrame(table)

    def patterns(self, start: int, stop: int):
        """The counts of patterns ``start`` to ``stop - 1`` in arrival order, one tensor (patterns, rows, cols) on
        the GPU. Raster and serpentine scans and scans saved as visits are read; any other pattern saved on its grid
        raises, naming the pattern."""
        import torch

        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.count:
            raise IndexError(f"Use 0 <= start < stop <= {self.count}; got {start}, {stop}.")
        pieces = []
        at = start
        while at < stop:
            row = self._row(at)
            first = row["first_index"]
            end = min(stop, first + row["received"])
            pieces.append(self._from_scan(row, at - first, end - first))
            at = end
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def _from_scan(self, row: dict, start: int, stop: int):
        """Patterns ``start`` to ``stop - 1`` of one scan, in arrival order."""
        scan = self._open(row)
        rows, cols = (int(side) for side in scan.shape[:2])
        pattern = row.get("scan_pattern") or "raster"
        if rows * cols != row["received"] or (rows > 1 and pattern not in ("raster", "serpentine")):
            raise NotImplementedError(f"Scan {row.get('series')} was taken as {pattern!r} and saved on its grid; "
                                      "reading it by arrival index is not built yet. Open it with session.scans[...].")
        top, bottom = start // cols, (stop - 1) // cols + 1
        counts = scan.read(scan_region=(top, bottom, 0, cols), output="torch")
        if pattern == "serpentine":                 # saved in raster order: every second line was taken backwards
            for line in range(top, bottom):
                if line % 2:
                    counts[line - top] = counts[line - top].flip(0)
        counts = counts.reshape(-1, *counts.shape[2:])
        return counts[start - top * cols:stop - top * cols]


def session(folder: str | PathLike[str], *, session: str | None = None, log=None, read_period_s: float = 0.2,
            device: int | str | None = None) -> Session:
    """Open a streamed session from the folder its scans were saved in.

    Parameters
    ----------
    folder
        Holds the session's saved scans (``.qem``) and, when they were copied with them, ``index.jsonl`` and
        ``session-log.jsonl``. Without ``index.jsonl`` the index is rebuilt from the scans' headers; scans that
        were never saved are then unknown.
    session
        Which session, when the folder holds scans of several.
    log
        The session log, when it is not in ``folder``: a file, a folder holding one, or several.
    read_period_s
        How often the microscope was read; a changed reading is placed within this time.

    Examples
    --------
    >>> session = io.session("scans/2026-11-20")        # doctest: +SKIP
    >>> session                                         # doctest: +SKIP
    Session 3f2a...: 12 scans, 3,145,728 patterns, 12 of 12 files in scans/2026-11-20, 418 readings, clock offset ...
    """
    from quantem.gpu.io.inspect import inspect

    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder of saved scans.")
    parts: dict[tuple[str, int], dict] = {}
    for path in sorted(folder.glob("*.qem")):
        metadata = inspect(path).metadata or {}
        if metadata.get("session") is None or metadata.get("first_index") is None:
            continue
        key = (str(metadata["session"]), int(metadata["first_index"]))
        if key in parts and "grid_of" in metadata:
            continue                                # the scan as it arrived is preferred to its copy on a grid
        parts[key] = {"path": path, "metadata": metadata}
    written = _index_rows(folder / INDEX_FILE) if (folder / INDEX_FILE).exists() else []
    written = [row for row in written if "first_index" in row]
    known = sorted({key[0] for key in parts} | {str(row["session"]) for row in written})
    if not known:
        raise ValueError(f"No session in {folder}: its scans carry no session and first_index, and there is no "
                         f"{INDEX_FILE}. Scans saved before 2026-11-19 open one at a time with io.load.")
    if session is None and len(known) > 1:
        raise ValueError(f"{folder} holds scans of {len(known)} sessions; choose one with session=: {known}.")
    identifier = str(session) if session is not None else known[0]
    if identifier not in known:
        raise ValueError(f"No session {identifier!r} in {folder}; it holds {known}.")
    index = [dict(row) for row in written if str(row["session"]) == identifier]
    listed = {row["first_index"] for row in index}
    for (owner, first), part in parts.items():      # a scan here that the index copy does not list yet, or no index
        if owner == identifier and first not in listed:
            metadata = part["metadata"]
            shape = metadata.get("scan_positions_shape") or metadata.get("scan_shape")
            frames = math.prod(int(side) for side in shape) if shape else None
            optics = metadata.get("optics") if isinstance(metadata.get("optics"), dict) else {}
            index.append({"session": identifier, "series_id": metadata.get("series_id"),
                          "series": metadata.get("series"), "first_index": first, "expected": frames,
                          "received": frames, "begin_epoch_s": metadata.get("begin_epoch_s"),
                          "scan_pattern": metadata.get("scan_pattern") or "raster",
                          "dwell_s": optics.get("dwell_s"), "flyback_s": optics.get("flyback_s")})
    index.sort(key=_first_index)
    for row in index:
        part = parts.get((identifier, row["first_index"]))
        row["path"] = None if part is None else part["path"]
        optics = None if part is None else part["metadata"].get("optics")
        row["optics"] = optics if isinstance(optics, dict) else None
        if row.get("received") is None:
            raise ValueError(f"{row['path']}: its header gives no scan shape, so its patterns cannot be counted.")
    if log is None and (folder / LOG_FILE).exists():
        log = folder / LOG_FILE
    events = [row for row in (_log_rows(log) if log is not None else []) if str(row.get("session")) == identifier]
    for row in events:
        if "time_utc" in row:
            row["_epoch_s"] = _epoch_s(row["time_utc"])
    events = [row for row in events if "_epoch_s" in row]
    return Session(folder, identifier, index, events, read_period_s=read_period_s, device=device)
