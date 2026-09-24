"""Convert Arina HDF5 acquisitions into verified ``.qem`` copies.

The conversion is ``io.load(master, representation="encoded")`` followed by
``io.save(destination, acquisition, format="quantem")``. This module adds what a
collection needs around that pair: finding acquisitions, recording which source
files a copy came from, carrying the session's calibrated beam from its
``dataset.yaml``, and comparing the saved copy with the source files.
Source files are opened read-only and are never modified or removed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
import zlib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_MASTER_SUFFIX = "_master.h5"
_STORED_DTYPES = ("uint8", "uint16", "uint32")
_HASH_BLOCK = 64 << 20
_READABLE_VALUES = 64
_EMBEDDED_MASTER_LIMIT = 8 << 20
_SESSION_FILE = "dataset.yaml"


@dataclass
class ConvertedAcquisition:
    """Outcome of converting one acquisition."""

    master: Path
    destination: Path
    source_bytes: int = 0
    qem_bytes: int | None = None
    seconds: float = 0.0
    skipped: str | None = None
    larger: bool = False
    master_embedded: bool = True
    failed: bool = False
    verification: dict = field(default_factory=dict)
    session_calibration: dict = field(default_factory=dict)

    @property
    def verified(self) -> bool | None:
        """True or False after verification; None when it did not run."""
        return self.verification.get("identical") if self.verification else None


def session_calibration(master: Path) -> tuple[dict, dict | None]:
    """Calibration an operator recorded for this acquisition in its session's ``dataset.yaml``.

    An Arina master records neither the probe semi-angle nor the scan step; the
    session file next to it does, once per session (``microscope``) and per
    magnification (``calibrations``) through the file's own ``files`` entry. The
    values become calibration overrides whose evidence is the session file's
    SHA-256, so every reader of the copy applies them ahead of the recorded
    values, and the fields used are attached as a JSON source document. Only an
    entry whose ``master`` is this file counts: a session can hold several series
    with the same numbers.

    Parameters
    ----------
    master : Path
        ARINA master file.

    Returns
    -------
    tuple[dict, dict | None]
        Overrides in calculation units (V, mrad, m) keyed by microscope path, and
        the source document, or ``({}, None)`` without a session file.

    Examples
    --------
    >>> session_calibration(Path("no/such/scan_master.h5"))
    ({}, None)
    """
    sidecar = Path(master).parent / _SESSION_FILE
    if not sidecar.is_file():
        return {}, None
    import yaml

    text = sidecar.read_bytes()
    document = yaml.safe_load(text) or {}
    evidence = f"{_SESSION_FILE} sha256:{hashlib.sha256(text).hexdigest()}"
    microscope = document.get("microscope") or {}
    files = document.get("files") or {}
    key, entry = next(((k, f) for k, f in files.items() if isinstance(f, dict) and f.get("master") == Path(master).name), (None, None))
    calibration = ((document.get("calibrations") or {}).get(entry.get("mag")) or {}) if entry else {}
    overrides = {}

    def override(path, value, factor, unit):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            overrides[path] = {"value": float(value) * factor, "unit": unit, "provenance": "user_override", "evidence": evidence}

    override("electron_source/accelerating_voltage", microscope.get("voltage_kV"), 1000, "V")
    override("illumination_system/semi_convergence_angle", microscope.get("semiangle_mrad"), 1, "mrad")
    for axis in ("row", "column"):
        override(f"scan_controller/regular_scan/pixel_size_{axis}", calibration.get("scan_sampling_A"), 1e-10, "m")
    if not overrides:
        return {}, None
    # only the fields this acquisition used: session notes can name people and links that do not belong in a portable copy
    used = {"session": (document.get("session") or {}).get("name"), "microscope": microscope,
            "file": {**(entry or {}), "key": key}, "calibration": calibration}
    content = json.dumps(used, sort_keys=True, default=str)
    attachment = {"filename": "dataset.json", "mediaType": "application/json", "content": content,
                  "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    return overrides, attachment


def find_masters(source: Path) -> list[Path]:
    """Return one master file, or every ``*_master.h5`` below a folder."""
    source = Path(source)
    if source.is_file():
        if not source.name.endswith(_MASTER_SUFFIX):
            raise ValueError(f"Expected a *_master.h5 acquisition; got {source.name!r}.")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"{source} is neither a master file nor a folder.")
    return sorted(
        path for path in source.rglob(f"*{_MASTER_SUFFIX}")
        if not path.name.startswith("._") and path.is_file()
    )


def detector_files(master: Path) -> list[Path]:
    """Return files declared by the master, not similarly named orphan files."""
    import h5py

    with h5py.File(master, "r") as handle:
        group = handle.get("entry/data")
        files = []
        if group is not None:
            for name in sorted(group):
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.ExternalLink) or link.path != "/entry/data/data":
                    raise ValueError("Collection conversion requires external detector chunks at /entry/data/data.")
                files.append((master.parent / link.filename).resolve())
    if len(set(files)) != len(files):
        raise ValueError("Repeated detector-file links are not supported by collection conversion.")
    return files


def destination_for(master: Path, source: Path, out: Path | None) -> Path:
    """Place the copy beside its master, or mirror the folder layout under ``out``."""
    name = master.name[: -len(_MASTER_SUFFIX)] + ".qem"
    if out is None:
        return master.with_name(name)
    source = Path(source)
    relative = master.parent.relative_to(source) if source.is_dir() else Path()
    return Path(out) / relative / name


def _stored_dtype(files: list[Path]) -> str:
    import h5py

    with h5py.File(files[0], "r") as handle:
        return handle["entry/data/data"].dtype.name


def _source_files(paths: list[Path]) -> list[dict]:
    """Record name, size and SHA-256 of every source file for the saved copy."""
    records = []
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while block := handle.read(_HASH_BLOCK):
                digest.update(block)
        records.append(dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return records


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def master_metadata(master: Path) -> dict:
    """Read every field and attribute of a master file under its HDF5 path.

    Attributes use ``path@name``, so ``.../count_time@units`` sits beside
    ``.../count_time``. Arrays longer than 64 values, such as the flatfield, are
    named with their shape here and travel complete inside the embedded master.
    """
    import h5py

    fields: dict = {}
    visited = set()

    def walk(group, prefix: str) -> None:
        if group.id in visited:
            fields[prefix] = "hard link to an already recorded group; complete in the embedded master"
            return
        visited.add(group.id)
        for name, value in group.attrs.items():
            fields[f"{prefix}@{name}"] = _json_value(value)
        for key in group:
            path = f"{prefix}/{key}".lstrip("/")
            link = group.get(key, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                fields[path] = f"external link to {link.filename}:{link.path}"
                continue
            if isinstance(link, h5py.SoftLink):
                fields[path] = f"soft link to {link.path}"
                continue
            item = group[key]
            if isinstance(item, h5py.Group):
                walk(item, path)
                continue
            for name, value in item.attrs.items():
                fields[f"{path}@{name}"] = _json_value(value)
            if item.size <= _READABLE_VALUES:
                fields[path] = _json_value(item[()])
            else:
                fields[path] = f"{item.dtype.name} array {tuple(item.shape)}; complete in the embedded master"

    with h5py.File(master, "r") as handle:
        walk(handle, "")
    description = str(fields.get("entry/instrument/detector/description", ""))
    fields["sourceFormat"] = (
        "dectris-arina-hdf5" if "ARINA" in description.upper() else "nexus-nxmx-hdf5"
    )
    return fields


def _embedded_master(master: Path) -> dict | None:
    """Carry the master file itself so that no field, table or attribute is lost."""
    if master.stat().st_size > _EMBEDDED_MASTER_LIMIT:
        return None
    packed = base64.b64encode(zlib.compress(master.read_bytes(), 9)).decode("ascii")
    if len(packed) > _EMBEDDED_MASTER_LIMIT:
        return None
    return dict(name=master.name, encoding="zlib+base64", data=packed)


def restore_master(qem: Path, destination: Path) -> Path:
    """Restore the original master without replacing an existing file.

    Parameters
    ----------
    qem : Path
        Copy written by :func:`convert` with its embedded source master.
    destination : Path
        New file path, or an existing directory for the original filename.

    Returns
    -------
    Path
        Restored master. External detector links still refer to original files.

    Examples
    --------
    >>> restored = restore_master(Path("scan.qem"), Path("restored_master.h5"))
    """
    from ._streamed_file import read_header

    header, _ = read_header(qem)
    embedded = header["metadata"].get("source_master_file")
    if not embedded:
        raise ValueError(f"{Path(qem).name} carries no embedded master file.")
    destination = Path(destination)
    if destination.is_dir():
        if Path(embedded["name"]).name != embedded["name"]:
            raise ValueError("Embedded master name must be a filename, not a path.")
        destination = destination / embedded["name"]
    if embedded.get("encoding") != "zlib+base64" or len(embedded["data"]) > _EMBEDDED_MASTER_LIMIT:
        raise ValueError("Unsupported or oversized embedded master; retain the original acquisition.")
    decoder = zlib.decompressobj()
    content = decoder.decompress(
        base64.b64decode(embedded["data"], validate=True), _EMBEDDED_MASTER_LIMIT + 1
    )
    if len(content) > _EMBEDDED_MASTER_LIMIT or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid or oversized embedded master; retain the original acquisition.")
    expected = next((record["sha256"] for record in header["metadata"].get("source_files", [])
                     if record["name"] == embedded["name"]), None)
    if expected is None or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Embedded master does not match its source checksum; recopy the .qem file.")
    with open(destination, "xb") as handle:
        handle.write(content)
    return destination


def verify_against_source(qem: Path, master: Path, *, backend: str = "auto") -> dict:
    """Compare every saved value with the detector files read through h5py.

    Every stored value is compared, including flagged detector pixels. Reads
    and comparisons are bounded; the full acquisition stays encoded on GPU.
    """
    import h5py
    import torch

    try:
        import hdf5plugin  # noqa: F401  registers bitshuffle/LZ4 for h5py
    except ImportError as error:
        raise RuntimeError("Verification reads the source with h5py; install hdf5plugin.") from error
    from . import load
    from ._read import _torch_value
    from ._streamed_file import read_header

    header, _ = read_header(qem)
    rows, columns, detector_rows, detector_columns = header["shape"]
    with h5py.File(master, "r") as handle:
        mask_path = "entry/instrument/detector/detectorSpecific/pixel_mask"
        if mask_path in handle:
            valid = np.asarray(handle[mask_path]) == 0
        else:
            valid = np.ones((detector_rows, detector_columns), bool)
    pixels = detector_rows * detector_columns
    with ExitStack() as stack:
        handles = [stack.enter_context(h5py.File(path, "r")) for path in detector_files(master)]
        datasets = [handle["entry/data/data"] for handle in handles]
        starts = np.cumsum([0] + [len(dataset) for dataset in datasets])
        if starts[-1] != rows * columns:
            raise ValueError(f"{master.name} holds {starts[-1]} frames; the copy declares {rows * columns}.")

        def frames(first: int, stop: int) -> np.ndarray:
            parts = []
            for dataset, start in zip(datasets, starts):
                low, high = max(first, start), min(stop, start + len(dataset))
                if low < high:
                    parts.append(dataset[low - start: high - start])
            return np.concatenate(parts)

        differing = compared = 0
        step = max(1, (64 << 20) // (pixels * 16))
        with load(qem, backend=backend, representation="encoded", verbose=False) as saved:
            # Public read() masks flagged pixels for analysis. Verification must
            # inspect the stored values, using the resident's bounded raw decode.
            decode = getattr(saved.data, "_decode_scan_range_torch", None)
            if decode is None:
                decode = saved.data.decode_scan_range_device
            for first in range(0, rows * columns, step):
                stop = min(rows * columns, first + step)
                copy = _torch_value(decode(first, stop))
                raw = frames(first, stop)
                source = torch.from_numpy(raw).to(copy.device).reshape(copy.shape)
                # int32 holds uint16 exactly and maps uint32 one-to-one, so a wide
                # source count can never compare equal by wrapping.
                differing += int((copy.to(torch.int32) != source.to(torch.int32)).sum())
                compared += pixels * (stop - first)
    flagged = int((~valid).sum())
    return dict(
        identical=differing == 0,
        compared_values=compared,
        differing_values=differing,
        flagged_pixels=flagged,
    )


def convert(master: Path, destination: Path, *, write: bool = True, verify: bool = True,
            backend: str = "auto") -> ConvertedAcquisition:
    """Preserve a complete acquisition in a verified, no-overwrite saved copy.

    Parameters
    ----------
    master : Path
        ARINA master with external detector-file links.
    destination : Path
        New ``.qem`` path. Original acquisitions are never modified.
    write : bool, default True
        False performs GPU encoding for a payload estimate without saving.
    verify : bool, default True
        Compare every decoded value with bounded independent source reads
        before publishing. False explicitly skips that integrity gate.
    backend : {"auto", "cuda", "mps"}, default "auto"
        Accelerator for loading, encoding and decoded-value comparison.

    Returns
    -------
    ConvertedAcquisition
        Sizes, verification result and any refusal reason. Larger copies and
        masters beyond the embedding limit are left in the original format.

    Examples
    --------
    >>> result = convert(Path("scan_master.h5"), Path("scan.qem"))
    >>> result.verified
    True
    """
    from . import load, save

    master, destination = Path(master), Path(destination)
    result = ConvertedAcquisition(master, destination)
    files = detector_files(master)
    result.source_bytes = sum(path.stat().st_size for path in [master, *files])
    if not files:
        result.skipped = "no detector files beside the master"
        return result
    if write and destination.exists():
        result.skipped = "destination exists; nothing was replaced"
        return result
    try:
        dtype = _stored_dtype(files)
    except (OSError, KeyError) as error:
        result.skipped = f"unreadable detector file: {error}"
        return result
    if dtype not in _STORED_DTYPES:
        result.skipped = f"stored as {dtype}; this collection command supports integer counts up to 65535"
        return result
    started = time.perf_counter()
    try:
        paths = [master, *files]
        signatures = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
        with load(master, backend=backend, representation="encoded", verbose=False,
                  hot_pixel_correction="none") as acquisition:
            result.qem_bytes = int(acquisition.resident_bytes)
            if result.qem_bytes >= result.source_bytes:
                result.larger = True
                write = False
            if write:
                metadata = dict(acquisition.metadata)
                metadata["source_files"] = _source_files([master, *files])
                metadata["source_metadata"] = master_metadata(master)
                embedded = _embedded_master(master)
                if embedded is None:
                    result.master_embedded = False
                    result.skipped = "master exceeds the metadata embedding limit; keep the original acquisition"
                    return result
                metadata["source_master_file"] = embedded
                overrides, attachment = session_calibration(master)
                if overrides:
                    metadata["calibration_overrides"] = overrides
                    metadata["source_documents"] = [*metadata.get("source_documents", []), attachment]
                    result.session_calibration = overrides
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".qem-convert-", dir=destination.parent) as scratch:
                    candidate = Path(scratch) / destination.name
                    save(candidate, acquisition, metadata=metadata, format="quantem")
                    acquisition.close()
                    result.qem_bytes = candidate.stat().st_size
                    if result.qem_bytes >= result.source_bytes:
                        result.larger = True
                    else:
                        if verify:
                            result.verification = verify_against_source(candidate, master, backend=backend)
                            result.failed = result.verified is False
                        if signatures != [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]:
                            raise ValueError("Source files changed during conversion; retry after acquisition finishes.")
                        if not result.failed:
                            os.link(candidate, destination)
    except (TypeError, ValueError, NotImplementedError, MemoryError) as error:
        result.skipped = f"{type(error).__name__}: {error}"
        result.failed = True
        return result
    result.seconds = time.perf_counter() - started
    return result
