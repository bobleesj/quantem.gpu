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
import re
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
    session_notes: list = field(default_factory=list)

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

    Raises
    ------
    ValueError
        The session file is not readable YAML with the expected sections;
        ``convert`` then converts without it and says why.

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
    try:
        document = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"{sidecar} is not readable YAML: {error}") from error
    evidence = f"{_SESSION_FILE} sha256:{hashlib.sha256(text).hexdigest()}"
    sections = {name: document.get(name) or {} for name in ("microscope", "files", "calibrations", "session")} if isinstance(document, dict) else {}
    if not sections or not all(isinstance(section, dict) for section in sections.values()):
        raise ValueError(f"{sidecar}: microscope, files, calibrations and session must be mappings")
    microscope, files, calibrations = sections["microscope"], sections["files"], sections["calibrations"]
    key, entry, matched_by = session_file_entry(files, Path(master))
    if entry is not None:
        evidence += f", files[{key}] by {matched_by}"
    mag = entry.get("mag") if entry else None
    calibration = calibrations.get(mag) if isinstance(mag, str) else None
    calibration = calibration if isinstance(calibration, dict) else {}
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
    # only the fields read: session and file notes can name people and links that do not belong in a portable copy
    used = {"session": str(sections["session"].get("name")),
            "microscope": {name: microscope.get(name) for name in ("voltage_kV", "semiangle_mrad")},
            "file": {"key": str(key), "master": entry.get("master"), "mag": mag} if entry else None,
            "calibration": {"scan_sampling_A": calibration.get("scan_sampling_A")}}
    content = json.dumps(used, sort_keys=True, default=str)
    attachment = {"filename": "dataset.json", "mediaType": "application/json", "content": content,
                  "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    return overrides, attachment


def session_specimen(master: Path) -> tuple[dict | None, list[dict]]:
    """The specimen an operator declared for this acquisition in its session's ``dataset.yaml``, as the ``.qem``
    ``sample`` group (specification 0.0.3), and each component's CIF as a JSON source document.

    ``specimen:`` (or an older one-crystal ``reference_structure:``) names the components; the file's own ``files``
    entry (``session_file_entry``) adds which components it shows and its recorded thickness estimates, converted from
    ``value_nm`` to ``value`` in angstrom. Only declared fields are written, nothing is defaulted.

    Parameters
    ----------
    master : Path
        ARINA master file or its ``.qem`` copy.

    Returns
    -------
    tuple[dict | None, list[dict]]
        The sample group and the CIF documents, or ``(None, [])`` without a declared specimen.

    Raises
    ------
    ValueError
        The session file is not readable YAML, or its specimen is malformed (``_qem_metadata.validate_sample``).

    Examples
    --------
    >>> session_specimen(Path("no/such/scan_master.h5"))
    (None, [])
    """
    from ._qem_metadata import validate_sample
    sidecar = Path(master).parent / _SESSION_FILE
    if not sidecar.is_file():
        return None, []
    import yaml

    text = sidecar.read_bytes()
    try:
        document = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"{sidecar} is not readable YAML: {error}") from error
    declared = document.get("specimen")
    if declared is None and isinstance(document.get("reference_structure"), dict):
        legacy = document["reference_structure"]
        declared = {"components": {Path(str(legacy.get("cif") or "crystal")).stem: {"cif": legacy.get("cif"), "zone_axis": legacy.get("zone_axis")}}}
    if not isinstance(declared, dict):
        return None, []
    key, entry, matched_by = session_file_entry(document.get("files") or {}, Path(master))
    evidence = f"{_SESSION_FILE} sha256:{hashlib.sha256(text).hexdigest()}" + (f", files[{key}] by {matched_by}" if entry else "")
    sample = {"provenance": _SESSION_FILE, "evidence": evidence}
    for name in ("id", "name", "geometry", "description", "orientation_relationship"):
        if declared.get(name) is not None:
            sample[name] = str(declared[name])
    if declared.get("growth_direction") is not None:
        sample["growth_direction"] = _indices(declared["growth_direction"])
    documents, components = [], {}
    for label, component in (declared.get("components") or {}).items():
        component = component or {}
        written = {name: str(component[name]) for name in ("role", "chemical_formula") if component.get(name)}
        if component.get("zone_axis") is not None:
            written["zone_axis"] = _indices(component["zone_axis"])
        if component.get("cif"):
            cif = (sidecar.parent / str(component["cif"])).resolve()
            if not cif.is_file():
                raise ValueError(f"{sidecar}: component {label} names {component['cif']}, which is not in the session folder")
            content = json.dumps({"cif": cif.read_text(errors="replace")})
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            documents.append({"filename": f"{cif.stem}.cif.json", "mediaType": "application/json", "content": content, "sha256": digest})
            written["cif"] = {"document": f"{cif.stem}.cif.json", "sha256": digest}
        estimates = [_thickness_estimate(e) for e in ((entry or {}).get("thickness") or {}).get(label, [])]
        if estimates:
            written["thickness_estimates"] = estimates
        components[str(label)] = written
    if components:
        sample["components"] = components
    if entry and entry.get("components_in_view"):
        sample["components_in_view"] = [str(label) for label in entry["components_in_view"]]
    validate_sample(sample)
    return sample, documents


def _indices(value) -> list[int]:
    """A direction [u, v, w] from a list or the older text form "[1-10]"."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in re.findall(r"-?\d", str(value))]


def _thickness_estimate(estimate: dict) -> dict:
    """A declared estimate in the ``.qem`` form: ``value_nm`` becomes ``value`` in angstrom, the rest as typed."""
    out = {"method": estimate.get("method"), "value": float(estimate.get("value_nm", 0)) * 10, "unit": "angstrom"}
    if estimate.get("uncertainty_nm") is not None:
        out["uncertainty"] = float(estimate["uncertainty_nm"]) * 10
    if estimate.get("range_nm") is not None:
        out["range"] = [float(v) * 10 for v in estimate["range_nm"]]
    for name in ("region", "reference", "date"):
        if estimate.get(name) is not None:
            out[name] = estimate[name] if name == "region" else str(estimate[name])
    if estimate.get("preferred") is not None:
        out["preferred"] = bool(estimate["preferred"])
    return out


def session_file_entry(files: dict, path: Path) -> tuple[object, dict | None, str]:
    """The ``files`` entry of a session ``dataset.yaml`` that describes one scan.

    By name: the entry whose ``master`` is this file, or the master a ``.qem``
    copy was converted from. By scan number: only when no entry of the session
    names its master (some sessions list scans by number alone) and exactly one
    scan in the folder ends in that number. A session can hold several series with
    the same numbers, and matching one of them by number would lend it another
    scan's calibration.

    Parameters
    ----------
    files : dict
        The session file's ``files`` mapping.
    path : Path
        A ``*_master.h5`` or ``.qem`` scan.

    Returns
    -------
    tuple
        ``(key, entry, "name" | "scan number")``, or ``(None, None, "")``.

    Examples
    --------
    >>> session_file_entry({3: {"master": "a_3_master.h5"}}, Path("a_3_master.h5"))
    (3, {'master': 'a_3_master.h5'}, 'name')
    """
    stem = _scan_stem(path)
    entries = [(key, entry) for key, entry in files.items() if isinstance(entry, dict)]
    for key, entry in entries:
        if entry.get("master") in (path.name, f"{stem}{_MASTER_SUFFIX}"):
            return key, entry, "name"
    number = _trailing_number(stem)
    if number is None or any("master" in entry for _, entry in entries):
        return None, None, ""
    stems = {_scan_stem(other) for pattern in (f"*{_MASTER_SUFFIX}", "*.qem") for other in path.parent.glob(pattern)}
    if sum(_trailing_number(other) == number for other in stems) != 1:
        return None, None, ""
    for key, entry in entries:
        if _trailing_number(str(key)) == number:
            return key, entry, "scan number"
    return None, None, ""


def _scan_stem(path: Path) -> str:
    """scan_16_master.h5 and scan_16.qem both name the scan scan_16."""
    return path.name[: -len(_MASTER_SUFFIX)] if path.name.endswith(_MASTER_SUFFIX) else path.stem


def _trailing_number(text: str) -> int | None:
    """The number a name ends with (sample_54___00 gives 0), or None."""
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else None


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
                try:
                    overrides, attachment = session_calibration(master)
                except ValueError as error:          # an optional sidecar never blocks a lossless copy
                    overrides, attachment = {}, None
                    result.session_notes.append(f"{_SESSION_FILE} ignored: {error}")
                if overrides:
                    metadata["calibration_overrides"] = overrides
                    metadata["source_documents"] = [*metadata.get("source_documents", []), attachment]
                    result.session_calibration = overrides
                try:
                    sample, cif_documents = session_specimen(master)
                except ValueError as error:          # a malformed specimen never blocks a lossless copy
                    sample, cif_documents = None, []
                    result.session_notes.append(f"{_SESSION_FILE} specimen ignored: {error}")
                if sample is not None:
                    metadata["sample"] = sample
                    metadata["source_documents"] = [*metadata.get("source_documents", []), *cif_documents]
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
