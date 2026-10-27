"""Convert Arina HDF5 acquisitions into ``.qem`` copies with optional verification.

The conversion is ``io.load(master, representation="encoded")`` followed by
``io.save(destination, acquisition, format="quantem")``. This module adds what a
collection needs around that pair: finding acquisitions, recording which source
files a copy came from, carrying the session's calibrated beam from its
``dataset.yaml``, and comparing the saved copy with the source files.
Source files are opened read-only and are never modified or removed.
"""

import base64
import hashlib
import json
import os
import re
import tempfile
import time
import zlib
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import torch
import yaml

from quantem.gpu.formats import dataset_yaml
from quantem.gpu.formats.float8 import hdf5_dtype, is_float8
from quantem.gpu.formats.hdf5.frames import detector_sources
from quantem.gpu.formats.hdf5.master import read_pixel_mask
from quantem.gpu.formats.publish import publish_file
from quantem.gpu.formats.qem.metadata import (
    acquisition_metadata,
    effective_metadata,
    validate_sample,
)
from quantem.gpu.formats.qem.snapshot import BIT_DTYPES, read_header
from quantem.gpu.io import load, save
from quantem.gpu.io.read import torch_value
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.hot_pixels import (
    fill_flagged,
    median_plan,
    normalize_hot_pixel_correction,
)
from quantem.gpu.resident.mps.counts import MPSStreamedCounts

_MASTER_SUFFIX = "_master.h5"
_STORED_DTYPES = BIT_DTYPES
_HASH_BLOCK = 8 << 20
_READABLE_VALUES = 64
_EMBEDDED_MASTER_LIMIT = 8 << 20
_MAX_CIF_DOCUMENTS = 8                 # of the 16 source documents a .qem carries; the rest stay for the source's own metadata
_MAX_CIF_BYTES = 2 << 20               # of the 4 MiB attachment budget


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
    hot_pixel_correction: dict = field(default_factory=dict)

    @property
    def verified(self) -> bool | None:
        """True or False after verification; None when it did not run."""
        return self.verification.get("identical") if self.verification else None


def _session_path(master: Path) -> Path:
    candidates = [master.parent / name for name in ("dataset.yaml", "dataset.yml")]
    present = [path for path in candidates if path.is_file()]
    if len(present) > 1:
        raise ValueError("Both dataset.yaml and dataset.yml exist; keep one session file before converting.")
    return present[0] if present else candidates[0]


def _sampling_pair(value, name: str) -> tuple[float, float]:
    values = np.atleast_1d(value).astype(float)
    if values.ndim != 1 or values.size not in (1, 2) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(f"{name} needs one or two positive finite values, in (row, column) order; got {value!r}.")
    return tuple(np.repeat(values, 2) if values.size == 1 else values)


def _explicit_calibration(voltage_kV, semiangle_mrad, scan_sampling_A, detector_sampling_mrad) -> dict:
    overrides = {}
    for field, value, path, factor, unit in (
        ("voltage_kV", voltage_kV, "electron_source/accelerating_voltage", 1000, "V"),
        ("semiangle_mrad", semiangle_mrad, "illumination_system/semi_convergence_angle", 1, "mrad"),
        ("scan_sampling_A", scan_sampling_A, "scan_controller/regular_scan/pixel_size", 1e-10, "m"),
        ("detector_sampling_mrad", detector_sampling_mrad, "imaging_system/reciprocal_pixel_size", 1, "mrad"),
    ):
        if value is None:
            continue
        if "sampling" in field:
            values = _sampling_pair(value, field)
            quantities = [(f"{path}_{axis}", number) for axis, number in zip(("row", "column"), values)]
        else:
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be positive and finite; got {value!r}.")
            quantities = [(path, value)]
        for key, number in quantities:
            overrides[key] = dict(value=float(number) * factor, unit=unit, provenance="user_override",
                                  evidence=f"Explicit conversion parameter {field}={value!r}")
    # Apply the same scientific validation used by the writer, before GPU work.
    acquisition_metadata((1, 1, 1, 1), {"calibration_overrides": overrides})
    return overrides


def session_calibration(master: Path) -> tuple[dict, dict | None]:
    """Calibration an operator recorded for this acquisition in its session's ``dataset.yaml``.

    An Arina master records neither the probe semi-angle nor the scan step; the
    session file next to it does, once per session (``microscope``) and as a
    ``[row, col]`` step through the file's own ``files`` entry
    (``formats.dataset_yaml.scan_sampling_A``). The values become calibration
    overrides whose evidence is the session file's SHA-256, so every reader of
    the copy applies them ahead of the recorded values, and the fields used are
    attached as a JSON source document. Only an entry whose ``master`` is this
    file counts: a session can hold several series with the same numbers.

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
        The session file is not readable YAML of schema ``quantem.dataset/2`` with
        the expected sections; ``convert`` then converts without it and says why.

    Examples
    --------
    >>> session_calibration(Path("no/such/scan_master.h5"))
    ({}, None)
    """
    sidecar = _session_path(Path(master))
    if not sidecar.is_file():
        return {}, None
    document, evidence = _session_document(sidecar)
    sections = {name: document.get(name) or {} for name in ("microscope", "files", "calibrations", "session")}
    if not all(isinstance(section, dict) for section in sections.values()):
        raise ValueError(f"{sidecar}: microscope, files, calibrations and session must be mappings")
    microscope, files = sections["microscope"], sections["files"]
    key, entry, matched_by = session_file_entry(files, Path(master))
    if entry is not None:
        evidence += f", files[{key}] by {matched_by}"
    # a scan with no entry of its own takes only the session-wide step (calibrations.default), never another entry's
    step = dataset_yaml.scan_sampling_A(document, entry, sidecar)
    overrides = {}

    def override(path, value, factor, unit):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            overrides[path] = {"value": float(value) * factor, "unit": unit, "provenance": "user_override", "evidence": evidence}

    override("electron_source/accelerating_voltage", microscope.get("voltage_kV"), 1000, "V")
    override("illumination_system/semi_convergence_angle", microscope.get("semiangle_mrad"), 1, "mrad")
    if step is not None:
        # the stored path names the column axis "column" (scientific_metadata strings are unchanged)
        for axis, value in zip(("row", "column"), step, strict=True):
            override(f"scan_controller/regular_scan/pixel_size_{axis}", value, 1e-10, "m")
    detector_step = (entry or {}).get("detector_sampling_mrad", microscope.get("detector_sampling_mrad"))
    if detector_step is not None:
        detector_step = _sampling_pair(detector_step, "detector_sampling_mrad")
        for axis, value in zip(("row", "column"), detector_step):
            override(f"imaging_system/reciprocal_pixel_size_{axis}", value, 1, "mrad")
    if not overrides:
        return {}, None
    # only the fields read: session and file notes can name people and links that do not belong in a portable copy
    used = {"session": str(sections["session"].get("name")),
            "microscope": {name: microscope.get(name) for name in ("voltage_kV", "semiangle_mrad")},
            "file": {"key": str(key), "master": entry.get("master"), "mag": entry.get("mag")} if entry else None,
            "calibration": {"scan_sampling_A": None if step is None else list(step),
                            "detector_sampling_mrad": detector_step}}
    content = json.dumps(used, sort_keys=True, default=str)
    attachment = {"filename": "dataset.json", "mediaType": "application/json", "content": content,
                  "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    return overrides, attachment


def session_specimen(master: Path) -> tuple[dict | None, list[dict]]:
    """The specimen an operator declared for this acquisition in its session's ``dataset.yaml``, as the ``.qem``
    ``sample`` group (specification 0.0.3), and each component's CIF as a JSON source document.

    ``specimen:`` names the components; the file's own ``files``
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
        The session file is not readable YAML of schema ``quantem.dataset/2``, or its specimen is malformed
        (``formats.qem.metadata.validate_sample``).

    Examples
    --------
    >>> session_specimen(Path("no/such/scan_master.h5"))
    (None, [])
    """
    sidecar = _session_path(Path(master))
    if not sidecar.is_file():
        return None, []
    document, evidence = _session_document(sidecar)
    declared = document.get("specimen")
    if declared is None:
        return None, []
    files = document.get("files") or {}
    if not isinstance(declared, dict) or not isinstance(declared.get("components") or {}, dict) or not isinstance(files, dict):
        raise ValueError(f"{sidecar}: specimen and its components, and files, must be mappings")
    key, entry, matched_by = session_file_entry(files, Path(master))
    evidence += f", files[{key}] by {matched_by}" if entry else ""
    sample: dict = {"provenance": sidecar.name, "evidence": evidence}
    for name in ("id", "name", "geometry", "description", "orientation_relationship"):
        if declared.get(name) is not None:
            sample[name] = str(declared[name])
    if declared.get("growth_direction") is not None:
        sample["growth_direction"] = _indices(declared["growth_direction"], f"{sidecar}: growth_direction")
    thickness = (entry or {}).get("thickness") or {}
    if not isinstance(thickness, dict) or not all(
        isinstance(estimates, list) and all(isinstance(estimate, dict) for estimate in estimates)
        for estimates in thickness.values()
    ):
        raise ValueError(f"{sidecar}: files[{key}].thickness maps each component to a list of estimates")
    documents: dict[Path, dict] = {}
    components = {}
    for label, declared_component in (declared.get("components") or {}).items():
        component = {} if declared_component is None else declared_component
        if not isinstance(component, dict):
            raise ValueError(f"{sidecar}: component {label} must be a mapping (role, chemical_formula, cif, zone_axis)")
        written = {name: str(component[name]) for name in ("role", "chemical_formula") if component.get(name)}
        if component.get("zone_axis") is not None:
            written["zone_axis"] = _indices(component["zone_axis"], f"{sidecar}: component {label} zone_axis")
        if component.get("cif"):
            written["cif"] = _cif_document(sidecar, str(label), str(component["cif"]), documents)
        estimates = [_thickness_estimate(estimate, f"{sidecar}: files[{key}].thickness.{label}") for estimate in thickness.get(label, [])]
        if estimates:
            written["thickness_estimates"] = estimates
        components[str(label)] = written
    if components:
        sample["components"] = components
    in_view = (entry or {}).get("components_in_view")
    if in_view:
        if not isinstance(in_view, list):
            raise ValueError(f"{sidecar}: files[{key}].components_in_view is a list of component labels")
        sample["components_in_view"] = [str(label) for label in in_view]
    attached = list(documents.values())
    if len(attached) > _MAX_CIF_DOCUMENTS or sum(len(attachment["content"].encode("utf-8")) for attachment in attached) > _MAX_CIF_BYTES:
        raise ValueError(f"{sidecar}: more CIF text than a .qem carries ({_MAX_CIF_DOCUMENTS} files, {_MAX_CIF_BYTES >> 20} MiB)")
    try:
        validate_sample(sample)
    except (ValueError, TypeError, KeyError) as error:  # named after the file to fix; a crafted value is still a malformed specimen
        raise ValueError(f"{sidecar}: {error}") from error
    return sample, attached


def _session_document(sidecar: Path) -> tuple[dict, str]:
    """Parse a session file of schema ``quantem.dataset/2``, with the evidence string that names its exact bytes.

    Calibration and specimen both cite the file by SHA-256, so a copy records
    which version of ``dataset.yaml`` its values came from.
    """
    text = sidecar.read_bytes()
    try:
        document = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"{sidecar} is not readable YAML: {error}") from error
    return dataset_yaml.checked_dataset(document, sidecar), f"{sidecar.name} sha256:{hashlib.sha256(text).hexdigest()}"


def _cif_document(sidecar: Path, label: str, name: str, documents: dict[Path, dict]) -> dict:
    """The CIF a component names, as a JSON source document (one per file, named uniquely): a file inside the session
    folder only, so a .qem never carries an unrelated file; returns the component's reference to it."""
    folder = sidecar.parent.resolve()
    cif = (folder / name).resolve()
    if not cif.is_relative_to(folder) or not cif.is_file():
        raise ValueError(f"{sidecar}: component {label} names {name}, which is not a file in the session folder")
    if cif not in documents:
        taken = {document["filename"] for document in documents.values()}
        filename = f"{cif.stem}.cif.json" if f"{cif.stem}.cif.json" not in taken else f"{cif.stem}_{len(documents)}.cif.json"
        content = json.dumps({"cif": cif.read_text(errors="replace")})
        documents[cif] = {"filename": filename, "mediaType": "application/json", "content": content,
                          "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    return {"document": documents[cif]["filename"], "sha256": documents[cif]["sha256"]}


def _indices(value, where: str) -> list[int]:
    """A direction [u, v, w] (or hexagonal [u, v, t, w]): a list of integers, or text "[1-10]" (single-digit indices, a
    minus applying to the next digit) or "[1 -1 0]" (separated). Anything else is refused, not rounded."""
    if isinstance(value, (list, tuple)):
        if not all(type(index) is int for index in value):
            raise ValueError(f"{where}: {value!r} must be integers")
        indices = list(value)
    elif isinstance(value, str):
        text = value.strip().strip("[]() ")
        parts = re.split(r"[,\s]+", text) if re.search(r"[,\s]", text) else re.findall(r"-?\d", text)
        if "".join(parts).replace("-", "") != re.sub(r"[^0-9]", "", text) or not all(re.fullmatch(r"-?\d+", part) for part in parts):
            raise ValueError(f"{where}: {value!r} is not a direction")
        indices = [int(part) for part in parts]
    else:
        raise ValueError(f"{where}: {value!r} is not a direction")
    if len(indices) not in (3, 4) or not any(indices):
        raise ValueError(f"{where}: {value!r} is not [u, v, w] or [u, v, t, w]")
    return indices


def _thickness_estimate(estimate: dict, where: str) -> dict:
    """A declared estimate in the ``.qem`` form: ``value_nm`` becomes ``value`` in angstrom, the rest as typed; a value
    of the wrong type is refused, not coerced."""
    def number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: {name} {value!r} must be a number")
        return float(value)

    out = {"method": estimate.get("method"), "value": number(estimate.get("value_nm"), "value_nm") * 10, "unit": "angstrom"}
    if estimate.get("uncertainty_nm") is not None:
        out["uncertainty"] = number(estimate["uncertainty_nm"], "uncertainty_nm") * 10
    if estimate.get("range_nm") is not None:
        bounds = estimate["range_nm"]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"{where}: range_nm {bounds!r} is [low, high]")
        out["range"] = [number(bound, "range_nm") * 10 for bound in bounds]
    region = estimate.get("region")
    if region is not None:
        out["region"] = region
    for name in ("reference", "date"):
        if estimate.get(name) is not None:
            out[name] = str(estimate[name])
    if estimate.get("preferred") is not None:
        if type(estimate["preferred"]) is not bool:
            raise ValueError(f"{where}: preferred {estimate['preferred']!r} is true or false")
        out["preferred"] = estimate["preferred"]
    return out


def session_file_entry(files: dict, path: Path) -> tuple[object, dict | None, str]:
    """The ``files`` entry of a session ``dataset.yaml`` that describes one scan.

    By name: the entry whose ``master`` is this file, or the master a ``.qem``
    copy was converted from. By scan number: an entry that names no master (some
    sessions list scans by number alone), and only when exactly one scan in the
    folder ends in that number. A session can hold several series with the same
    numbers, and matching one of them by number would lend it another scan's
    calibration. Entries that name their master and entries listed by number can
    share a session (quantem.live records a thickness under a named entry).

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
    if number is None:
        return None, None, ""
    stems = {_scan_stem(other) for pattern in (f"*{_MASTER_SUFFIX}", "*.qem") for other in path.parent.glob(pattern)}
    if sum(_trailing_number(other) == number for other in stems) != 1:
        return None, None, ""
    for key, entry in entries:
        if "master" not in entry and _trailing_number(str(key)) == number:
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
    with h5py.File(master, "r") as handle:
        sources = detector_sources(handle)
    if any(not source.external or source.dataset_path != "/entry/data/data" for source in sources):
        raise ValueError("Collection conversion requires external detector chunks at /entry/data/data.")
    files = [Path(source.path).resolve() for source in sources]
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


def _source_files(paths: list[Path], *, stop: Event | None = None) -> list[dict]:
    """Record complete ordered source fingerprints, or cancel abandoned work."""
    def fingerprint(path):
        digest = hashlib.sha256()
        buffer = bytearray(_HASH_BLOCK)
        view = memoryview(buffer)
        with open(path, "rb", buffering=0) as handle:
            while True:
                if stop is not None and stop.is_set():
                    raise CancelledError("Source hashing was cancelled before saving.")
                count = handle.readinto(buffer)
                if not count:
                    break
                digest.update(view[:count])
        return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    # Independent shards preserve their ordered provenance while reusing at
    # most sixteen 8 MiB read buffers, without allocating every read block.
    with ThreadPoolExecutor(max_workers=min(16, len(paths))) as workers:
        return list(workers.map(fingerprint, paths))


def _json_value(value):
    """Convert one HDF5 field or attribute to JSON; non-finite floats become text."""
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
    content = master.read_bytes()
    packed = base64.b64encode(zlib.compress(content, 9)).decode("ascii")
    if len(packed) > _EMBEDDED_MASTER_LIMIT:
        return None
    return {"name": master.name, "encoding": "zlib+base64", "data": packed,
            "sha256": hashlib.sha256(content).hexdigest()}


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
    expected = embedded.get("sha256") or next(
        (record.get("sha256") for record in header["metadata"].get("source_files", [])
         if record["name"] == embedded["name"]), None
    )
    if expected is None or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Embedded master does not match its source checksum; recopy the .qem file.")
    with open(destination, "xb") as handle:
        handle.write(content)
    return destination


def verify_against_source(qem: Path, master: Path, *, backend: str = "auto") -> dict:
    """Compare every original measurement bit, including flagged detector pixels.

    Counts stored in a narrower unsigned type than the source are compared by
    value. Without an explicit correction, no marker substitution is accepted as raw
    preservation. Declared mask corrections are checked against bounded source
    neighborhoods; all other values retain their original bits. Bit comparisons
    run on the requested GPU in bounded windows, including for float64 on
    Metal, whose tensor arithmetic does not support that floating dtype.
    Uncorrected supported integer HDF5 uses bounded GPU decoding and exact
    unsigned comparison; other formats and correction references use h5py
    source decoding followed by accelerator comparison.
    """
    import hdf5plugin  # noqa: F401  registers bitshuffle/LZ4 for h5py; loaded on use only
    header, _ = read_header(qem)
    rows, columns, detector_rows, detector_columns = header["shape"]
    mask = read_pixel_mask(master)
    valid = np.ones((detector_rows, detector_columns), bool) if mask is None else mask == 0
    correction = header.get("metadata", {}).get("hot_pixel_correction", {})
    method = correction.get("method", "none") if correction.get("applied") else "none"
    method = normalize_hot_pixel_correction(method)
    pixels = detector_rows * detector_columns
    plan = median_plan(~valid)
    with ExitStack() as stack:
        handles = [stack.enter_context(h5py.File(path, "r")) for path in detector_files(master)]
        datasets = [handle["entry/data/data"] for handle in handles]
        if any(dataset.shape[1:] != (detector_rows, detector_columns) for dataset in datasets):
            raise ValueError("Source detector shape differs from the QEM copy; verify the matching acquisition.")
        starts = np.cumsum([0] + [len(dataset) for dataset in datasets])
        if starts[-1] != rows * columns:
            raise ValueError(f"{master.name} holds {starts[-1]} frames; the copy declares {rows * columns}.")

        def frames(first: int, stop: int) -> np.ndarray:
            parts = []
            for dataset, start in zip(datasets, starts):
                low, high = max(first, start), min(stop, start + len(dataset))
                if low < high:
                    parts.append(dataset[low - start: high - start].view(hdf5_dtype(dataset)))
            return np.concatenate(parts)

        differing = compared = 0
        step = min(512, max(1, (32 << 20) // (pixels * max(2, np.dtype(header["dtype"]).itemsize))))
        with load(qem, backend=backend, representation="encoded", verbose=False) as saved:
            # Compare stored values, including flagged pixels, using bounded
            # raw decode. Detector-product masking is a separate operation.
            if isinstance(saved.data, MPSStreamedCounts):
                decode = saved.data.decode_scan_range_torch
            else:
                decode = saved.data.decode_scan_range_device
            accelerated = False
            if isinstance(saved.data, MPSStreamedCounts) and method == "none":
                from quantem.gpu.io.hdf5.mps.verify import (
                    supports_counts,
                    verify_counts,
                )

                if supports_counts(datasets, np.dtype(header["dtype"])):
                    differing = verify_counts(master, saved.data, datasets[0].dtype)
                    compared = rows * columns * pixels
                    accelerated = True
            if isinstance(saved.data, StreamedCounts) and method == "none":
                from quantem.gpu.io.hdf5.cuda.verify import (
                    supports_counts,
                    verify_counts,
                )

                if supports_counts(datasets, np.dtype(header["dtype"])):
                    differing = verify_counts(master, saved.data, datasets[0].dtype)
                    compared = rows * columns * pixels
                    accelerated = True
            if not accelerated:
                for first in range(0, rows * columns, step):
                    stop = min(rows * columns, first + step)
                    decoded = decode(first, stop)
                    raw = np.ascontiguousarray(frames(first, stop))
                    if method != "none":
                        # the CPU reference of the GPU correction, bit for bit
                        raw = fill_flagged(raw.copy(), plan, method)
                    stored = np.dtype(header["dtype"])
                    if stored != raw.dtype:
                        # Counts stored in a narrower unsigned type: compare every value exactly.
                        if not (stored.kind == raw.dtype.kind == "u" and stored.itemsize < raw.dtype.itemsize):
                            raise ValueError(f"The QEM stores {stored.name} values of a {raw.dtype.name} source; "
                                             "only exact narrower unsigned counts are supported.")
                        copy = torch_value(decoded).to(torch.int32).to(torch.int64).reshape(stop - first, -1)
                        source = torch.from_numpy(raw.astype(np.int64).reshape(stop - first, -1)).to(copy.device)
                        differing += int((copy != source).sum())
                        compared += pixels * (stop - first)
                        del decoded, copy, source, raw
                        continue
                    if hasattr(decoded, "_mtl"):
                        decoded.dtype = np.dtype("uint8")
                        decoded.shape = (raw.nbytes,)
                        copy = torch_value(decoded)
                    elif isinstance(decoded, torch.Tensor):
                        copy = decoded.view(torch.uint8).reshape(-1)
                    elif is_float8(header["dtype"]):
                        copy = torch_value(decoded).view(torch.uint8).reshape(-1)
                    else:
                        copy = torch_value(decoded.view(np.uint8)).reshape(-1)
                    source = torch.from_numpy(raw.view(np.uint8).reshape(-1)).to(copy.device)
                    unequal = (copy != source).reshape(-1, raw.dtype.itemsize).any(dim=1)
                    differing += int(unequal.sum())
                    compared += pixels * (stop - first)
                    del decoded, copy, source, raw, unequal
    flagged = int((~valid).sum())
    return dict(
        identical=differing == 0,
        compared_values=compared,
        differing_values=differing,
        flagged_pixels=flagged,
        flagged_markers_stored_as_zero=0,
        hot_pixel_correction=method,
        corrected_values_compared=flagged * rows * columns if method != "none" else 0,
    )


def convert(master: Path, destination: Path, *, write: bool = True, verify: bool = False,
            source_hashes: bool = False,
            backend: str = "auto", voltage_kV: float | None = None,
            semiangle_mrad: float | None = None,
            scan_sampling_A: float | tuple[float, float] | None = None,
            detector_sampling_mrad: float | tuple[float, float] | None = None,
            hot_pixel_correction: str = "none") -> ConvertedAcquisition:
    """Preserve a complete acquisition in a no-overwrite saved copy.

    Parameters
    ----------
    master : Path
        ARINA master with external detector-file links.
    destination : Path
        New ``.qem`` path. Original acquisitions are never modified.
    write : bool, default True
        False performs GPU encoding for a payload estimate without saving.
    verify : bool, default False
        Compare every decoded value with bounded independent source reads
        before publishing when True. QEM checksums and file synchronization
        remain enabled regardless of this option.
    source_hashes : bool, default False
        Read every original file to record its SHA-256 when True. Otherwise
        record source names and sizes without this extra read pass. The small
        embedded master retains its own checksum in both modes.
    backend : {"auto", "cuda", "mps"}, default "auto"
        Accelerator for loading, encoding and decoded-value comparison.
    voltage_kV, semiangle_mrad
        Explicit positive microscope calibration, overriding the session file
        and source metadata. Omit to retain available calibration.
    scan_sampling_A, detector_sampling_mrad
        Positive scan spacing in angstrom and detector spacing in mrad per
        pixel. One value applies to both axes; a pair is (row, column).
    hot_pixel_correction : {"none", "median", "zero"}, default "none"
        Optional correction for uint8/uint16/uint32 acquisitions only. The
        copy is an archive of the acquisition (``restore_master`` rebuilds the
        master), so it keeps every stored value unless a correction is
        requested. Integer counts are stored in the narrowest unsigned type
        that holds every value, and the copy records the source dtype; ``io.load``'s median default applies when data are loaded
        for analysis. Corrects stored-mask pixels before encoding. Median uses the valid
        pixels of the 3x3 window (or the smallest larger window that holds
        one), rounded half to even. Verification
        compares against the declared correction, not against unchanged raw data.

    Returns
    -------
    ConvertedAcquisition
        Sizes, verification result and any refusal reason. Larger copies and
        masters beyond the embedding limit are left in the original format.

    Examples
    --------
    >>> result = convert(Path("scan_master.h5"), Path("scan.qem"))
    >>> result.verified is None
    True
    >>> checked = convert(Path("scan_master.h5"), Path("checked.qem"),
    ...                   verify=True, source_hashes=True)
    >>> checked.verified
    True
    """
    import hdf5plugin  # noqa: F401  registers bitshuffle/LZ4 for h5py; loaded on use only
    master, destination = Path(master), Path(destination)
    explicit = _explicit_calibration(voltage_kV, semiangle_mrad, scan_sampling_A, detector_sampling_mrad)
    hot_pixel_correction = normalize_hot_pixel_correction(hot_pixel_correction)
    _session_path(master)  # Ambiguous sidecars must not silently discard calibration.
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
        with h5py.File(files[0], "r") as handle:
            dtype = hdf5_dtype(handle["entry/data/data"]).name
    except (OSError, KeyError) as error:
        result.skipped = f"unreadable detector file: {error}"
        return result
    if dtype not in _STORED_DTYPES:
        result.skipped = f"stored as {dtype}; use a supported real numeric acquisition dtype"
        return result
    if hot_pixel_correction != "none" and dtype not in {"uint8", "uint16", "uint32"}:
        raise ValueError(
            f"Hot-pixel correction requires uint8/uint16/uint32 counts; got {dtype}. "
            "Use hot_pixel_correction='none' to preserve the original values."
        )
    started = time.perf_counter()
    try:
        paths = [master, *files]
        signatures = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
        # Integer counts are stored in the narrowest unsigned type that holds every value (after the
        # requested correction); the copy records the source dtype, and verification compares values.
        with ExitStack() as stack:
            # Hash unchanged source files while the accelerator prepares the
            # resident. The executor drains before this scope exits, including
            # loading failures and copies rejected by the size comparison.
            fingerprints = None
            stop = Event()
            if write and source_hashes:
                workers = stack.enter_context(ThreadPoolExecutor(max_workers=1))
                stack.callback(stop.set)
                fingerprints = workers.submit(_source_files, paths, stop=stop)
            acquisition = stack.enter_context(load(
                master, backend=backend, representation="encoded", verbose=False,
                hot_pixel_correction=hot_pixel_correction,
            ))
            result.hot_pixel_correction = dict(acquisition.metadata.get("hot_pixel_correction", {}))
            result.qem_bytes = int(acquisition.resident_bytes)
            if result.qem_bytes >= result.source_bytes:
                result.larger = True
                write = False
                stop.set()
            if write:
                metadata = dict(acquisition.metadata)
                metadata["source_files"] = (
                    fingerprints.result() if fingerprints is not None else
                    [{"name": path.name, "bytes": signature[0]}
                     for path, signature in zip(paths, signatures, strict=True)]
                )
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
                    result.session_notes.append(f"{_session_path(master).name} ignored: {error}")
                if overrides:
                    metadata["calibration_overrides"] = overrides
                    metadata["source_documents"] = [*metadata.get("source_documents", []), attachment]
                    result.session_calibration = overrides
                if explicit:
                    for key in explicit.keys() & overrides.keys():
                        result.session_notes.append(f"Explicit {key} overrides the session value.")
                    metadata["calibration_overrides"] = {**overrides, **explicit}
                    result.session_calibration = metadata["calibration_overrides"]
                scientific = acquisition_metadata(acquisition.shape, metadata)
                result.session_calibration = scientific["calibration_overrides"]
                effective = effective_metadata(metadata, scientific)
                missing = [key for key in ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad")
                           if key not in effective]
                if missing:
                    result.session_notes.append(f"Calibration unknown: {', '.join(missing)}; values were not guessed.")
                try:
                    sample, cif_documents = session_specimen(master)
                except ValueError as error:          # a malformed specimen never blocks a lossless copy
                    sample, cif_documents = None, []
                    result.session_notes.append(f"{_session_path(master).name} specimen ignored: {error}")
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
                            publish_file(candidate, destination)
    except (TypeError, ValueError, NotImplementedError, MemoryError) as error:
        result.skipped = f"{type(error).__name__}: {error}"
        result.failed = True
        return result
    result.seconds = time.perf_counter() - started
    return result
