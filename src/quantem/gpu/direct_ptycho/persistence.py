"""Saved-result storage for the public SSB workflow."""

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import fields
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.inspect import inspect as inspect_source
from quantem.gpu.direct_ptycho.branches import RotationBranches
from quantem.gpu.direct_ptycho.results import SSBResult, host_array, retired_estimator_note

# One number versions everything a saved SSB fit means: the layout of its record and the convention its values follow.
# Bump it when fitted values acquire a different physical meaning or record layout: the automatic
# detector sampling rule, the unit, sign or angle reference of an aberration, the rotation convention,
# the bright-field selection rule, the loss definition, or the record layout. Every older fit is then fitted again.
# A reconstruction-only policy belongs in the result signature instead. This invalidates exact image reuse
# without declaring unchanged fitted coefficients unreadable or rescheduling a historical acquisition.
# 7: the record follows the shared vocabulary. The rotation is one physical angle, scan_detector_rotation_deg in
#    [0, 360), with scan_detector_rotation_flipped; it was an angle below 180 with com_reversed. The saved result's own "schema"
#    is a string like the fit block's.
# 6: schema 5 covered two automatic detector samplings, 2 x semiangle / half-maximum radius and then
#    semiangle / disk-edge radius; C10 and C12 fitted under the first reconstruct a different image under the second.
# 5: replayable, optimizer-neutral trial records and explicit correction overrides.
SCHEMA = 7
# What a stored fit must carry under "schema" to be read at all (Live's computed.ssb in config.json). Named like the
# .qem scientific-metadata schema string, so a reader refuses another version instead of reinterpreting its values.
FIT_SCHEMA = f"quantem.ssb-fit/{SCHEMA}"
# The top-level "schema" of the screening record that holds the fit block (live/screen/<dataset>/config.json). The
# record belongs to quantem.live: this is the schema of the record its reader accepts, and it takes no other. It
# is written here only when SSB.reconstruct_series creates the record, and required of a record that exists.
SCREEN_ACQUISITION_SCHEMA = "quantem.screen-acquisition/2"
# The same for a saved result record (ssb-find_aberrations.json, ssb-reconstruct.json) and its signature. It follows
# SCHEMA and adds the result record's own layout changes:
# 10: the record holds no absolute path, inode, device or time: the source is its portable acquisition_identity, and
#     input names files relative to the results folder (portable_source), so a moved or copied folder is reused.
# 9: the rotation check is rotation_branches, both 180-degree branches fitted independently (their aberrations, loss,
#    column test, phase tail and native phases), replacing column_sign, initial_column_sign,
#    initial_scan_detector_rotation_deg and rotation_check_phases.
# 8: the result field refine_method is refinement, the name of the find_aberrations keyword it records.
RESULT_VERSION = 10
RESULT_SCHEMA = f"quantem.ssb-result/{RESULT_VERSION}"


# Not saved in "result": arrays and branches have their own entries, and the session's paths (saved_path, source_path)
# are its own, not the result's; a reused result takes them from the session that reads it.
_ARTIFACT_FIELDS = {"object_wave", "rotation_branches", "reused", "saved_path", "source_path", "metadata",
                    "parameter_slices", "calibration_context", "_calibration_source"}


def save_result(
    result: SSBResult,
    *,
    paths: tuple[Path, Path],
    signature: dict[str, object],
    input_metadata: dict[str, object],
) -> SSBResult:
    """Persist one result and attach its complete provenance."""

    array_path, metadata_path = paths
    result_metadata = {
        descriptor.name: getattr(result, descriptor.name)
        for descriptor in fields(result)
        if descriptor.name not in _ARTIFACT_FIELDS
    }
    metadata = {
        "schema": RESULT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_provenance(),
        "input": json_value(input_metadata),
        "signature": signature,
        "result": json_value(result_metadata),
    }
    if result.calibration_context is not None:
        # Optional metadata remains readable by older result readers. File identity was already computed.
        context = json_value(result.calibration_context)
        context["session_identity"]["source"] = signature["source"]
        metadata["calibration_context"] = context
    # Optional diagnostics leave the scientific result payload readable by
    # readers of this schema that predate parameter plots.
    if result.parameter_slices is not None:
        metadata["parameter_slices"] = json_value(result.parameter_slices)
    arrays = {"object_wave": host_array(result.object_wave)}
    if result.rotation_branches is not None:
        metadata["rotation_branches"], branch_arrays = result.rotation_branches.to_record()
        metadata["rotation_branches"] = json_value(metadata["rotation_branches"])
        arrays.update(branch_arrays)
    _write_atomically(array_path, "wb", lambda file: np.savez(file, **arrays))
    _write_atomically(metadata_path, "w", lambda file: file.write(json.dumps(metadata, indent=2) + "\n"), encoding="utf-8")
    result.reused = False
    result.saved_path = array_path
    result.metadata = metadata
    return result


def load_result(
    *,
    paths: tuple[Path, Path],
    signature: dict[str, object],
    backend: Literal["cuda", "mps"],
) -> SSBResult | None:
    """Load one exact saved result, or return ``None`` on any mismatch."""

    array_path, metadata_path = paths
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        found = metadata.get("schema")
        if found != RESULT_SCHEMA:
            print(f"SSB: {metadata_path.name} in {metadata_path.parent} is a {found!r} record; this version reads "
                  f"{RESULT_SCHEMA!r} only, so the result is computed again and replaces it.")
            return None
        if metadata.get("signature") != signature:
            return None
        saved = dict(metadata["result"])
        saved.setdefault("parameter_slices", metadata.get("parameter_slices"))
        saved.setdefault("calibration_context", metadata.get("calibration_context"))
        with np.load(array_path, allow_pickle=False) as arrays:
            object_wave = arrays["object_wave"]
            rotation_branches = (None if metadata.get("rotation_branches") is None
                                 else RotationBranches.from_record(metadata["rotation_branches"], arrays))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    if backend == "cuda":
        object_wave = cp.asarray(object_wave)
    for name in ("object_sampling_A", "bright_field_center_px", "sample_tilt_mrad", "phase_clim", "fit_scan_region"):
        if isinstance(saved.get(name), list):
            saved[name] = tuple(saved[name])
    note = retired_estimator_note(saved.get("phase_estimator"), f"{array_path.name} in {array_path.parent}")
    if note is not None:
        print(note)
    return SSBResult(
        object_wave=object_wave,
        rotation_branches=rotation_branches,
        reused=True,
        saved_path=array_path,
        metadata=metadata,
        **saved,
    )


def result_paths(
    save_dir: str | Path,
    operation: str,
) -> tuple[Path, Path]:
    """Resolve the array and readable metadata paths of one saved result in the results directory ``save_dir``.

    The result is ``ssb-<operation>.npz`` beside its readable ``ssb-<operation>.json``. A path ending in ``.npz`` or
    ``.json`` names a file, not a directory, and is refused rather than read as a folder of that name.
    """

    destination = Path(save_dir).expanduser().resolve()
    if destination.suffix.lower() in (".npz", ".json"):
        raise ValueError(
            f"save_dir names a results directory; got the file {destination.name!r}. Pass its folder, for example "
            f"save_dir={str(destination.with_suffix('').name)!r}: the result is written there as "
            f"ssb-{operation}.npz and ssb-{operation}.json."
        )
    return (
        destination / f"ssb-{operation}.npz",
        destination / f"ssb-{operation}.json",
    )


# The "schema" of acquisition_identity(); a saved record whose identity carries another is not matched.
ACQUISITION_IDENTITY_SCHEMA = "quantem.acquisition-identity/1"
# Bytes of a file's start that its identity hashes: an HDF5 master's superblock, header and metadata, a .qem
# header, and enough of any other file to tell two acquisitions apart without reading the whole scan.
IDENTITY_HEADER_BYTES = 1 << 20


def file_identity(path: str | os.PathLike[str]) -> dict[str, object]:
    """Identify one file by its content, wherever it lives: its name, size and the SHA-256 of its first MiB.

    No path, inode, device or time enters it, so a copied or moved file keeps its identity and a different file
    of the same name does not share it.
    """
    path = Path(path).expanduser()
    with path.open("rb") as handle:
        header = handle.read(IDENTITY_HEADER_BYTES)
    return {
        "file_name": path.name,
        "size": path.stat().st_size,
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "header_bytes": len(header),
    }


def acquisition_identity(
    source_path: str | os.PathLike[str],
    *,
    scan_shape: tuple[int, int] | None = None,
    hot_pixel_correction: dict[str, object] | None = None,
) -> dict[str, object]:
    """A portable content signature of one acquisition, what a saved, shareable record matches on.

    It is the file's ``file_identity`` (name, size, SHA-256 of its first MiB), the ``file_identity`` of each data file
    an HDF5 master links (named relative to the master's folder), the scan and detector shapes, and how its flagged
    detector pixels were held (``hot_pixel_correction``: method, count, whether applied, largest median window).
    A record saved with it is restored after the data folder is moved or copied, and refused for another
    acquisition (another size or header) or for the same file with its flagged pixels held differently. It never
    holds an absolute path, inode, device or modification time.

    Examples
    --------
    >>> identity = acquisition_identity("scan_master.h5", hot_pixel_correction=loaded.metadata["hot_pixel_correction"])
    """
    path = Path(source_path).expanduser()
    identity: dict[str, object] = {"schema": ACQUISITION_IDENTITY_SCHEMA, **file_identity(path)}
    linked, detector_shape = [], None
    try:
        inspected = inspect_source(path, scan_shape=scan_shape)
    except (OSError, RuntimeError, TypeError, ValueError):
        inspected = None  # a format io.inspect cannot read is still identified by its own bytes
    if inspected is not None:
        scan_shape = scan_shape if scan_shape is not None else inspected.scan_shape
        detector_shape = inspected.detector_shape
        folder = path.resolve().parent
        for record in (inspected.source_signature or {}).get("files", ()):
            other = Path(str(record.get("path", "")))
            if other.resolve() != path.resolve() and other.is_file():
                # the start of each data file holds its first compressed frames: two scans with equal metadata differ here
                linked.append({**file_identity(other), "file_name": Path(os.path.relpath(other.resolve(), folder)).as_posix()})
    record = hot_pixel_correction or {}
    identity.update(
        linked_files=sorted(linked, key=lambda item: item["file_name"]),
        scan_shape=None if scan_shape is None else [int(value) for value in scan_shape],
        detector_shape=None if detector_shape is None else [int(value) for value in detector_shape],
        hot_pixel_correction=None if not record else {
            name: json_value(record.get(name)) for name in ("method", "pixel_count", "applied", "largest_window_px")},
    )
    return identity


def portable_source(source_path: str | os.PathLike[str] | None, record_folder: str | os.PathLike[str]) -> dict | None:
    """Where a shareable record's data is, without an absolute path: its file name, and its path relative to the
    record's folder when the data lives under that folder (``relative_path``, else None).

    A reader finds the data from the record's own location, so the folder can be moved, copied or shared whole.
    """
    if source_path is None:
        return None
    source = Path(source_path).expanduser().resolve()
    folder = Path(record_folder).expanduser().resolve()
    return {
        "file_name": source.name,
        "relative_path": source.relative_to(folder).as_posix() if source.is_relative_to(folder) else None,
    }


def source_identity(
    source_path: str | None,
    *,
    scan_shape: tuple[int, int] | None,
    hot_pixel_correction: dict[str, object] | None,
    storage_path: str | None,
    calibration_path: str | None,
    manifest_path: str | None,
) -> dict[str, object]:
    """Return the complete portable identity of an SSB source, so a saved result is reused only for the same source.

    The detector source is its ``acquisition_identity``; the storage actually read (exact BF columns), the
    calibration and the manifest are each their ``file_identity``. No entry holds a path, inode, device or time.
    """
    if source_path is None:
        raise ValueError(
            "save_dir requires source_path when constructing SSB from an array. "
            "Pass the detector source path to record the array's provenance. "
            "Array inputs are saved but never reused from disk automatically."
        )
    if not Path(source_path).expanduser().exists():
        raise FileNotFoundError(
            f"Cannot save an SSB result because source_path does not exist: "
            f"{Path(source_path).expanduser()}"
        )
    storage = None
    if (
        storage_path is not None
        and Path(storage_path).expanduser().resolve()
        != Path(source_path).expanduser().resolve()
    ):
        storage = _optional_file_identity(storage_path)
    return json_value({
        "detector": acquisition_identity(source_path, scan_shape=scan_shape, hot_pixel_correction=hot_pixel_correction),
        "storage": storage,
        "calibration": None if calibration_path is None else _optional_file_identity(calibration_path),
        "manifest": None if manifest_path is None else _optional_file_identity(manifest_path),
    })


def _optional_file_identity(path: str | Path) -> dict[str, object]:
    """``file_identity`` of a companion file, or its name marked missing; a folder is identified by its files."""
    path = Path(path).expanduser()
    if path.is_file():
        return file_identity(path)
    if path.is_dir():
        return {"file_name": path.name, "files": [
            {"name": item.relative_to(path).as_posix(), **file_identity(item)}
            for item in sorted(path.rglob("*")) if item.is_file()]}
    return {"file_name": path.name, "missing": True}


def software_signature() -> dict[str, object]:
    """Identify the installed SSB implementation that produced a result."""

    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".metal", ".py", ".ts"}:
            digest.update(str(path.relative_to(package)).encode("utf-8"))
            digest.update(path.read_bytes())
    return {"quantem.gpu": version("quantem.gpu"), "ssb_source_sha256": digest.hexdigest()}


def json_value(value: object) -> object:
    """Return a deterministic JSON-safe representation of one value."""

    if isinstance(value, dict):
        return {
            str(key): json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"Cannot save SSB metadata value of type {type(value).__name__}."
    )


def _git_provenance() -> dict[str, object] | None:
    """Return compact repository provenance when running from a checkout."""

    repository = Path(__file__).resolve().parents[4]

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = git("status", "--short", "--untracked-files=all")
    return {
        "commit": commit,
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
    }


def _write_atomically(path: Path, mode: str, write: Callable, **open_options: str) -> None:
    """Write one saved-result file through a sibling temporary file renamed over ``path``.

    A crash or a concurrent reader never sees a half-written array or metadata file: ``os.replace`` swaps the
    complete file in one step, and the temporary file is removed when writing fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode=mode, prefix=f".{path.name}.", dir=path.parent, delete=False, **open_options
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            write(temporary)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
