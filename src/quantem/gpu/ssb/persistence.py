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
from quantem.gpu.ssb.results import SSBResult, host_array

# One number versions everything a saved SSB fit means: the layout of its record and the convention its values follow.
# Bump it when fitted values acquire a different physical meaning or record layout: the automatic
# detector sampling rule, the unit, sign or angle reference of an aberration, the rotation convention,
# the bright-field selection rule, the loss definition, or the record layout. Every older fit is then fitted again.
# A reconstruction-only policy belongs in the result signature instead. This invalidates exact image reuse
# without declaring unchanged fitted coefficients unreadable or rescheduling a historical acquisition.
# 7: the record follows the shared vocabulary. The rotation is one physical angle, scan_detector_rotation_deg in
#    [0, 360), with detector_transposed; it was an angle below 180 with com_reversed. The saved result's own "schema"
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
# The same for a saved result record (ssb-find_aberrations.json, ssb-reconstruct.json) and its signature.
RESULT_SCHEMA = f"quantem.ssb-result/{SCHEMA}"


_ARTIFACT_FIELDS = {"object_wave", "rotation_check_phases", "reused", "saved_path", "metadata", "parameter_slices"}


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
    # Optional diagnostics leave the scientific result payload readable by
    # readers of this schema that predate parameter plots.
    if result.parameter_slices is not None:
        metadata["parameter_slices"] = json_value(result.parameter_slices)
    arrays = {"object_wave": host_array(result.object_wave)}
    if result.rotation_check_phases is not None:
        arrays["rotation_check_phases"] = result.rotation_check_phases
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
        if metadata.get("schema") != RESULT_SCHEMA or metadata.get("signature") != signature:
            return None
        saved = dict(metadata["result"])
        saved.setdefault("parameter_slices", metadata.get("parameter_slices"))
        with np.load(array_path, allow_pickle=False) as arrays:
            object_wave = arrays["object_wave"]
            rotation_check_phases = arrays["rotation_check_phases"] if saved.get("initial_scan_detector_rotation_deg") is not None else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    if backend == "cuda":
        object_wave = cp.asarray(object_wave)
    for name in ("object_sampling_A", "bright_field_center_px", "sample_tilt_mrad", "phase_clim", "fit_scan_region"):
        if isinstance(saved.get(name), list):
            saved[name] = tuple(saved[name])
    return SSBResult(
        object_wave=object_wave,
        rotation_check_phases=rotation_check_phases,
        reused=True,
        saved_path=array_path,
        metadata=metadata,
        **saved,
    )


def result_paths(
    save_to: str | Path,
    operation: Literal["find_aberrations", "reconstruct"],
) -> tuple[Path, Path]:
    """Resolve the array and readable metadata paths for one saved result."""

    destination = Path(save_to).expanduser().resolve()
    if destination.suffix:
        if destination.suffix.lower() != ".npz":
            raise ValueError(
                "save_to must be a directory or an .npz file, for example "
                "save_to='results/ssb' or save_to='results/ssb-fit.npz'."
            )
        return destination, destination.with_suffix(".json")
    return (
        destination / f"ssb-{operation}.npz",
        destination / f"ssb-{operation}.json",
    )


def source_signature(
    source_path: str | None,
    *,
    scan_shape: tuple[int, int] | None,
    storage_path: str | None,
    calibration_path: str | None,
    manifest_path: str | None,
) -> dict[str, object]:
    """Return the complete identity of an SSB source, so a saved result is reused only for the same source.

    The detector source is identified by its inspected signature (file size and times when ``io.inspect`` cannot
    read the format); the storage actually read (exact BF columns), the calibration and the manifest are identified
    by their paths, sizes and times.
    """
    if source_path is None:
        raise ValueError(
            "save_to requires source_path when constructing SSB from an array. "
            "Pass the detector source path to record the array's provenance. "
            "Array inputs are saved but never reused from disk automatically."
        )
    if not Path(source_path).expanduser().exists():
        raise FileNotFoundError(
            f"Cannot save an SSB result because source_path does not exist: "
            f"{Path(source_path).expanduser()}"
        )
    try:
        signature = json_value(inspect_source(source_path, scan_shape=scan_shape).source_signature)
    except (OSError, RuntimeError, TypeError, ValueError):
        # formats inspect cannot read are still identified by their file size and times
        signature = path_signature(source_path)
    storage = None
    if (
        storage_path is not None
        and Path(storage_path).expanduser().resolve()
        != Path(source_path).expanduser().resolve()
    ):
        storage = path_signature(storage_path)
    return {
        "detector": signature,
        "storage": storage,
        "calibration": None if calibration_path is None else path_signature(calibration_path),
        "manifest": None if manifest_path is None else path_signature(manifest_path),
    }


def path_signature(path: str | Path) -> dict[str, object]:
    """Describe a source path without reading detector payloads."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return {"path": str(resolved), "missing": True}

    paths = [resolved]
    if resolved.is_dir():
        paths.extend(sorted(item for item in resolved.rglob("*") if item.is_file()))
    files = []
    for item in paths:
        stat = item.stat()
        files.append(
            {
                "path": str(item),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "device": stat.st_dev,
                "inode": stat.st_ino,
            }
        )
    return {"path": str(resolved), "files": files}


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
