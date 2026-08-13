"""Private saved-result storage for the public SSB workflow."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from .results import SSBResult


SCHEMA = 1
_ARTIFACT_FIELDS = {"object_wave", "reused", "saved_path", "metadata"}


def software_signature() -> dict[str, object]:
    """Identify the installed SSB implementation that produced a result."""

    from quantem.gpu import __version__

    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".metal", ".py", ".ts"}:
            digest.update(str(path.relative_to(package)).encode("utf-8"))
            digest.update(path.read_bytes())
    return {"quantem.gpu": __version__, "ssb_source_sha256": digest.hexdigest()}


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
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
            }
        )
    return {"path": str(resolved), "files": files}


def result_paths(
    save_to: str | Path,
    operation: Literal["fit", "reconstruct"],
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


def _save_npz(path: Path, object_wave: object) -> None:
    """Atomically save one complex object wave."""

    path.parent.mkdir(parents=True, exist_ok=True)
    get = getattr(object_wave, "get", None)
    array = np.asarray(get() if callable(get) else object_wave)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            np.savez(temporary, object_wave=array)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_json(path: Path, metadata: dict[str, object]) -> None:
    """Atomically save readable SSB provenance."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(metadata, temporary, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_provenance(),
        "input": json_value(input_metadata),
        "signature": signature,
        "result": json_value(result_metadata),
    }
    _save_npz(array_path, result.object_wave)
    _save_json(metadata_path, metadata)
    result.reused = False
    result.saved_path = array_path
    result.metadata = metadata
    return result


def load_result(
    *,
    paths: tuple[Path, Path],
    signature: dict[str, object],
    backend: Literal["cuda", "mps", "webgpu"],
) -> SSBResult | None:
    """Load one exact saved result, or return ``None`` on any mismatch."""

    array_path, metadata_path = paths
    if not array_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != SCHEMA or metadata.get("signature") != signature:
            return None
        saved = dict(metadata["result"])
        with np.load(array_path, allow_pickle=False) as arrays:
            object_wave = arrays["object_wave"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    if backend == "cuda":
        import cupy as cp

        object_wave = cp.asarray(object_wave)
    for name in ("scan_sampling_A", "bf_center"):
        if isinstance(saved.get(name), list):
            saved[name] = tuple(saved[name])
    return SSBResult(
        object_wave=object_wave,
        reused=True,
        saved_path=array_path,
        metadata=metadata,
        **saved,
    )
