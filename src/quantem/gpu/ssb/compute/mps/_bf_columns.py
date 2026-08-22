"""Validation for exact bright-field column companion sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _BfColumnCompanion:
    """Validated exact BF-column source declared by an export."""

    path: Path
    dtype: np.dtype
    max_value: int | None
    detector_bin: int
    provenance: dict[str, object]


class _BfColumnCompanionNotDeclared(FileNotFoundError):
    """No exact BF-column declaration exists at this candidate location."""


def _bf_column_dtype(encoding: object, *, source: Path) -> np.dtype:
    """Return the exact integer dtype declared by a BF-column source."""
    token = str(encoding).lower()
    if token in {"u8", "uint8"}:
        return np.dtype(np.uint8)
    if token in {"u16", "uint16"}:
        return np.dtype(np.uint16)
    raise ValueError(f"Unsupported BF-column encoding {encoding!r}: {source}")


def _legacy_companion(
    cal_path: Path,
    payload: dict[str, object],
) -> _BfColumnCompanion:
    """Resolve the legacy calibration-embedded declaration."""
    relative = payload["bf_column_companion_path"]
    if not relative:
        raise ValueError(f"BF-column companion path is empty: {cal_path}")
    relative_path = Path(str(relative))
    path_candidates = (
        [relative_path]
        if relative_path.is_absolute()
        else [cal_path.parent / relative_path, cal_path.parent.parent / relative_path]
    )
    bf_path = next((path.resolve() for path in path_candidates if path.is_file()), None)
    if bf_path is None:
        raise FileNotFoundError(
            "Exact BF-column companion was not found. Checked: "
            + ", ".join(str(path) for path in path_candidates)
        )
    dtype = _bf_column_dtype(payload["bf_column_encoding"], source=cal_path)
    max_value = None
    for manifest_path in (
        cal_path.parent / "manifest.json",
        cal_path.parent.parent / "manifest.json",
    ):
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bf_meta = (manifest.get("source") or {}).get("bf_columns") or {}
        if bf_meta.get("max_value") is not None:
            max_value = int(bf_meta["max_value"])
            break
    detector_bin = int(payload.get("detector_bin", 1) or 1)
    return _BfColumnCompanion(
        path=bf_path,
        dtype=dtype,
        max_value=max_value,
        detector_bin=detector_bin,
        provenance={
            "declaration": "calibration",
            "calibration_path": str(cal_path),
            "detector_bin": detector_bin,
            "detector_bin_source": (
                "calibration" if "detector_bin" in payload else "default"
            ),
        },
    )


def _linked_manifest(
    cal_path: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Return the linked manifest and its BF-column metadata."""
    for candidate in (
        cal_path.parent / "manifest.json",
        cal_path.parent.parent / "manifest.json",
    ):
        if not candidate.is_file():
            continue
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
        bf_meta = (manifest.get("source") or {}).get("bf_columns")
        if not bf_meta:
            continue
        calibration = manifest.get("calibration")
        if not isinstance(calibration, str) or not calibration:
            raise ValueError(
                f"BF-column manifest must identify its calibration file: {candidate}"
            )
        linked = (candidate.parent / calibration).resolve()
        if linked != cal_path:
            raise ValueError(
                f"BF-column manifest {candidate} links {linked}, not {cal_path}."
            )
        return candidate.resolve(), manifest, bf_meta
    raise ValueError(
        "Calibration declares an exact BF-column companion, but no linked "
        f"manifest with source.bf_columns was found for {cal_path}."
    )


def _validate_storage(
    manifest_path: Path,
    bf_meta: dict[str, object],
) -> tuple[Path, np.dtype]:
    """Validate the companion path and integer encoding."""
    if bf_meta.get("kind") != "bf_columns":
        raise ValueError(
            f"BF-column manifest kind must be 'bf_columns': {manifest_path}"
        )
    if bf_meta.get("order") != "bf,scan":
        raise ValueError(
            f"BF-column manifest order must be 'bf,scan': {manifest_path}"
        )
    relative = bf_meta.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"BF-column manifest path is missing: {manifest_path}")
    export_root = manifest_path.parent.resolve()
    bf_path = (export_root / relative).resolve()
    if not bf_path.is_relative_to(export_root):
        raise ValueError(
            f"BF-column path must stay inside the export folder: {manifest_path}"
        )
    if not bf_path.is_file():
        raise FileNotFoundError(f"Exact BF-column companion was not found: {bf_path}")

    dtype = _bf_column_dtype(bf_meta.get("encoding"), source=manifest_path)
    declared_dtype = bf_meta.get("dtype")
    if declared_dtype is not None and np.dtype(str(declared_dtype)) != dtype:
        raise ValueError(
            f"BF-column dtype {declared_dtype!r} disagrees with encoding "
            f"{bf_meta.get('encoding')!r}: {manifest_path}"
        )
    expected_suffix = ".u8" if dtype == np.dtype(np.uint8) else ".u16"
    if bf_path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"BF-column filename suffix must be {expected_suffix} for {dtype}: {bf_path}"
        )
    return bf_path, dtype


def _validate_geometry(
    cal_path: Path,
    manifest_path: Path,
    payload: dict[str, object],
    bf_meta: dict[str, object],
) -> tuple[list[int], list[int], int]:
    """Validate scan, BF, and detector coordinate grids."""
    scan_region = payload.get("scan_region")
    scan_shape = scan_region.get("shape") if isinstance(scan_region, dict) else None
    if not isinstance(scan_shape, list) or len(scan_shape) != 2:
        raise ValueError(f"Calibration has no valid 2D scan shape: {cal_path}")
    scan_shape = [int(value) for value in scan_shape]
    if bf_meta.get("scan_shape") != scan_shape:
        raise ValueError(
            f"BF-column scan shape {bf_meta.get('scan_shape')} does not match "
            f"calibration {scan_shape}: {manifest_path}"
        )

    rows = payload.get("bf_rows")
    cols = payload.get("bf_cols")
    if not isinstance(rows, list) or not isinstance(cols, list) or len(rows) != len(cols):
        raise ValueError(f"Calibration BF coordinates are invalid: {cal_path}")
    expected_shape = [len(rows), int(np.prod(scan_shape))]
    if bf_meta.get("shape") != expected_shape:
        raise ValueError(
            f"BF-column shape {bf_meta.get('shape')} does not match "
            f"{expected_shape}: {manifest_path}"
        )

    working_shape = payload.get("detector_shape")
    column_shape = bf_meta.get("detector_shape")
    if (
        not isinstance(working_shape, list)
        or len(working_shape) != 2
        or not isinstance(column_shape, list)
        or len(column_shape) != 2
    ):
        raise ValueError(
            f"BF-column detector shapes must be two-dimensional: {manifest_path}"
        )
    working_shape = [int(value) for value in working_shape]
    column_shape = [int(value) for value in column_shape]
    if any(value <= 0 for value in working_shape + column_shape):
        raise ValueError(f"BF-column detector shapes must be positive: {manifest_path}")
    if column_shape != working_shape:
        raise ValueError(
            "BF-column coordinates use detector shape "
            f"{column_shape}, but calibration uses {working_shape}. Exact "
            "columns cannot infer detector binning; export values on the "
            f"calibration grid and declare detector_bin explicitly: {manifest_path}"
        )

    declared_bin = bf_meta.get("detector_bin", payload.get("detector_bin"))
    detector_bin = 1 if declared_bin is None else int(declared_bin)
    if detector_bin < 1:
        raise ValueError(f"detector_bin must be a positive integer: {manifest_path}")
    return scan_shape, expected_shape, detector_bin


def _validate_byte_count(
    manifest_path: Path,
    bf_path: Path,
    bf_meta: dict[str, object],
    dtype: np.dtype,
    expected_shape: list[int],
) -> int:
    """Validate payload size and return the expected byte count."""
    expected_bytes = int(np.prod(expected_shape)) * dtype.itemsize
    declared_bytes = bf_meta.get("bytes")
    actual_bytes = bf_path.stat().st_size
    if declared_bytes != expected_bytes or actual_bytes != expected_bytes:
        raise ValueError(
            "BF-column byte count mismatch: "
            f"declared={declared_bytes}, expected={expected_bytes}, "
            f"actual={actual_bytes}: {manifest_path}"
        )
    if bf_meta.get("bytes_per_bf") not in {
        None,
        expected_shape[1] * dtype.itemsize,
    }:
        raise ValueError(f"BF-column bytes_per_bf is inconsistent: {manifest_path}")
    if bf_meta.get("bits_per_value") not in {None, dtype.itemsize * 8}:
        raise ValueError(f"BF-column bits_per_value is inconsistent: {manifest_path}")
    return expected_bytes


def _manifest_companion(
    cal_path: Path,
    payload: dict[str, object],
) -> _BfColumnCompanion:
    """Resolve and validate the current manifest-declared source."""
    manifest_path, _manifest, bf_meta = _linked_manifest(cal_path)
    bf_path, dtype = _validate_storage(manifest_path, bf_meta)
    _scan_shape, expected_shape, detector_bin = _validate_geometry(
        cal_path,
        manifest_path,
        payload,
        bf_meta,
    )
    expected_bytes = _validate_byte_count(
        manifest_path,
        bf_path,
        bf_meta,
        dtype,
        expected_shape,
    )
    max_value = bf_meta.get("max_value")
    if max_value is not None:
        max_value = int(max_value)
        if max_value < 0 or max_value > int(np.iinfo(dtype).max):
            raise ValueError(f"BF-column max_value is invalid for {dtype}: {manifest_path}")
    declared_bin = bf_meta.get("detector_bin", payload.get("detector_bin"))
    return _BfColumnCompanion(
        path=bf_path,
        dtype=dtype,
        max_value=max_value,
        detector_bin=detector_bin,
        provenance={
            "declaration": "manifest",
            "manifest_path": str(manifest_path),
            "calibration_path": str(cal_path),
            "order": "bf,scan",
            "shape": expected_shape,
            "bytes": expected_bytes,
            "detector_shape": [int(value) for value in payload["detector_shape"]],
            "detector_bin": detector_bin,
            "detector_bin_source": (
                "declared" if declared_bin is not None else "default"
            ),
        },
    )


def _resolve_bf_column_companion(
    cal_path: Path,
    payload: dict[str, object],
) -> _BfColumnCompanion:
    """Resolve one declared BF-column source without a silent fallback."""
    has_legacy = (
        "bf_column_companion_path" in payload
        or "bf_column_encoding" in payload
    )
    if has_legacy:
        missing = [
            name
            for name in ("bf_column_companion_path", "bf_column_encoding")
            if name not in payload
        ]
        if missing:
            raise ValueError(
                "Calibration has an incomplete legacy BF-column declaration "
                f"{missing}: {cal_path}"
            )
        return _legacy_companion(cal_path, payload)

    declared = payload.get("bf_column_companion")
    transport = payload.get("source_transport")
    if declared is True or transport == "bf_columns":
        if declared is False:
            raise ValueError(
                f"Calibration disables BF columns but requests their transport: {cal_path}"
            )
        if declared not in {None, True}:
            raise ValueError(f"bf_column_companion must be true or absent: {cal_path}")
        return _manifest_companion(cal_path, payload)
    if declared not in {None, False}:
        raise ValueError(f"bf_column_companion must be boolean: {cal_path}")
    raise _BfColumnCompanionNotDeclared(
        f"No exact BF-column companion is declared: {cal_path}"
    )
