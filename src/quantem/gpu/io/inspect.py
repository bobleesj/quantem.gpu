"""Header-only inspection for 4D-STEM sources."""
from __future__ import annotations

from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from .load import (
    get_metadata,
    inspect_master_readiness,
    read_pixel_mask,
)
from .representation import DataRepresentation


@dataclass(frozen=True)
class Inspection:
    """Header and small calibration state for one 4D-STEM source."""

    ready: bool
    reason: str
    action: str
    metadata: dict[str, Any]
    pixel_mask: np.ndarray | None
    source_kind: str
    actual_frames: int | None
    expected_frames: int | None
    scan_shape: tuple[int, int] | None
    detector_shape: tuple[int, int] | None
    dtype: str | None
    source_signature: dict[str, Any]


def inspect(
    filepath: str | PathLike[str],
    *,
    scan_shape: tuple[int, int] | None = None,
) -> Inspection:
    """Inspect source geometry and small validity metadata before loading.

    Parameters
    ----------
    filepath
        HDF5/ANS file or complete prepared-series folder.
    scan_shape
        Optional expected ``(scan_row, scan_col)`` shape.

    Returns
    -------
    Inspection
        Readiness, metadata, and the small detector pixel mask. Detector frames
        are not loaded. Encoded header readiness is not payload authentication;
        ``io.load`` performs the required validation before exposing the source.

    Examples
    --------
    >>> info = inspect("acquisition.ans")  # doctest: +SKIP
    >>> info.scan_shape, info.detector_shape  # doctest: +SKIP
    """
    path = Path(filepath)
    if path.is_dir() and (path / "checkpoint.json").is_file():
        document = json.loads((path / "checkpoint.json").read_text())
        layout = document["layout"]
        shape = tuple(layout["shape"])
        if len(shape) != 5 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("Prepared series must declare five positive dimensions.")
        metadata = dict(layout, representation="ans", series_shape=shape[:1],
                        source_kind="prepared", acquisitions=document.get("original_acquisitions", []))
        matches = scan_shape is None or tuple(scan_shape) == shape[1:3]
        ready = matches and document.get("complete") is True and layout.get("complete") is True
        reason = ("scan_shape_mismatch" if not matches else
                  "header_complete_payload_unverified" if ready else "incomplete")
        return Inspection(ready, reason,
                          "Load the complete prepared folder to validate its records.",
                          metadata, None, "prepared", shape[1] * shape[2],
                          int(np.prod(scan_shape)) if scan_shape is not None else shape[1] * shape[2],
                          shape[1:3], shape[3:], np.dtype(layout["source_dtype"]).name,
                          {"path": str(path.resolve())})
    representation = DataRepresentation.detect_source(filepath)
    if representation is DataRepresentation.ANS:
        return _inspect_ans(path, scan_shape)
    if representation is DataRepresentation.PACKED:
        return _inspect_packed(filepath, scan_shape)
    readiness = inspect_master_readiness(filepath, scan_shape=scan_shape)
    try:
        metadata = get_metadata(str(filepath))
    except (OSError, KeyError, TypeError, ValueError):
        metadata = {}
    # Explicit 4D dimensions take precedence over square-scan inference.
    candidates = []
    try:
        with h5py.File(filepath, "r") as source:
            def visit(name, value):
                if isinstance(value, h5py.Dataset) and value.ndim == 4:
                    candidates.append((name, value.shape, value.dtype))
            source.visititems(visit)
    except (OSError, KeyError, TypeError, ValueError):
        # Preserve the readiness diagnosis when a source cannot be traversed.
        candidates.clear()
    if len(candidates) == 1:
        name, shape, dtype = candidates[0]
        matches = scan_shape is None or tuple(scan_shape) == tuple(shape[:2])
        metadata.update(dataset_path=name, scan_shape=shape[:2], detector_shape=shape[2:],
                        source_shape=shape, dtype=np.dtype(dtype).name,
                        representation="dense", n_frames=shape[0] * shape[1])
        return Inspection(matches, "complete_dataset" if matches else "scan_shape_mismatch",
                          "Load the complete dataset.", metadata, read_pixel_mask(filepath),
                          "hdf5_dataset", shape[0] * shape[1],
                          int(np.prod(scan_shape)) if scan_shape is not None else shape[0] * shape[1],
                          tuple(shape[:2]), tuple(shape[2:]), np.dtype(dtype).name,
                          readiness.source_signature)
    metadata.setdefault("scan_shape", scan_shape)
    metadata["representation"] = DataRepresentation.DENSE.value
    metadata["detector_shape"] = readiness.detector_shape
    metadata["dtype"] = (
        np.dtype(readiness.dtype).name
        if readiness.dtype is not None
        else metadata.get("dtype")
    )
    metadata["n_frames"] = readiness.actual_frames
    try:
        pixel_mask = read_pixel_mask(filepath)
    except (OSError, KeyError, TypeError, ValueError):
        pixel_mask = None
    return Inspection(
        ready=readiness.ready,
        reason=readiness.reason,
        action=readiness.action,
        metadata=metadata,
        pixel_mask=pixel_mask,
        source_kind=readiness.source_kind,
        actual_frames=readiness.actual_frames,
        expected_frames=readiness.expected_frames,
        scan_shape=(
            tuple(int(value) for value in scan_shape)
            if scan_shape is not None
            else (
                tuple(int(value) for value in metadata["scan_shape"])
                if metadata.get("scan_shape") is not None
                else None
            )
        ),
        detector_shape=readiness.detector_shape,
        dtype=(
            np.dtype(readiness.dtype).name
            if readiness.dtype is not None
            else None
        ),
        source_signature=readiness.source_signature,
    )


def _inspect_ans(path: Path, scan_shape) -> Inspection:
    """Read bounded declared geometry; payload admission remains the loader's job."""
    from ._ans import MAGIC, _DATA_START, _HEADER, _no_duplicate_keys, _reject_constant

    with path.open("rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("Truncated ANS header; choose a complete encoded file.")
        magic, length, start = _HEADER.unpack(header)
        if magic != MAGIC or start != _DATA_START or not 0 < length <= start - _HEADER.size:
            raise ValueError("Invalid ANS header; choose a supported encoded file.")
        document = json.loads(stream.read(length), object_pairs_hook=_no_duplicate_keys,
                              parse_constant=_reject_constant)
    shape = document.get("shape", ())
    if (len(shape) != 4 or any(type(size) is not int or size < 1 for size in shape)
            or document.get("dtype") not in ("uint8", "uint16")):
        raise ValueError("ANS header must declare four positive dimensions and native integer counts.")
    shape = tuple(shape)
    matches = scan_shape is None or tuple(scan_shape) == shape[:2]
    metadata = dict(document.get("metadata", {}), representation="ans", working_shape=shape,
                    scan_shape=shape[:2], detector_shape=shape[2:], dtype=document["dtype"],
                    encoded_bytes=sum(section["count"] * np.dtype(section["dtype"]).itemsize
                                      for section in document["sections"].values()))
    mask = np.zeros(shape[2:], np.uint32)
    excluded = metadata.get("excluded_detector_pixels", ())
    if len(excluded):
        mask.reshape(-1)[np.asarray(excluded, dtype=np.intp)] = 1
    return Inspection(matches, "header_complete_payload_unverified" if matches else "scan_shape_mismatch",
                      "Load to validate all encoded streams.", metadata, mask, "ans",
                      shape[0] * shape[1],
                      int(np.prod(scan_shape)) if scan_shape is not None else shape[0] * shape[1],
                      shape[:2], shape[2:], document["dtype"],
                      {"path": str(path.resolve()), "size": path.stat().st_size})


def _inspect_packed(
    filepath: str | PathLike[str], scan_shape: tuple[int, int] | None
) -> Inspection:
    """Inspect a packed source without authenticating or decoding its payload."""
    from ._compact_h5 import CompactH5Index

    try:
        index = CompactH5Index.from_file(filepath)
    except (OSError, ValueError) as error:
        return Inspection(
            ready=False,
            reason=f"invalid_lossless_pack_index: {error}",
            action="Use a complete, authenticated Lossless Pack Format container.",
            metadata={"representation": DataRepresentation.PACKED.value},
            pixel_mask=None,
            source_kind="packed",
            actual_frames=None,
            expected_frames=int(np.prod(scan_shape)) if scan_shape else None,
            scan_shape=scan_shape,
            detector_shape=None,
            dtype=None,
            source_signature={},
        )
    shape = index.shape
    matches = scan_shape is None or tuple(scan_shape) == shape[:2]
    raw_lossless = index.raw_reconstruction_available
    reason = (
        "scan_shape_mismatch" if not matches
        else "index_complete_payload_unverified" if raw_lossless
        else "raw_reconstruction_unavailable"
    )
    metadata = dict(index.manifest)
    metadata.update(
        representation=DataRepresentation.PACKED.value,
        working_shape=shape,
        scan_shape=shape[:2],
        detector_shape=shape[2:],
        dtype=index.manifest["working_dtype"],
        n_frames=shape[0] * shape[1],
        raw_reconstruction_available=raw_lossless,
    )
    mask = np.zeros(shape[2] * shape[3], dtype=np.uint32)
    mask[list(index.excluded_detector_pixels)] = 1
    return Inspection(
        ready=matches and raw_lossless,
        reason=reason,
        action=(
            f"Use scan_shape={shape[:2]}." if not matches
            else "Load to authenticate the payload." if raw_lossless
            else "Prepare a raw-lossless container retaining excluded detector values."
        ),
        metadata=metadata,
        pixel_mask=mask.reshape(shape[2:]),
        source_kind="packed",
        actual_frames=shape[0] * shape[1],
        expected_frames=(
            int(np.prod(scan_shape)) if scan_shape is not None else shape[0] * shape[1]
        ),
        scan_shape=shape[:2],
        detector_shape=shape[2:],
        dtype=str(index.manifest["working_dtype"]),
        source_signature={
            "source_identity_sha256": index.source_identity_sha256,
            "container_bytes": index.file_bytes,
            "storage_schema": index.manifest["schema"],
        },
    )
