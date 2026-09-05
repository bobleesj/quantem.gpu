"""Header-only inspection for 4D-STEM sources."""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

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
    """Inspect whether a master is complete and internally consistent.

    Parameters
    ----------
    filepath
        HDF5 master path.
    scan_shape
        Optional expected ``(scan_row, scan_col)`` shape.

    Returns
    -------
    Inspection
        Readiness, metadata, and the small detector pixel mask. Detector frames
        are not loaded.
    """
    if (
        DataRepresentation.detect_source(filepath)
        is DataRepresentation.LOSSLESS_PACKED
    ):
        return _inspect_lossless_packed(filepath, scan_shape)
    readiness = inspect_master_readiness(filepath, scan_shape=scan_shape)
    try:
        metadata = get_metadata(str(filepath))
    except (OSError, KeyError, TypeError, ValueError):
        metadata = {}
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


def _inspect_lossless_packed(
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
            metadata={"representation": DataRepresentation.LOSSLESS_PACKED.value},
            pixel_mask=None,
            source_kind="lossless_packed",
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
        representation=DataRepresentation.LOSSLESS_PACKED.value,
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
        source_kind="lossless_packed",
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
