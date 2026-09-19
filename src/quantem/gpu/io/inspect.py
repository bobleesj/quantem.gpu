"""Header-only inspection for 4D-STEM sources."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
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

    def _summary(self):
        """Return concise header evidence, without dumping masks or metadata."""
        import pandas as pd

        geometry = lambda shape: ' × '.join(map(str, shape)) if shape else 'Unknown'
        rows = {
            'Source': self.source_signature.get('path', self.source_kind),
            'Format': self.metadata.get('container', self.source_kind),
            'Encoding': self.metadata.get('resident_profile', 'Not specified'),
            'Scan shape': geometry(self.scan_shape),
            'Diffraction shape': geometry(self.detector_shape),
            'Stored dtype': self.dtype or 'Unknown',
            'Frames': self.actual_frames,
            'Representation': self.metadata.get('representation', 'Not specified'),
            'Header status': self.reason.replace('_', ' '),
            'Next step': self.action,
        }
        return pd.DataFrame({'Value': rows}).rename_axis('Property')

    def __repr__(self):
        return self._summary().to_string()

    def _repr_html_(self):
        return self._summary().to_html(escape=True)


def inspect(
    filepath: str | PathLike[str],
    *,
    scan_shape: tuple[int, int] | None = None,
) -> Inspection:
    """Inspect source geometry and small validity metadata before loading.

    Parameters
    ----------
    filepath
        HDF5/encoded file or complete prepared-series folder.
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
    >>> info = inspect("acquisition.qem")  # doctest: +SKIP
    >>> info.scan_shape, info.detector_shape  # doctest: +SKIP
    """
    path = Path(filepath)
    from ._streamed_file import is_streamed_file
    from ._qem_reference import read_envelope
    from ._qem_metadata import effective_metadata

    if is_streamed_file(path):
        header, _ = read_envelope(path)
        shape = tuple(header["shape"])
        matches = scan_shape is None or tuple(scan_shape) == shape[:2]
        metadata = dict(effective_metadata(header.get("metadata", {}), header["scientific_metadata"]), resident_bytes=header["bytes"],
                        source_kind="resident", representation="encoded")
        metadata["scientific_metadata"] = header["scientific_metadata"]
        metadata['container'] = header.get('container', 'QEM')
        metadata['resident_profile'] = header.get('codec', header.get('profile', 'Unknown'))
        mask = None
        if "valid" in header:
            valid = np.unpackbits(np.frombuffer(bytes.fromhex(header["valid"]), np.uint8))
            mask = (valid[:math.prod(shape[2:])] == 0).astype(np.uint32).reshape(shape[2:])
        return Inspection(
            matches, "header_complete_payload_unverified" if matches else "scan_shape_mismatch",
            "Load to verify and decode the saved measurements.",
            metadata, mask, "resident", math.prod(shape[:2]), math.prod(shape[:2]),
            shape[:2], shape[2:], header["dtype"], {"path": str(path.resolve())},
        )
    if path.suffix.lower() in {".dm3", ".dm4"}:
        from ._digitalmicrograph import NoDiffractionImage, read_dm_source

        try:
            source = read_dm_source(path, scan_shape)
        except NoDiffractionImage:
            return Inspection(
                False, "not_4dstem", "Choose the 4D diffraction image, not a survey image.",
                {}, None, "digitalmicrograph", None, None, None, None, None,
                {"path": str(path.resolve())},
            )
        shape = source.shape
        return Inspection(
            True, "complete_dataset", "Load the complete DigitalMicrograph acquisition.",
            source.metadata, None, "digitalmicrograph", shape[0] * shape[1],
            shape[0] * shape[1], shape[:2], shape[2:], source.dtype.name,
            {"path": str(source.path), "size": source.signature[0],
             "mtime_ns": source.signature[1], "data_offset": source.offset},
        )
    if path.is_dir() and (path / "checkpoint.json").is_file():
        document = json.loads((path / "checkpoint.json").read_text())
        layout = document["layout"]
        shape = tuple(layout["shape"])
        if len(shape) != 5 or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("Prepared series must declare five positive dimensions.")
        metadata = dict(layout, representation="encoded", series_shape=shape[:1],
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
    if representation is DataRepresentation.PAIRED:
        return _inspect_paired(path, scan_shape)
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


def _inspect_paired(path: Path, scan_shape) -> Inspection:
    """Read the declared geometry of a saved paired resident form; arrays stay unread."""
    from quantem.gpu._compact.paired import QUERY_ABI, _read_header

    header, _ = _read_header(path)
    shape = tuple(header.get("shape", ()))
    if header.get("query_abi") != QUERY_ABI or len(shape) != 4 or header.get("dtype") not in ("uint8", "uint16"):
        raise ValueError(f"{path.name} is not a paired resident form for query ABI {QUERY_ABI}.")
    matches = scan_shape is None or tuple(scan_shape) == shape[:2]
    resident_bytes = sum(spec["nbytes"] for chunk in header["chunks"] for spec in chunk["arrays"])
    metadata = dict(representation="paired", resident_profile=QUERY_ABI, working_shape=shape,
                    scan_shape=shape[:2], detector_shape=shape[2:], dtype=header["dtype"],
                    resident_bytes=resident_bytes, chunks=len(header["chunks"]))
    valid = np.unpackbits(np.frombuffer(bytes.fromhex(header["valid"]), np.uint8))[: shape[2] * shape[3]].reshape(shape[2:])
    mask = (valid == 0).astype(np.uint32)
    return Inspection(matches, "header_complete_payload_unverified" if matches else "scan_shape_mismatch",
                      "Load to reopen the exact resident arrays.", metadata, mask, "paired",
                      shape[0] * shape[1],
                      int(np.prod(scan_shape)) if scan_shape is not None else shape[0] * shape[1],
                      shape[:2], shape[2:], header["dtype"],
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
