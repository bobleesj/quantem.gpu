"""Header-only inspection for 4D-STEM sources."""

import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from quantem.gpu.formats.emd import dataset_metadata
from quantem.gpu.formats.empad import load_array_source
from quantem.gpu.formats.hdf5.master import get_metadata, read_pixel_mask
from quantem.gpu.formats.hdf5.readiness import inspect_master_readiness
from quantem.gpu.formats.prepared_stack import prepared_stack_metadata
from quantem.gpu.formats.qem.metadata import effective_metadata
from quantem.gpu.formats.qem.snapshot import decode_valid, is_qem_file, read_envelope
from quantem.gpu.io.digitalmicrograph import NoDiffractionImage, read_dm_source
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.resident.cuda.paired import QUERY_ABI, _read_header


@dataclass(frozen=True)
class Inspection:
    """Header and small calibration state for one 4D-STEM source."""

    ready: bool
    reason: str
    action: str
    metadata: dict[str, object]
    pixel_mask: np.ndarray | None
    source_kind: str
    actual_frames: int | None
    expected_frames: int | None
    scan_shape: tuple[int, int] | None
    detector_shape: tuple[int, int] | None
    dtype: str | None
    source_signature: dict[str, object]

    def _summary(self):
        """Return concise header evidence, without dumping masks or metadata."""
        # pandas only formats this summary; importing it on first display keeps
        # it out of every load, and environments without it can still load.
        import pandas as pd

        def geometry(shape):
            return ' × '.join(map(str, shape)) if shape else 'Unknown'

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
    dataset_path: str | None = None,
) -> Inspection:
    """Inspect source geometry and small validity metadata before loading.

    Parameters
    ----------
    filepath
        HDF5 or encoded file.
    scan_shape
        Optional expected ``(scan_row, scan_col)`` shape.
    dataset_path
        HDF5 measurement dataset to inspect when the container has several
        acquisitions. Four-dimensional arrays use scan row, scan column,
        detector row, detector column order. A frame-first 3D array requires
        an explicit ``scan_shape``.

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
    if is_qem_file(path):
        header, start = read_envelope(path)
        if start + header["bytes"] != path.stat().st_size:
            raise ValueError("Incomplete QEM payload; recopy the complete file.")
        shape = tuple(header["shape"])
        matches = scan_shape is None or tuple(scan_shape) == shape[:2]
        metadata = dict(effective_metadata(header.get("metadata", {}), header["scientific_metadata"]), resident_bytes=header["bytes"],
                        source_kind="resident", representation="encoded")
        metadata["scientific_metadata"] = header["scientific_metadata"]
        metadata['container'] = header.get('container', 'QEM')
        metadata['resident_profile'] = header.get('codec', header.get('profile', 'Unknown'))
        mask = None
        if "valid" in header:
            mask = (~decode_valid(header["valid"], shape[2:])).astype(np.uint32)
        return Inspection(
            matches, "header_complete_payload_unverified" if matches else "scan_shape_mismatch",
            "Load to verify and decode the saved measurements.",
            metadata, mask, "resident", math.prod(shape[:2]), math.prod(shape[:2]),
            shape[:2], shape[2:], header["dtype"], {"path": str(path.resolve())},
        )
    if path.suffix.lower() in {".dm3", ".dm4"}:
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
    if path.suffix.lower() in {".npy", ".raw", ".xml"}:
        data, source_metadata = load_array_source(path, scan_shape)
        try:
            shape = tuple(data.shape)
            dtype = data.dtype.name
            metadata = dict(
                source_metadata, scan_shape=shape[:2], detector_shape=shape[2:],
                source_shape=shape, dtype=dtype, n_frames=math.prod(shape[:2]),
            )
            status = path.stat()
            return Inspection(
                True, "complete_dataset", "Load the complete acquisition into ANS.",
                metadata, None, metadata["source_kind"], math.prod(shape[:2]),
                math.prod(shape[:2]), shape[:2], shape[2:], dtype,
                {"path": str(path.resolve()), "size": status.st_size,
                 "mtime_ns": status.st_mtime_ns},
            )
        finally:
            data._mmap.close()
    representation = DataRepresentation.detect_source(filepath)
    if representation is DataRepresentation.PAIRED:
        return _inspect_paired(path, scan_shape)
    readiness = inspect_master_readiness(filepath, scan_shape=scan_shape)
    try:
        metadata = get_metadata(str(filepath))
    except (OSError, KeyError, TypeError, ValueError):
        metadata = {}

    prepared = {}
    try:
        with h5py.File(filepath, "r") as source:
            prepared = prepared_stack_metadata(source["dp"]) if "dp" in source else {}
    except (OSError, KeyError, TypeError, ValueError):
        # Preserve the readiness diagnostic for corrupt or incomplete acquisitions.
        pass
    if prepared:
        metadata.update(prepared)
        if dataset_path is None and prepared.get("scan_shape") is not None:
            dataset_path = "/dp"
    # Explicit 4D dimensions take precedence over square-scan inference.
    candidates = []
    try:
        with h5py.File(filepath, "r") as source:
            if dataset_path is not None:
                selected = source[dataset_path]
                if not isinstance(selected, h5py.Dataset) or selected.ndim not in (3, 4):
                    raise ValueError("dataset_path must select a 3D or 4D measurement array.")
                if selected.ndim == 3:
                    selected_scan = scan_shape or prepared.get("scan_shape")
                    if selected_scan is None:
                        raise ValueError("A frame-first 3D dataset needs scan_shape=(rows, columns).")
                    shape = (*selected_scan, *selected.shape[1:])
                    if math.prod(selected_scan) != selected.shape[0]:
                        raise ValueError("scan_shape must cover every frame in the selected dataset.")
                else:
                    shape = selected.shape
                candidates.append((selected.name, shape, selected.dtype))
            def visit(name, value):
                if isinstance(value, h5py.Dataset) and value.ndim == 4:
                    candidates.append((name, value.shape, value.dtype))
            if dataset_path is None:
                source.visititems(visit)
    except (OSError, KeyError, TypeError, ValueError):
        if dataset_path is not None:
            raise
        # Preserve the readiness diagnosis when a source cannot be traversed.
        candidates.clear()
    if len(candidates) == 1:
        name, shape, dtype = candidates[0]
        with h5py.File(filepath, "r") as source:
            metadata = dataset_metadata(source[name], metadata)
        if prepared:
            metadata["source_metadata"].update(prepared["source_metadata"])
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
    if len(candidates) > 1:
        raise ValueError(
            "Multiple 4D acquisitions found; pass dataset_path to select one: "
            + ", ".join(name for name, _, _ in candidates)
        )
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
    if scan_shape is None and metadata.get("scan_shape") is not None:
        scan_shape = metadata["scan_shape"]
    ready, reason, action = readiness.ready, readiness.reason, readiness.action
    if ready and scan_shape is None:
        # A complete acquisition still cannot be loaded until its raster is known.
        ready = False
        reason = (
            f"{readiness.actual_frames} stored frames do not form a square scan "
            "and the master records no scan shape"
        )
        action = "Pass scan_shape=(rows, columns) to io.load."
    return Inspection(
        ready=ready,
        reason=reason,
        action=action,
        metadata=metadata,
        pixel_mask=pixel_mask,
        source_kind=readiness.source_kind,
        actual_frames=readiness.actual_frames,
        expected_frames=readiness.expected_frames,
        scan_shape=None if scan_shape is None else tuple(int(value) for value in scan_shape),
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
    header, _ = _read_header(path)
    shape = tuple(header.get("shape", ()))
    if header.get("query_abi") != QUERY_ABI or len(shape) != 4 or header.get("dtype") not in ("uint8", "uint16"):
        raise ValueError(f"{path.name} is not a paired resident form for query ABI {QUERY_ABI}.")
    matches = scan_shape is None or tuple(scan_shape) == shape[:2]
    resident_bytes = sum(spec["nbytes"] for chunk in header["chunks"] for spec in chunk["arrays"])
    metadata = dict(representation="paired", resident_profile=QUERY_ABI, working_shape=shape,
                    scan_shape=shape[:2], detector_shape=shape[2:], dtype=header["dtype"],
                    resident_bytes=resident_bytes, chunks=len(header["chunks"]))
    mask = (~decode_valid(header["valid"], shape[2:])).astype(np.uint32)
    return Inspection(matches, "header_complete_payload_unverified" if matches else "scan_shape_mismatch",
                      "Load to reopen the exact resident arrays.", metadata, mask, "paired",
                      shape[0] * shape[1],
                      int(np.prod(scan_shape)) if scan_shape is not None else shape[0] * shape[1],
                      shape[:2], shape[2:], header["dtype"],
                      {"path": str(path.resolve()), "size": path.stat().st_size})
