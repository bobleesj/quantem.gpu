"""Lossless-packed source dispatch and common provenance normalization."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from quantem.gpu.device._cupy import cp

from .integrity import SourceIntegrity
from .models import FourDSTEMData, _release_owned_storage
from .representation import DataRepresentation
from .uint4 import is_packed_uint4


def _source_paths(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> list[str | os.PathLike[str]]:
    """Return source paths without treating one path string as a sequence."""
    if isinstance(source, (str, os.PathLike)):
        return [source]
    return list(source)


def _selected_representation(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    requested: DataRepresentation | str | None,
) -> DataRepresentation:
    """Resolve an explicit representation or inspect existing containers."""
    paths = _source_paths(source)
    if requested is not None:
        selected = DataRepresentation.parse(requested)
        detected = [DataRepresentation.detect_source(path) for path in paths]
        if selected is DataRepresentation.PAIRED and any(
            item not in {DataRepresentation.DENSE, DataRepresentation.PAIRED} for item in detected
        ):
            raise NotImplementedError(
                "representation='paired' streams ordinary HDF5 acquisitions or reopens "
                "saved paired resident forms; transcoding packed or encoded sources into the "
                "paired layout is not implemented."
            )
        if selected is not DataRepresentation.PAIRED and any(item is DataRepresentation.PAIRED for item in detected):
            raise ValueError(
                "A saved paired resident form reopens only as representation='paired'; "
                "omit representation= or request 'paired'."
            )
        if selected is DataRepresentation.ENCODED and any(item is DataRepresentation.PACKED for item in detected):
            raise NotImplementedError("Conversion from prepared packed H5 requires an encoded source or ordinary H5; load prepared packed sources natively.")
        if selected is DataRepresentation.PACKED and any(
            item not in {DataRepresentation.PACKED, DataRepresentation.ENCODED}
            for item in detected
        ):
            raise ValueError(
                "representation='packed' requires a packed or encoded source. "
                "The supplied source is ordinary HDF5. Prepare an immutable "
                "lossless-packed source first or request representation='dense'."
            )
        if selected is DataRepresentation.DENSE and any(
            item is DataRepresentation.PACKED for item in detected
        ):
            raise ValueError(
                "Dense expansion of a Lossless Pack Format source is not a load "
                "side effect. Load its source-native representation and run an "
                "explicit materialization step only when an algorithm requires it."
            )
        return selected
    detected = {DataRepresentation.detect_source(path) for path in paths}
    if len(detected) > 1:
        raise ValueError(
            "One load call cannot mix dense and lossless-packed source containers. "
            "Load each representation separately."
        )
    return detected.pop() if detected else DataRepresentation.DENSE


def _packed_metadata(data: object, backend: str) -> dict[str, Any]:
    """Build common exact metadata from one backend-specific packed source."""
    index = getattr(data, "index", None) or getattr(data, "metadata", None)
    if index is None or not hasattr(index, "manifest"):
        raise TypeError(
            f"The {backend!r} backend did not return lossless-packed source metadata."
        )
    manifest = index.manifest
    shape = tuple(int(value) for value in index.shape)
    source_dtype = np.dtype(str(manifest["source_dtype"])).name
    working_dtype = np.dtype(str(manifest["working_dtype"])).name
    metrics = getattr(data, "load_metrics", None)
    resident_bytes = getattr(data, "memory_pool_used_bytes", None)
    if resident_bytes is None:
        resident_bytes = getattr(metrics, "resident_bytes", None)
    if resident_bytes is None:
        resident_bytes = getattr(metrics, "total_resident_bytes", None)
    if resident_bytes is None:
        resident_bytes = getattr(index, "packed_resident_bytes", None)
    metadata = dict(manifest)
    metadata.update({
        "backend": backend,
        "representation": DataRepresentation.PACKED.value,
        "residency": "device",
        "source_shape": tuple(manifest["source_shape"]),
        "working_shape": shape,
        "scan_shape": shape[:2],
        "detector_shape": shape[2:],
        "source_dtype": source_dtype,
        "working_dtype": working_dtype,
        "dtype": working_dtype,
        "source_logical_tensor_bytes": int(np.prod(shape, dtype=np.int64))
        * np.dtype(source_dtype).itemsize,
        "working_logical_tensor_bytes": int(np.prod(shape, dtype=np.int64))
        * np.dtype(working_dtype).itemsize,
        "physical_resident_bytes": (
            int(resident_bytes) if resident_bytes is not None else None
        ),
        "container_bytes": int(index.file_bytes),
        "lossless_exact": True,
        "source_identity_sha256": index.source_identity_sha256,
        "storage_schema": str(manifest["schema"]),
        "scan_bin": int(manifest["scan_bin"]),
        "detector_bin": int(manifest["detector_bin"]),
        "crop": manifest["crop"],
    })
    return metadata


def _load_packed(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    backend: str,
    expected_source_sha256: str | None,
    device: int | str | None,
    source_integrity: SourceIntegrity | None = None,
) -> FourDSTEMData:
    """Load one prepared exact representation through its accelerator backend."""
    paths = _source_paths(source)
    if len(paths) != 1:
        raise ValueError(
            "Lossless-packed multi-source loading requires a dataset-series owner. "
            "Load one source per call until that public series contract is available."
        )
    from ._compact_h5 import CompactH5Index
    from .backends import resolve_backend

    path = os.fspath(paths[0])
    index = CompactH5Index.from_file(path)
    # Legacy mask-only containers remain readable by their explicit reference
    # APIs, but cannot be promoted to the public raw-lossless representation.
    index.require_raw_reconstruction()
    if source_integrity is not None:
        source_integrity.validate_source(path)
        if expected_source_sha256 not in {None, source_integrity.whole_file_sha256}:
            raise ValueError("expected_source_sha256 conflicts with source_integrity.")
        expected_source_sha256 = source_integrity.whole_file_sha256
    selected_backend = resolve_backend(backend)
    if selected_backend == "cuda":
        from .backends.cuda import load_compact_h5_cuda

        integrity_options = {}
        if index.schema_version == 1 and expected_source_sha256 is not None:
            integrity_options["integrity_mode"] = "whole_file"
            if source_integrity is not None:
                integrity_options.update(
                    integrity_mode="chunked",
                    expected_chunk_sha256=source_integrity.chunk_sha256,
                    integrity_chunk_bytes=source_integrity.chunk_bytes,
                )
        if device is not None:
            if cp is None:
                raise RuntimeError("CUDA loading requires CuPy.")
            with cp.cuda.Device(int(device)):
                data = load_compact_h5_cuda(
                    path,
                    expected_whole_file_sha256=expected_source_sha256,
                    **integrity_options,
                )
        else:
            data = load_compact_h5_cuda(
                path,
                expected_whole_file_sha256=expected_source_sha256,
                **integrity_options,
            )
    elif selected_backend == "mps":
        if index.schema_version != 3:
            raise ValueError(
                "Python MPS currently supports the direct-bitpacked Lossless Pack "
                "profile. Use the native Metal loader for this uint16/LZ4 profile "
                "or select a supported prepared representation."
            )
        from .backends.mps.packed import load_compact_v3_mps

        data = load_compact_v3_mps(
            path, expected_whole_file_sha256=expected_source_sha256
        )
    else:
        raise ValueError(
            "representation='packed' requires an accelerator backend; "
            "backend='cpu' is an explicit dense reference path."
        )
    try:
        return FourDSTEMData(data, _packed_metadata(data, selected_backend))
    except BaseException as error:
        _release_owned_storage(data, failure=error)
        raise


def _record_dense_representation(
    result: FourDSTEMData | list[FourDSTEMData],
) -> FourDSTEMData | list[FourDSTEMData]:
    """Attach common representation fields to existing dense load results."""
    if isinstance(result, list):
        return [_record_dense_representation(item) for item in result]
    metadata = dict(result.metadata)
    data = result.data
    shape = tuple(int(value) for value in getattr(data, "shape", ()))
    dtype = getattr(data, "dtype", metadata.get("dtype"))
    if dtype is not None:
        token = str(dtype).removeprefix("torch.")
        dtype = "uint4" if token == "uint4" else np.dtype(token).name
    selected = (
        DataRepresentation.PACKED
        if is_packed_uint4(data)
        else DataRepresentation.DENSE
    )
    metadata.setdefault("representation", selected.value)
    metadata.setdefault(
        "residency", "host" if metadata.get("backend") == "cpu" else "device"
    )
    if shape:
        metadata.setdefault("working_shape", shape)
    if dtype is not None:
        metadata.setdefault("working_dtype", dtype)
    nbytes = getattr(data, "nbytes", None)
    if nbytes is not None:
        logical_bytes = (
            int(np.prod(shape, dtype=np.int64)) if is_packed_uint4(data) else int(nbytes)
        )
        metadata.setdefault("working_logical_tensor_bytes", logical_bytes)
        metadata.setdefault("physical_resident_bytes", int(nbytes))
    if "lossless_exact" not in metadata:
        source_dtype = metadata.get("source_dtype")
        exact_cast = (
            source_dtype is not None
            and dtype is not None
            and dtype != "uint4"
            and np.can_cast(np.dtype(source_dtype), np.dtype(dtype), casting="safe")
        )
        if (
            exact_cast
            and np.dtype(source_dtype).kind in "iu"
            and np.dtype(dtype).kind == "f"
        ):
            # NumPy calls uint64 -> float64 safe, but its mantissa cannot
            # preserve all integer counts. Require enough significant bits.
            integer_bits = np.iinfo(source_dtype).bits - (
                np.dtype(source_dtype).kind == "i"
            )
            exact_cast = np.finfo(dtype).nmant + 1 >= integer_bits
        # A missing saturation counter is not proof that narrowing was exact.
        # Interpolated scan resampling is not a source-count-preserving cast.
        metadata["lossless_exact"] = bool(
            exact_cast
            and metadata.get("clipped_count", 0) == 0
            and not metadata.get("scan_resampling")
        )
    return FourDSTEMData(data, metadata)
