"""Bounded ingestion for HDF5 layouts outside the direct chunk decoder."""

import bisect
import math
from pathlib import Path

import h5py
import numpy as np

from . import _array_resident
from ._array_resident import load_array_resident


def _read_four_dimensional_frames(data, result, first, stop):
    """Read one flattened interval as one union of at most three scan slabs."""
    columns = data.shape[1]
    source_space = data.id.get_space()
    memory_space = h5py.h5s.create_simple(result.shape)
    try:
        source_space.select_none()
        cursor = first
        while cursor < stop:
            row, column = divmod(cursor, columns)
            rows = (stop - cursor) // columns if column == 0 else 0
            count = columns if rows else min(columns - column, stop - cursor)
            rows = max(1, rows)
            source_space.select_hyperslab(
                (row, column, 0, 0),
                (rows, count, *data.shape[2:]),
                op=h5py.h5s.SELECT_OR,
            )
            cursor += rows * count
        data.id.read(memory_space, source_space, result)
    finally:
        source_space.close()
        memory_space.close()


def load_hdf5_array_resident(
    path,
    *,
    scan_shape,
    dataset_path,
    backend,
    device,
    verbose,
    hot_pixel_correction,
    info,
    auto_narrow=True,
):
    """Use storage-library reads only where the direct chunk reader cannot apply."""
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    selected = dataset_path or info.metadata.get("dataset_path")
    with h5py.File(path, "r") as handle:
        if selected is not None:
            datasets = [handle[selected]]
        elif "entry/data" in handle:
            group = handle["entry/data"]
            datasets = [
                group[name] for name in sorted(group) if name.startswith("data")
            ]
        else:
            return None
        if not datasets:
            return None
        direct = all(
            data.ndim == 3
            and data.chunks is not None
            and data.chunks[0] == 1
            and 32008
            in {
                data.id.get_create_plist().get_filter(index)[0]
                for index in range(data.id.get_create_plist().get_nfilters())
            }
            for data in datasets
        )
        # Keep the existing accelerated bitshuffle/LZ4 reader for its native layout.
        if (
            dataset_path is None
            and direct
            and np.dtype(info.dtype) in (np.dtype("uint8"), np.dtype("uint16"))
        ):
            return None
        if backend == "cuda" and np.dtype(info.dtype) == np.dtype("uint32"):
            return None  # Existing count-range verification owns exact narrowing.
        starts = [0]
        for data in datasets:
            if (
                data.ndim not in (3, 4)
                or data.shape[-2:] != shape[2:]
                or np.dtype(data.dtype) != np.dtype(info.dtype)
            ):
                raise ValueError(
                    "HDF5 detector datasets disagree in geometry or dtype; select one calibrated acquisition."
                )
            starts.append(starts[-1] + math.prod(data.shape[:-2]))
        if starts[-1] != math.prod(shape[:2]):
            raise ValueError(
                "HDF5 frame count disagrees with scan geometry; restore the complete acquisition."
            )
        signatures = {
            Path(data.file.filename): (
                Path(data.file.filename).stat().st_size,
                Path(data.file.filename).stat().st_mtime_ns,
            )
            for data in datasets
        }
        status = Path(path).stat()
        signatures[Path(path)] = (status.st_size, status.st_mtime_ns)

        def read_frames(first, stop):
            result = np.empty((stop - first, *shape[2:]), np.dtype(info.dtype))
            cursor = first
            while cursor < stop:
                index = bisect.bisect_right(starts, cursor) - 1
                end = min(stop, starts[index + 1])
                data = datasets[index]
                local = cursor - starts[index]
                if data.ndim == 3:
                    data.read_direct(
                        result,
                        source_sel=np.s_[local : local + end - cursor],
                        dest_sel=np.s_[cursor - first : end - first],
                    )
                    cursor = end
                else:
                    _read_four_dimensional_frames(
                        data,
                        result[cursor - first : end - first],
                        local,
                        local + end - cursor,
                    )
                    cursor = end
            return result

        # A full-scan storage chunk must be audited once, not once per scan row.
        # Oversized or contiguous storage keeps the existing bounded frame path.
        chunk_audit = np.dtype(info.dtype) == np.dtype("float64") and all(
            data.chunks is not None
            and math.prod(data.chunks) * 24 <= _array_resident.MAX_INGEST_BYTES
            for data in datasets
        )

        def audit_blocks():
            for data in datasets:
                for selection in data.iter_chunks():
                    yield data[selection]

        loaded = load_array_resident(
            shape,
            info.dtype,
            read_frames,
            info.metadata,
            backend=backend,
            device=device,
            pixel_mask=info.pixel_mask,
            hot_pixel_correction=hot_pixel_correction,
            verbose=verbose,
            auto_narrow=auto_narrow,
            float_audit_blocks=audit_blocks if chunk_audit else None,
        )
        try:
            for source, signature in signatures.items():
                status = source.stat()
                if (status.st_size, status.st_mtime_ns) != signature:
                    raise ValueError(
                        "HDF5 source changed during loading; reopen the acquisition."
                    )
        except BaseException:
            loaded.close()
            raise
        loaded.metadata["load_timings"]["storage_reader"] = "bounded-hdf5"
        return loaded
