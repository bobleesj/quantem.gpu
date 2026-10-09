"""CPU reference decode of complete HDF5 acquisitions.

Parity tests compare every accelerated load against an independent decode:
h5py plus hdf5plugin, whose bitshuffle+LZ4 filter decompresses each dataset
slice on the CPU with no custom kernel. ``io.load`` reaches it only with an
explicit ``backend="cpu"``; accelerated ``backend="auto"`` never selects it.
It is slower than CUDA or MPS but returns bit-identical stored frames.
"""

import os
import time

import h5py
import hdf5plugin  # noqa: F401 - registers the bitshuffle+LZ4 filter
import numpy as np
from tqdm.auto import tqdm

from quantem.gpu.formats.hdf5.frames import detector_sources
from quantem.gpu.formats.hdf5.master import get_metadata, read_pixel_mask
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.io.selection import _apply_scan_shape, _normalize_scan_order


def load_reference(
    paths: list[str | os.PathLike[str]],
    *,
    single: bool,
    scan_shape: tuple[int, int] | None,
    scan_order: str,
    apply_mask: bool,
    auto_narrow: bool,
    stack: bool,
    verbose: bool,
) -> Dataset4dstemGPU | list[Dataset4dstemGPU]:
    """Decode complete HDF5 acquisitions on the CPU as the explicit reference.

    Stored dead pixels are zeroed when ``apply_mask`` is set, uint32 counts
    narrow to uint16 only when every count fits, and the returned NumPy array
    keeps the stored detector geometry. Several paths stack along a leading
    file axis unless ``stack`` is False; ``single`` returns one result for one
    path given on its own.
    """
    scan_order = _normalize_scan_order(scan_order)

    def load_one(path):
        meta = get_metadata(str(path))
        mask = read_pixel_mask(str(path)) if apply_mask else None
        data = load_master(str(path), pixel_mask=mask, verbose=verbose)
        source_dtype = np.dtype(data.dtype)
        if mask is not None:
            meta["pixel_mask"] = mask
        if auto_narrow and data.dtype == np.uint32 and int(data.max()) < 65536:
            # Every count fits, so uint16 halves the array without changing a value.
            data = data.astype(np.uint16)
        data = _apply_scan_shape(data, scan_shape, meta, scan_order)
        # Report the same detector geometry and dtype keys as an encoded load,
        # so consumers never branch on the backend.
        detector_shape = tuple(int(value) for value in data.shape[-2:])
        source_detector_shape = next(
            (tuple(int(value) for value in meta[key])
             for key in ("source_detector_shape", "raw_detector_shape", "detector_shape")
             if meta.get(key) is not None),
            detector_shape,
        )
        meta["source_detector_shape"] = source_detector_shape
        meta.setdefault("raw_detector_shape", source_detector_shape)
        meta["detector_shape"] = detector_shape
        meta["det_bin"] = 1
        meta["source_dtype"] = str(np.dtype(meta.get("source_dtype", source_dtype)))
        meta["dtype"] = str(np.dtype(data.dtype))
        meta["scan_order"] = scan_order
        return data, meta

    started = time.perf_counter()
    if single or not stack:
        results = []
        for path in paths:
            data, meta = load_one(path)
            if verbose:
                print(f"  Loaded {tuple(data.shape)} ({data.nbytes / 1e9:.1f} GB) in "
                      f"{time.perf_counter() - started:.2f}s (cpu backend)")
            results.append(record_dense_representation(Dataset4dstemGPU(data, meta)))
        return results[0] if single else results
    first, meta = load_one(paths[0])
    out = np.empty((len(paths), *first.shape), dtype=first.dtype)
    out[0] = first
    for index, path in enumerate(paths[1:], start=1):
        out[index] = load_one(path)[0]
    meta["n_files"] = len(paths)
    if verbose:
        print(f"  Loaded {len(paths)} files {out.shape} ({out.nbytes / 1e9:.1f} GB) in "
              f"{time.perf_counter() - started:.2f}s (cpu backend)")
    return record_dense_representation(Dataset4dstemGPU(out, meta))


def record_dense_representation(result: Dataset4dstemGPU) -> Dataset4dstemGPU:
    """Attach the common representation fields to a CPU reference result.

    Encoded loaders record these fields themselves; the reference array reports
    its shape and bytes the same way so every result describes its memory. It
    is lossless only while it keeps the stored dtype (auto-narrowed uint32
    counts are not claimed).
    """
    metadata = dict(result.metadata)
    data = result.data
    dtype = np.dtype(data.dtype).name
    metadata.setdefault("representation", DataRepresentation.DENSE.value)
    metadata.setdefault("residency", "host")
    metadata.setdefault("working_shape", tuple(int(value) for value in data.shape))
    metadata.setdefault("working_dtype", dtype)
    metadata.setdefault("working_logical_tensor_bytes", int(data.nbytes))
    metadata.setdefault("physical_resident_bytes", int(data.nbytes))
    source_dtype = metadata.get("source_dtype")
    metadata.setdefault(
        "lossless_exact",
        source_dtype is not None and np.dtype(source_dtype) == np.dtype(dtype),
    )
    return Dataset4dstemGPU(data, metadata)


def load_master(
    filepath: str,
    *,
    pixel_mask: np.ndarray | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Decompress a master's detector datasets to one ``(n_frames, det_row, det_col)`` array.

    Frames keep the stored dtype; :func:`load_reference` owns the scan-shape
    unflatten, narrowing and metadata. ``pixel_mask`` (the raw detector mask,
    nonzero = dead) zeroes dead pixels, whose stored value is an Arina sentinel
    such as 65535 rather than a count. A mask of another shape raises
    ``ValueError`` rather than being skipped.
    """
    with h5py.File(filepath, "r") as master:
        sources = detector_sources(master)
    if not sources:
        raise ValueError(f"{filepath}: no entry/data/data dataset or data_NNNNNN chunks")
    for source in sources:
        if not os.path.exists(source.path):
            raise FileNotFoundError(f"Missing chunk file: {os.path.basename(source.path)}")
    datasets = []
    for source in sources:
        with h5py.File(source.path, "r") as handle:
            dataset = handle[source.dataset_path]
            datasets.append((dataset.shape, dataset.dtype))
    (_, det_row, det_col), dtype = datasets[0]
    output = np.empty((sum(shape[0] for shape, _ in datasets), det_row, det_col), dtype=dtype)

    dead_pixels = None
    if pixel_mask is not None:
        dead_pixels = np.asarray(pixel_mask) != 0
        if dead_pixels.shape != (det_row, det_col):
            # Skipping the mask would keep dead-pixel sentinels as counts without saying so.
            raise ValueError(
                f"{filepath}: the pixel mask is {dead_pixels.shape[0]} x {dead_pixels.shape[1]}, but the frames are "
                f"{det_row} x {det_col}; load with apply_mask=False to read the stored values without it."
            )

    offset = 0
    source_iter = sources
    if verbose and len(sources) > 1:
        source_iter = tqdm(sources, desc="chunks", leave=False)
    for source in source_iter:
        with h5py.File(source.path, "r") as handle:
            raw = handle[source.dataset_path][:]  # hdf5plugin decompresses here
        if dead_pixels is not None:
            raw[:, dead_pixels] = 0
        frame_count = raw.shape[0]
        output[offset : offset + frame_count] = raw
        offset += frame_count
    return output
