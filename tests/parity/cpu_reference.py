"""CPU reference decode and encode of stored acquisitions, the oracle the GPU loads are tested against.

Parity tests compare every accelerated load against an independent decode on
the host: h5py plus hdf5plugin for HDF5 (its bitshuffle+LZ4 filter
decompresses each dataset slice with no custom kernel), the portable QEM
reference codec for ``.qem`` files, and plain memory maps for NumPy, EMPAD,
Zarr and DigitalMicrograph sources. ``quantem.gpu`` itself never loads or
saves scientific data on the CPU, so this oracle lives with the tests. It is
slower than CUDA or MPS but returns bit-identical stored values.
"""

import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from quantem.gpu.formats import empad, zarr_zip
from quantem.gpu.formats.hdf5.frames import detector_sources
from quantem.gpu.formats.hdf5.master import HOT_PIXEL_CORRECTION_ATTRIBUTE, get_metadata, read_pixel_mask
from quantem.gpu.formats.qem.metadata import MAPED_RECORDS, json_metadata
from quantem.gpu.formats.qem.reference import load_array, save_array
from quantem.gpu.formats.qem.snapshot import is_qem_file
from quantem.gpu.io.dataset import Dataset4dstem, drop_unread_calibration
from quantem.gpu.io.digitalmicrograph import read_dm_source
from quantem.gpu.io.hdf5.write import _DEFAULT_BATCH_SIZE, _infer_save_dtype, save_compressed_arina_h5
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.resident.hot_pixels import (
    fill_flagged,
    hot_pixel_outcome,
    hot_pixel_record,
    median_plan,
    normalize_hot_pixel_correction,
)


def load(
    source,
    *,
    scan_shape: tuple[int, int] | None = None,
    apply_mask: bool | None = None,
    hot_pixel_correction: str = "median",
    auto_narrow: bool = False,
    stack: bool | None = None,
    verbose: bool = False,
) -> Dataset4dstem | list[Dataset4dstem]:
    """Decode one source (or a list) into host NumPy arrays, the stored values the GPU loads must reproduce.

    The format follows the file as ``io.load`` detects it. HDF5 masters read
    the stored detector mask unless ``apply_mask=False`` and treat its pixels
    with ``hot_pixel_correction`` through the CPU reference of the GPU kernels,
    so a reference load holds the counts an encoded load does; several masters
    stack along a leading file axis unless ``stack`` is False. ``.qem`` copies
    decode through the portable reference codec; NumPy, EMPAD and Zarr sources
    and DigitalMicrograph files are memory-mapped as stored.
    """
    hot_pixel_correction = normalize_hot_pixel_correction(hot_pixel_correction)
    single = isinstance(source, (str, os.PathLike))
    sources = [source] if single else list(source)
    if not sources:
        raise ValueError("Empty file list")
    if all(is_qem_file(path) for path in sources):
        loaded = []
        for path in sources:
            data, metadata = load_array(path)
            if scan_shape is not None and tuple(scan_shape) != data.shape[:2]:
                raise ValueError("scan_shape disagrees with the saved QEM file.")
            loaded.append(record_dense_representation(Dataset4dstem(data, metadata)))
        return loaded[0] if single else loaded
    suffixes = {Path(path).suffix.lower() for path in sources}
    if suffixes <= {".npy", ".xml", ".raw"} or all(zarr_zip.is_zarr_zip(path) for path in sources):
        loaded = []
        for path in sources:
            reader = zarr_zip if zarr_zip.is_zarr_zip(path) else empad
            loaded.append(Dataset4dstem(*reader.load_array_source(path, scan_shape)))
        return loaded[0] if single else loaded
    if suffixes <= {".dm3", ".dm4"}:
        loaded = []
        for path in sources:
            dm_source = read_dm_source(path, scan_shape)
            metadata = dict(dm_source.metadata)
            metadata.update(backend="cpu", residency="host", file_counts_exact=True,
                            lossless_exact=True, scan_bin=1, detector_bin=1, crop=None)
            loaded.append(Dataset4dstem(dm_source.memmap(), metadata))
        return loaded[0] if single else loaded
    return drop_unread_calibration(load_reference(
        sources,
        single=single,
        scan_shape=scan_shape,
        apply_mask=True if apply_mask is None else apply_mask,
        hot_pixel_correction=hot_pixel_correction,
        auto_narrow=auto_narrow,
        # Unlike compressed acquisitions, reference arrays stack by default.
        stack=True if stack is None else stack,
        verbose=verbose,
    ), sources)


def save(
    path,
    data,
    *,
    metadata: dict | None = None,
    dtype=None,
    scan_shape: tuple[int, int] | None = None,
    batch_size: int | None = None,
    frames_per_file: int = 32768,
) -> None:
    """Write host data as a ``.qem`` copy (portable reference encoder) or an Arina HDF5 master (h5py, hdf5plugin).

    Tests build their fixture files with it; the files are the ones the GPU
    writers produce, so a GPU load of them is checked against known values.
    A dense :class:`Dataset4dstem` keeps its metadata, as ``io.save`` keeps it.
    """
    if Path(path).suffix.lower() == ".qem":
        if isinstance(data, Dataset4dstem):
            metadata = dict(data.metadata) | (metadata or {})
            data = data.data
        save_array(path, np.asarray(data), metadata, chunk_scans=512 if batch_size is None else batch_size)
        return
    if isinstance(data, Dataset4dstem):
        metadata = dict(data.metadata if metadata is None else metadata)
        for key, attribute in MAPED_RECORDS.items():
            if isinstance(metadata.get(key), dict):
                metadata[attribute] = json.dumps(metadata[key], default=json_metadata)
        data = data.data
    save_compressed_arina_h5(
        path,
        data,
        scan_shape=scan_shape,
        metadata=metadata,
        dtype=_infer_save_dtype(data) if dtype is None else dtype,
        batch_size=_DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size),
        frames_per_file=frames_per_file,
        compression_backend="hdf5",
    )


# ---

def load_reference(
    paths: list[str | os.PathLike[str]],
    *,
    single: bool,
    scan_shape: tuple[int, int] | None,
    apply_mask: bool,
    hot_pixel_correction: str,
    auto_narrow: bool,
    stack: bool,
    verbose: bool,
) -> Dataset4dstem | list[Dataset4dstem]:
    """Decode complete HDF5 acquisitions on the CPU.

    With ``apply_mask`` the pixels the stored mask flags receive
    ``hot_pixel_correction`` through the CPU reference of the GPU kernels, so a
    reference load stores the counts an encoded load does; uint32 counts
    narrow to uint16 only when every count fits, and the returned NumPy array
    keeps the stored detector geometry. Several paths stack along a leading
    file axis unless ``stack`` is False; ``single`` returns one result for one
    path given on its own.
    """
    import hdf5plugin  # noqa: F401 - registers the bitshuffle+LZ4 filter; loaded on use only

    def load_one(path):
        meta = get_metadata(str(path))
        meta["source_path"] = str(path)
        mask = read_pixel_mask(str(path)) if apply_mask else None
        saved_correction = meta.get(HOT_PIXEL_CORRECTION_ATTRIBUTE)
        correction = hot_pixel_record(
            mask, hot_pixel_correction if apply_mask and saved_correction is None else "none", backend="cpu",
        )
        data = load_master(str(path), pixel_mask=mask if correction["applied"] else None,
                           method=correction["method"], verbose=verbose)
        source_dtype = np.dtype(data.dtype)
        if saved_correction is not None:
            correction = saved_correction
        elif correction["applied"]:
            correction["stage"] = "cpu_load_before_dense_storage"
        meta["hot_pixel_correction"] = correction
        if mask is not None:
            meta["pixel_mask"] = mask
        if auto_narrow and data.dtype == np.uint32 and int(data.max()) < 65536:
            # Every count fits, so uint16 halves the array without changing a value.
            data = data.astype(np.uint16)
        # Frames are stored in row-major scan order; the explicit scan_shape wins over the master's.
        shape = scan_shape if scan_shape is not None else meta.get("scan_shape")
        if shape is not None and data.ndim == 3:
            if math.prod(shape) != data.shape[0]:
                raise ValueError(f"scan_shape {shape} incompatible with frame count {data.shape[0]}")
            data = data.reshape(*shape, *data.shape[-2:])
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
        meta["scan_order"] = "row-major"
        return data, meta

    started = time.perf_counter()
    if single or not stack:
        results = []
        for path in paths:
            data, meta = load_one(path)
            if verbose:
                print(f"  Loaded {tuple(data.shape)} ({data.nbytes / 1e9:.1f} GB) in "
                      f"{time.perf_counter() - started:.2f}s (cpu backend). "
                      f"{hot_pixel_outcome(meta['hot_pixel_correction'])}.")
            results.append(record_dense_representation(Dataset4dstem(data, meta)))
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
    return record_dense_representation(Dataset4dstem(out, meta))


def record_dense_representation(result: Dataset4dstem) -> Dataset4dstem:
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
    return Dataset4dstem(data, metadata)


def load_master(
    filepath: str,
    *,
    pixel_mask: np.ndarray | None = None,
    method: str = "median",
    verbose: bool = True,
) -> np.ndarray:
    """Decompress a master's detector datasets to one ``(n_frames, det_row, det_col)`` array.

    Frames keep the stored dtype; :func:`load_reference` owns the scan-shape
    unflatten, narrowing and metadata. ``pixel_mask`` (the raw detector mask,
    nonzero = dead) marks pixels whose stored value is an Arina sentinel such
    as 65535 rather than a count; ``method`` (``"median"`` or ``"zero"``)
    replaces them as ``quantem.gpu.resident.hot_pixels.fill_flagged`` does. A
    mask of another shape raises ``ValueError`` rather than being skipped.
    """
    import hdf5plugin  # noqa: F401 - registers the bitshuffle+LZ4 filter; loaded on use only
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

    plan = None if dead_pixels is None else median_plan(dead_pixels)
    if plan is not None and method == "median" and np.dtype(dtype).name not in ("uint8", "uint16", "uint32"):
        raise ValueError(
            f"{filepath}: a median fill of flagged pixels needs uint8/uint16/uint32 counts, not {np.dtype(dtype)}; "
            "load with hot_pixel_correction='none' to keep the stored values."
        )
    offset = 0
    source_iter = sources
    if verbose and len(sources) > 1:
        source_iter = tqdm(sources, desc="chunks", leave=False)
    for source in source_iter:
        with h5py.File(source.path, "r") as handle:
            raw = handle[source.dataset_path][:]  # hdf5plugin decompresses here
        if plan is not None:
            raw = fill_flagged(np.ascontiguousarray(raw), plan, method)
        frame_count = raw.shape[0]
        output[offset : offset + frame_count] = raw
        offset += frame_count
    return output
