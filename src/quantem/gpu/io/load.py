"""
GPU-accelerated HDF5 loading for 4D-STEM diffraction data.

The public load verb coordinates scientific selection and backend dispatch.
Result models, metadata, and host-staging ownership have
separate private modules. Performance depends on the source and device;
qualified measurements live in the benchmark registry.

Examples
--------
>>> from quantem.gpu.io import load
>>> with load("acquisition.qem") as acquisition:
...     pattern = acquisition[0, 0]
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from quantem.gpu.device import select
from quantem.gpu.formats import empad
from quantem.gpu.formats.float8 import FLOAT8_DTYPES, is_float8
from quantem.gpu.formats.qem.snapshot import is_qem_file
from quantem.gpu.io import arrays, digitalmicrograph, encoded, paired, qem
from quantem.gpu.io.dataset import Dataset4dstemGPU, drop_unread_calibration
from quantem.gpu.io.hdf5 import cpu
from quantem.gpu.io.precision import load_precision, precision_name, saved_precision
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.resident.hot_pixels import (
    hot_pixel_record,
    normalize_hot_pixel_correction,
)

def load(
    source: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | np.ndarray | torch.Tensor,
    *,
    dtype: str | type | np.dtype | None = None,
    backend: str = "auto",
    representation: DataRepresentation | str | None = None,
    dataset_path: str | None = None,
    scan_shape: tuple[int, int] | None = None,
    scan_region: tuple[int, int, int, int] | None = None,
    detector_region: tuple[int, int, int, int] | None = None,
    apply_mask: bool | None = None,
    hot_pixel_correction: str = "none",
    auto_narrow: bool = False,
    stack: bool | None = None,
    device: int | str | None = None,
    verbose: bool = True,
) -> Dataset4dstemGPU | list[Dataset4dstemGPU]:
    """Load one or more 4D-STEM sources through an accelerated backend.

    Supported original acquisitions default to bounded ANS ingestion
    on CUDA and MPS. Real numeric sources retain their 8/16/32/64-bit integer
    or 16/32/64-bit floating dtype and original bits at native detector geometry. Use ``scan_shape`` for
    headerless EMPAD RAW. CPU reference access is explicit, never a fallback.
    Generic HDF5 layouts use bounded storage reads where the direct compressed
    chunk decoder cannot apply. Saved copies retain calibration and provenance.

    Original FP8 NumPy (``ml_dtypes``) and Torch arrays are accepted directly.
    HDF5 byte arrays must name their FP8 variant with the dataset attribute
    ``quantem_dtype``. No FP8 quantization or external scale inference occurs.
    DP reads retain the original FP8 dtype; detector and SSB calculations use
    float32 on the selected GPU. See the IO guide for supported variants.

    Fractional intensity exports support explicit ``dtype="scaled_uint16"``
    on CUDA and Metal/MPS. They remain ANS encoded and print a measured
    conversion report. Omit dtype to preserve the original float32 bits.
    Scaled codes restore their saved intensity units for detector queries. ``scan_region`` and
    ``detector_region`` select values before resident allocation. New scaled
    storage automatically calibrates bounded regions in one pass; saved files
    retain their recorded calibration.
    With explicit precision, ``source`` can also be a GPU array or an object
    providing ``shape``, ``dtype``, and ``blocks()`` of ordered GPU frames.
    Each generated frame must appear once in row-major order.
    ``io.load("display_master.h5", dtype="scaled_uint16")`` is approximate;
    preserve the original float32 file for exact scientific analysis.

    Here ``dtype`` selects stored precision, not calculation precision or a
    compression codec. ``"scaled_uint16"`` stores calibrated integer codes
    and reconstructs float32 intensities on the GPU; it does not recover discarded
    precision. Plain ``"uint16"`` is not calibrated scaled storage.
    Omit ``dtype`` when reopening a precision file to retain its recorded
    values and calibration.

    Complete native HDF5 acquisitions can be loaded together into compact
    encoded accelerator storage with ``stack=False``. Encoded is the default
    on CUDA and MPS. Each result remains an independent encoded owner, not a
    dense stacked array. Stored detector-mask values are preserved by default;
    request GPU median or zero replacement explicitly when needed.

    All spatial arguments use ``(row, col)`` order. ``representation`` selects
    how the complete logical data is retained. Ordinary HDF5 selects
    ``"encoded"`` automatically on CUDA and MPS. Dense GPU overrides
    fail before allocation. Use ``loaded.read(scan_region=...)`` for bounded
    tensor access.

    Self-contained ANS files default to ``representation="encoded"`` and retain
    stored native counts. A list of compressed or independently calibrated
    files returns one resident acquisition per file when ``stack`` is omitted.
    Preparing that list exposes one detector session without materializing a
    dense stack. Explicit ``stack=True`` is rejected for these sources.
    ``representation="paired"`` streams complete uint16
    acquisitions (one path or a list, each returned as its own source) into
    the CUDA paired-count tANS resident layout, and a saved paired resident
    form reopens under the same name without decoding. CPU reference expansion
    requires ``backend="cpu", representation="dense"``. Unsupported conversions
    raise instead of silently loading HDF5, expanding densely, or using CPU.
    ``apply_mask=None`` zeroes stored dead pixels in the CPU reference.
    Compact HDF5 loads use ``hot_pixel_correction``; saved compact sources
    retain the correction already recorded in their metadata.
    File format and compression are detected from contents, independently of
    resident representation.
    ``backend="auto"`` selects CUDA or MPS and never selects CPU silently.

    The default ``hot_pixel_correction="none"`` preserves every original
    measurement and its dtype, including flagged-pixel markers. Scientific
    detector products exclude recorded invalid pixels without changing storage.
    Explicit ``"median"`` or ``"zero"`` correction changes flagged counts on
    GPU and records that change in the saved copy. ``auto_narrow`` applies only
    to an explicitly selected dense CPU reader; encoded loading retains dtype,
    except that on MPS a bitshuffle master whose uint32 values all fit 16 bits
    is held as uint16 counts that read back unchanged (one printed line).

    Parameters
    ----------
    source
        One master/data HDF5 path, a folder, a list of master paths, or a
        standalone QuantEM encoded file.
    dtype
        Omit (or ``"native"``) to keep stored values. ``"scaled_uint16"``
        selects calibrated precision storage for float32 sources.
    representation
        ``"dense"``, ``"encoded"`` or ``"paired"``. The authenticated storage
        schema selects the exact decoder within each representation.
        When omitted, ordinary HDF5 uses encoded CUDA/MPS storage; saved
        ANS sources retain their recorded representation. GPU dense
        overrides are rejected before loading; use bounded ``read`` selections
        from the encoded acquisition. Unsupported
        source/representation/backend combinations raise rather than transform
        implicitly. Representation never changes scan coverage,
        detector coverage, binning, calibration, or scientific dtype.
    backend
        ``"auto"``, ``"cuda"``, ``"mps"``, or explicit reference ``"cpu"``.
    device
        Omit to let ``backend`` choose. ``"cpu"`` loads the dense CPU
        reference (the same as ``backend="cpu"``), ``"mps"`` the Apple GPU,
        ``"cuda"`` or ``"cuda:N"`` (or the integer ``N``) a CUDA GPU.
        ``"auto"`` picks CUDA, then MPS, then CPU, and prints which; the
        meaning is the same as ``quantem.gpu.device.resolve_device``.
    stack
        Omit to retain compressed or independently calibrated acquisitions as
        a list, while stacking CPU reference arrays. Use ``False``
        to request independent acquisitions explicitly.
    scan_shape
        Optional full scan shape as ``(row, col)``.
    scan_region, detector_region
        Optional precision-storage bounds as
        ``(row_start, row_stop, col_start, col_stop)``.

    Returns
    -------
    Dataset4dstemGPU or list[Dataset4dstemGPU]
        Data stays backend-resident.
    """
    backend, device = _device_backend(backend, device)
    # Residency policy does not depend on whether a device is available.
    if (
        representation is not None
        and DataRepresentation.parse(representation) is DataRepresentation.DENSE
        and backend in (None, "auto", "cuda", "mps")
    ):
        raise NotImplementedError(
            "GPU acquisitions must remain ANS encoded; omit representation "
            "or use representation='encoded'. Read bounded regions from the "
            "loaded acquisition instead of expanding the full cube. "
            "Tiny reference arrays require backend='cpu'."
        )
    hot_pixel_correction = normalize_hot_pixel_correction(hot_pixel_correction)
    # One path, array or generated block source is one acquisition; a list or
    # tuple names several, each returned as its own result.
    single = isinstance(source, (str, os.PathLike)) or not isinstance(source, Sequence)
    sources = [source] if single else list(source)
    precision = precision_name(dtype)

    input_dtype = getattr(source, "dtype", "")
    if single and str(input_dtype).removeprefix("torch.").startswith("float8_") and not is_float8(input_dtype):
        raise NotImplementedError(
            f"Unsupported original FP8 dtype {input_dtype}; supported variants are "
            + ", ".join(FLOAT8_DTYPES) + ". No conversion or upload occurred."
        )

    if single and is_float8(getattr(source, "dtype", "")):
        if (len(source.shape) != 4 or min(source.shape) <= 0
                or any(value is not None for value in (dataset_path, scan_shape, scan_region, detector_region))
                or dtype not in (None, "native") or apply_mask
                or hot_pixel_correction != "none"):
            raise ValueError("Load the original 4D FP8 array without conversion or selection options; use read() for a region.")
        if representation not in (None, "encoded"):
            raise ValueError("FP8 acquisitions require encoded residency; omit representation.")
        selected_backend = select.resolve_backend(backend)
        return arrays.load_array_resident(
            tuple(source.shape), source.dtype,
            lambda first, stop: arrays.read_frame_block(source, source.shape, first, stop),
            {"source_kind": "array", "source_dtype": str(source.dtype).removeprefix("torch.")},
            backend=selected_backend, device=device, verbose=verbose,
        )

    # NumPy and EMPAD arrays: complete original measurements, encoded on the GPU.
    array_sources = [isinstance(path, (str, os.PathLike))
                     and Path(path).suffix.lower() in {".npy", ".xml", ".raw"}
                     for path in sources]
    if any(array_sources) and not precision:
        array_backend = select.resolve_backend(backend)
        allowed = (None, "dense") if array_backend == "cpu" else (None, "encoded")
        if not all(array_sources) or representation not in allowed:
            raise NotImplementedError(
                "Load NumPy/EMPAD separately with encoded GPU residency; "
                "CPU dense access is explicit reference-only."
            )
        if (any(value is not None for value in (dataset_path, scan_region, detector_region))
                or dtype not in (None, "native") or apply_mask):
            raise ValueError(
                "Original-array loading preserves complete measurements; remove "
                "selection and dtype options, then use read() for a region."
            )
        if len(sources) > 1 and stack:
            raise ValueError("Independently calibrated acquisitions cannot be stacked; omit stack to receive one per file.")
        results = []
        try:
            for path in sources:
                data, metadata = empad.load_array_source(path, scan_shape)
                if array_backend == "cpu":
                    results.append(Dataset4dstemGPU(data, metadata))
                else:
                    try:
                        source_files = {Path(path), Path(data.filename)}
                        if metadata.get("source_metadata_path"):
                            source_files.add(Path(metadata["source_metadata_path"]))
                        signatures = {
                            file: (file.stat().st_size, file.stat().st_mtime_ns)
                            for file in source_files
                        }
                        results.append(arrays.load_array_resident(
                            data.shape,
                            data.dtype,
                            lambda first, stop, data=data: arrays.read_frame_block(
                                data, data.shape, first, stop
                            ),
                            metadata,
                            backend=array_backend,
                            device=device,
                            hot_pixel_correction=hot_pixel_correction,
                            verbose=verbose,
                        ))
                        for file, signature in signatures.items():
                            status = file.stat()
                            if (status.st_size, status.st_mtime_ns) != signature:
                                raise ValueError(
                                    "Source changed during ANS ingestion; "
                                    "reopen the acquisition."
                                )
                    finally:
                        # Array sources are NumPy memory maps; close the file
                        # mapping now rather than when garbage is collected.
                        mapping = data._mmap
                        if mapping is not None:
                            mapping.close()
        except BaseException:
            for result in results:
                result.close()
            raise
        return results[0] if single else results

    # DigitalMicrograph acquisitions and saved .qem copies.
    snapshots = [is_qem_file(path) for path in sources]
    dm_sources = [isinstance(path, (str, os.PathLike))
                  and Path(path).suffix.lower() in {".dm3", ".dm4"}
                  for path in sources]
    if any(dm_sources) or any(snapshots):
        if not all(dm_sources) and not all(snapshots):
            raise ValueError("Load DigitalMicrograph acquisitions separately from other source formats.")
        if any(value is not None for value in (dataset_path, scan_region, detector_region)) or apply_mask:
            raise NotImplementedError(
                "DM loading preserves the complete acquisition; remove selection "
                "and masking options."
            )
        if dtype not in (None, "native"):
            raise ValueError("DM loading preserves stored counts; omit dtype for lossless loading.")
        if len(sources) > 1 and stack:
            raise ValueError("DM acquisitions cannot be stacked; omit stack to receive one resident acquisition per file.")
        loader = qem.load_streamed if all(snapshots) else digitalmicrograph.load_dm
        loaded = []
        try:
            for path in sources:
                loaded.append(loader(path, backend=backend, representation=representation,
                                      scan_shape=scan_shape, device=device, verbose=verbose))
                if all(dm_sources):
                    # DM has no stored detector-validity mask. Record the policy
                    # without pretending that unflagged measurements were changed.
                    loaded[-1].metadata["hot_pixel_correction"] = hot_pixel_record(
                        None, hot_pixel_correction,
                        backend=loaded[-1].metadata["backend"],
                    )
        except BaseException:
            for item in loaded:
                item.close()
            raise
        return loaded[0] if single else loaded

    # Scaled precision: explicit conversion, or a saved precision export.
    saved = [saved_precision(path) for path in sources if isinstance(path, (str, os.PathLike)) and Path(path).is_file()]
    if precision or any(saved):
        if any(saved) and dtype not in (None, "native") and precision is None:
            raise ValueError("Saved precision includes intensity scaling. Omit dtype to restore its units, or request scaled_uint16 explicitly; raw-code casts are not supported.")
        precision_backend = select.resolve_backend(backend)
        if precision_backend not in {"cuda", "mps"}:
            raise NotImplementedError("Precision loading requires CUDA or Metal; no CPU conversion is used.")
        storage_types = {precision} if precision else {item["storage"] for item in saved if item}
        if storage_types != {"scaled_uint16"}:
            raise NotImplementedError(
                "Only dtype='scaled_uint16' is a stored precision. Omit dtype to keep "
                "the original float32 values lossless, or request dtype='scaled_uint16'."
            )
        if representation is not None and DataRepresentation.parse(representation) is not DataRepresentation.ENCODED:
            raise ValueError(
                "This precision uses representation='encoded'; "
                "omit representation to select its default storage."
            )
        if apply_mask:
            raise NotImplementedError("Precision loading supports scan_region and detector_region; remove the masking control.")
        loaded = []
        try:
            for path in sources:
                loaded.append(load_precision(path, dtype=dtype, device=device,
                    scan_shape=scan_shape, dataset_path=dataset_path,
                    scan_region=scan_region, detector_region=detector_region, verbose=verbose,
                    backend=precision_backend))
        except BaseException:
            for item in loaded:
                item.close()
            raise
        return drop_unread_calibration(loaded[0] if single else loaded, sources)

    # Ordinary HDF5: encoded on CUDA and MPS by default, paired on request.
    if representation is None and sources and all(
        DataRepresentation.detect_source(path) is DataRepresentation.DENSE
        for path in sources
    ):
        resolved_backend = select.resolve_backend(backend)
        representation = (
            DataRepresentation.ENCODED
            if resolved_backend in {"cuda", "mps"}
            else DataRepresentation.DENSE
        )
    selected_representation = _selected_representation(sources, representation)
    if selected_representation is DataRepresentation.PAIRED:
        if select.resolve_backend(backend) != "cuda":
            raise NotImplementedError("The paired resident layout requires backend='cuda'.")
        if any(value is not None for value in (dataset_path, scan_region, detector_region)):
            raise ValueError("The paired layout preserves complete native acquisitions; remove selection options.")
        if dtype not in {None, "native"} or apply_mask:
            raise ValueError("The paired layout preserves raw native counts; use dtype='native' and apply_mask=False.")
        saved = [DataRepresentation.detect_source(path) is DataRepresentation.PAIRED for path in sources]
        if all(saved):
            loaded = [paired.load_paired_file(path, device=device, verbose=verbose) for path in sources]
        elif any(saved):
            raise ValueError("Load saved paired resident forms and original HDF5 acquisitions in separate calls.")
        else:
            loaded = paired.load_h5_paired(sources, scan_shape=scan_shape, device=device,
                                           verbose=verbose, hot_pixel_correction=hot_pixel_correction)
        return drop_unread_calibration(loaded[0] if single else loaded, sources)
    if selected_representation is DataRepresentation.ENCODED:
        ans_backend = select.resolve_backend(backend)
        if ans_backend not in {"cuda", "mps"}:
            raise NotImplementedError("H5-to-ANS loading requires CUDA or MPS.")
        if scan_region is not None or detector_region is not None:
            raise ValueError("H5-to-ANS preserves complete native acquisitions; remove selection options, then use read() for a region.")
        if len(sources) > 1 and stack:
            raise ValueError(
                "Several ANS acquisitions cannot be stacked into one array; omit stack "
                "to receive one resident acquisition per file."
            )
        if dtype not in {None, "native"} or apply_mask:
            raise ValueError("H5-to-ANS preserves raw native counts; use dtype='native' and apply_mask=False.")
        if ans_backend == "mps" and device not in (None, "mps"):
            raise ValueError("Metal ANS runs on the one Apple GPU; use device='mps' or omit device.")
        results = []
        try:
            for path in sources:
                results.append(
                    encoded.load_h5_ans(
                        path,
                        scan_shape=scan_shape,
                        dataset_path=dataset_path,
                        device=None if ans_backend == "mps" else device,
                        verbose=verbose,
                        backend=ans_backend,
                        hot_pixel_correction=hot_pixel_correction,
                    )
                )
        except BaseException:
            for result in results:
                result.close()
            raise
        return drop_unread_calibration(results[0] if single else results, sources)

    # The explicit CPU reference.
    if scan_region is not None or detector_region is not None:
        raise RuntimeError(
            "scan_region= and detector_region= select bounded precision storage on "
            "CUDA and MPS; the CPU reference loads complete acquisitions."
        )
    if dataset_path is not None or device is not None:
        raise ValueError(
            "The CPU reference decodes Arina master chunks on the host; "
            "remove dataset_path and device."
        )
    if dtype not in (None, "native"):
        raise ValueError("The CPU reference returns stored counts; omit dtype.")
    if not sources:
        raise ValueError("Empty file list")
    return drop_unread_calibration(cpu.load_reference(
        sources,
        single=single,
        scan_shape=scan_shape,
        apply_mask=True if apply_mask is None else apply_mask,
        auto_narrow=auto_narrow,
        # Unlike compressed acquisitions, CPU reference arrays stack by default.
        stack=True if stack is None else stack,
        verbose=verbose,
    ), sources)


def bin(data, factor: int = 2, axes: str = "detector", dtype=None, reduction: str = "sum", edge: str = "crop"):
    """Sum (or average) ``factor`` x ``factor`` blocks of a CuPy array or Torch tensor on its GPU; the same library out.

    A binned pixel is the sum of the ``factor**2`` pixels it covers, out[i, j] = sum_{a, b < factor} data[f i + a, f j + b]
    (divided by ``factor**2`` for ``reduction="mean"``): summing a detector b x b keeps every electron, which is why a
    reconstruction reads a binned crop instead of an interpolated one. ``io.load`` refuses ``detector_bin`` for HDF5 and
    .qem sources, so a caller bins as part of a read, ``bin(dataset.read(scan_region=...), factor=4)``.

    ``data`` is 4D ``(scan_row, scan_col, det_row, det_col)``, 3D ``(frame, det_row, det_col)`` or 2D ``(det_row,
    det_col)``. ``axes`` is ``"detector"``, ``"scan"`` (4D only) or ``"all"`` (4D only). ``edge="crop"`` drops the
    trailing pixels that do not fill a block; ``edge="partial"`` keeps them by zero padding before an exact sum (a
    partial mean would divide by the wrong count, so it is refused). Integer data sum exactly in a 64-bit integer (uint64 or
    int64 by the input's sign for CuPy, int64 for Torch), anything else in float32, unless ``dtype`` is given; the sum
    accumulates in that dtype.

    Examples
    --------
    >>> from quantem.gpu.io.load import bin
    >>> crop = bin(dataset.read(scan_region=(0, 64, 0, 64)), factor=4, axes="detector")
    """
    import torch

    from quantem.gpu.device.cuda_runtime import cp

    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
    if edge not in ("crop", "partial"):
        raise ValueError(f"edge must be 'crop' or 'partial', got {edge!r}")
    if edge == "partial" and reduction == "mean":
        raise ValueError("edge='partial' supports exact sum reduction only")
    is_torch = isinstance(data, torch.Tensor)
    if not is_torch and not (cp is not None and isinstance(data, cp.ndarray)):
        raise TypeError(f"bin reads a CuPy array or a Torch tensor on the GPU, got {type(data).__name__}.")
    if factor == 1:
        return data
    if axes not in ("detector", "scan", "all"):
        raise ValueError(f"axes must be 'detector', 'scan', or 'all', got {axes!r}")
    if (axes != "detector" and data.ndim != 4) or data.ndim not in (2, 3, 4):
        raise ValueError(f"axes={axes!r} bins a {data.ndim}D array; scan axes need 4D (scan_row, scan_col, det_row, det_col).")
    if dtype is None:
        integer = (not data.dtype.is_floating_point and not data.dtype.is_complex) if is_torch else cp.issubdtype(data.dtype, cp.integer)
        if reduction == "mean" or not integer:
            dtype = torch.float32 if is_torch else cp.float32
        else:
            # a wide accumulator of the input's sign: uint32 counts summed in uint32 would wrap
            dtype = torch.int64 if is_torch else (cp.int64 if cp.issubdtype(data.dtype, cp.signedinteger) else cp.uint64)
    # the binned axes, counted from the end for the detector and from the start for the scan
    binned = {"detector": [data.ndim - 2, data.ndim - 1], "scan": [0, 1], "all": [0, 1, 2, 3]}[axes]
    sizes = list(data.shape)
    for axis in binned:
        sizes[axis] = -(-sizes[axis] // factor) * factor if edge == "partial" else sizes[axis] // factor * factor
        if sizes[axis] == 0:
            raise ValueError(f"Shape {tuple(data.shape)} is too small for factor {factor}.")
    if edge == "partial" and sizes != list(data.shape):
        padded = torch.zeros(sizes, dtype=data.dtype, device=data.device) if is_torch else cp.zeros(sizes, dtype=data.dtype)
        padded[tuple(slice(0, size) for size in data.shape)] = data
        data = padded
    data = data[tuple(slice(0, size) for size in sizes)]
    if is_torch:
        data = data.contiguous()
    shape, reduced = [], []
    for axis, size in enumerate(sizes):
        if axis in binned:
            shape += [size // factor, factor]
            reduced.append(len(shape) - 1)
        else:
            shape.append(size)
    blocks = data.reshape(shape)
    if is_torch:
        return blocks.mean(dim=reduced, dtype=dtype) if reduction == "mean" else blocks.sum(dim=reduced, dtype=dtype)
    return blocks.mean(axis=tuple(reduced), dtype=dtype) if reduction == "mean" else blocks.sum(axis=tuple(reduced), dtype=dtype)


def _device_backend(backend: str, device: int | str | None) -> tuple[str, int | str | None]:
    """Turn a ``device=`` name into the backend it implies and the CUDA device the loaders take.

    ``device="cpu"`` and ``backend="cpu"`` must mean the same thing, as must
    ``device="mps"`` and ``backend="mps"``; a device naming a different
    backend than an explicit ``backend=`` is a contradiction, not a choice.
    """
    if device is None or isinstance(device, int):
        return backend, device
    requested = str(device).strip().lower()
    if requested == "auto":
        requested = select.resolve_device("auto")
    implied = "cuda" if requested.startswith("cuda") else requested
    if implied not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device {device!r}; use 'auto', 'cuda', 'cuda:N', 'mps', or 'cpu'.")
    if backend not in (None, "auto", implied):
        raise ValueError(f"device={device!r} contradicts backend={backend!r}; pass one of them.")
    return implied, (requested if requested.startswith("cuda:") else None)


def _selected_representation(
    paths: list[str | os.PathLike[str]],
    requested: DataRepresentation | str | None,
) -> DataRepresentation:
    """Resolve an explicit representation or inspect existing containers."""
    if requested is not None:
        selected = DataRepresentation.parse(requested)
        detected = [DataRepresentation.detect_source(path) for path in paths]
        if selected is DataRepresentation.PAIRED and any(
            item not in {DataRepresentation.DENSE, DataRepresentation.PAIRED} for item in detected
        ):
            raise NotImplementedError(
                "representation='paired' streams ordinary HDF5 acquisitions or reopens "
                "saved paired resident forms; transcoding encoded sources into the "
                "paired layout is not implemented."
            )
        if selected is not DataRepresentation.PAIRED and any(item is DataRepresentation.PAIRED for item in detected):
            raise ValueError(
                "A saved paired resident form reopens only as representation='paired'; "
                "omit representation= or request 'paired'."
            )
        return selected
    detected = {DataRepresentation.detect_source(path) for path in paths}
    if len(detected) > 1:
        raise ValueError(
            "One load call cannot mix ordinary HDF5, .qem and saved paired sources. "
            "Load each representation separately."
        )
    return detected.pop() if detected else DataRepresentation.DENSE
