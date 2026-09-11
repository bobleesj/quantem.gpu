"""Bounded precision conversion with persistent scientific error measurements."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

_PRECISION_ATTRIBUTE = "quantem_precision_v1"


def precision_name(dtype):
    """Recognize explicit approximate storage without changing integer casts."""
    if isinstance(dtype, str) and dtype == "scaled_uint16":
        return dtype
    if dtype is None:
        return None
    try:
        return "float16" if np.dtype(dtype) == np.dtype("float16") else None
    except TypeError:
        return None


def saved_precision(path):
    """Read only persisted precision metadata, never detector values."""
    if Path(path).suffix.lower() not in {".h5", ".hdf5"} or not h5py.is_hdf5(path):
        return None
    with h5py.File(path, "r") as handle:
        value = handle.attrs.get(_PRECISION_ATTRIBUTE)
    if value is None:
        return None
    report = json.loads(value)
    if report.get("complete") is False:
        raise ValueError(
            "This precision export did not complete; repeat the export from its source."
        )
    if report.get("version") != 1 or report.get("storage") not in {
        "float16",
        "scaled_uint16",
    }:
        raise ValueError(
            "Unsupported saved precision metadata; use a compatible QuantEM version."
        )
    return report


class _Source:
    """Read selected data in bounded blocks; decompression stays on the GPU."""

    def __init__(self, source, *, scan_shape=None, dataset_path=None):
        self.stack = ExitStack()
        self.metadata = {}
        self.saved = None
        self.signatures = {}
        self.array = None
        self.session = None
        if not isinstance(source, (str, Path)):
            self.array = source
            self.shape = tuple(source.shape)
            if len(self.shape) == 3 and scan_shape is not None:
                self.shape = (*scan_shape, *self.shape[-2:])
            self.dtype = np.dtype(source.dtype)
        elif Path(source).suffix.lower() == ".npy":
            self.array = np.load(source, mmap_mode="r", allow_pickle=False)
            self.shape, self.dtype = self.array.shape, self.array.dtype
            self._seal(source)
        else:
            from .inspect import inspect
            from .load import _discover_chunk_names, _SparseFrameReadSession

            info = inspect(source, scan_shape=scan_shape)
            if not info.ready:
                raise ValueError(f"{info.reason}: {info.action}")
            if info.pixel_mask is not None:
                import cupy as cp

                if bool(cp.any(cp.asarray(info.pixel_mask) != 0).get()):
                    raise NotImplementedError(
                        "This source has invalid detector pixels. Precision loading "
                        "does not yet preserve detector validity; use a processed "
                        "float32 result or native packed loading."
                    )
            self.shape = (*info.scan_shape, *info.detector_shape)
            self.dtype = np.dtype(info.dtype)
            self.metadata = dict(info.metadata)
            self.saved = saved_precision(source)
            dataset_path = dataset_path or info.metadata.get("dataset_path")
            if dataset_path:
                handle = self.stack.enter_context(h5py.File(source, "r"))
                self.array = handle[dataset_path]
                if self.array.id.get_create_plist().get_nfilters():
                    self.stack.close()
                    raise NotImplementedError(
                        "Filtered generic HDF5 needs a supported GPU decoder. "
                        "Use io.save's master layout or an unfiltered dataset."
                    )
            else:
                self.names = _discover_chunk_names(str(source)) or ["data"]
                self.session = _SparseFrameReadSession(
                    str(source), self.names, apply_mask=False
                )
                self.stack.callback(self.session.close)
                for entry in self.session.source_infos:
                    self._seal(entry["path"])
            self.path = str(source)
            self._seal(source)
        if len(self.shape) != 4 or min(self.shape) < 1:
            self.stack.close()
            raise ValueError(
                "Precision loading needs a 4D scan; supply scan_shape for a flat scan."
            )
        if self.dtype not in (
            np.dtype("float16"),
            np.dtype("float32"),
            np.dtype("uint8"),
            np.dtype("uint16"),
        ):
            self.stack.close()
            raise TypeError(
                f"Precision conversion supports float16/float32 and native uint8/uint16; got {self.dtype}. Preserve this source without conversion."
            )

    def _seal(self, path):
        from .load import _file_source_signature

        self.signatures[str(path)] = _file_source_signature(path)

    def check(self):
        from .load import _file_source_signature

        if any(_file_source_signature(p) != s for p, s in self.signatures.items()):
            raise RuntimeError(
                "Source changed during conversion; retry with immutable input files."
            )

    def blocks(self, region=None, detector_region=None):
        import cupy as cp

        from .load import _decompress_prepared, _prepare_master_frames

        self.check()
        rows, cols, det_rows, det_cols = self.shape
        row0, row1, col0, col1 = region or (0, rows, 0, cols)
        if not (0 <= row0 < row1 <= rows and 0 <= col0 < col1 <= cols):
            raise ValueError(
                f"scan_region must lie within {self.shape[:2]}; got {region}."
            )
        dr0, dr1, dc0, dc1 = detector_region or (0, det_rows, 0, det_cols)
        if not (0 <= dr0 < dr1 <= det_rows and 0 <= dc0 < dc1 <= det_cols):
            raise ValueError(
                f"detector_region must lie within {self.shape[2:]}; got {detector_region}."
            )
        count = (row1 - row0) * (col1 - col0)
        # Bounds decode and conversion scratch even for large detector geometries.
        batch = max(
            128,
            (32 * 1024**2 // (det_rows * det_cols * self.dtype.itemsize) // 128) * 128,
        )
        for first in range(0, count, batch):
            stop = min(first + batch, count)
            selected = np.arange(first, stop, dtype=np.int64)
            selected = (
                (selected // (col1 - col0) + row0) * cols
                + selected % (col1 - col0)
                + col0
            )
            if self.session is not None:
                prepared = _prepare_master_frames(
                    self.path,
                    self.names,
                    selected,
                    apply_mask=False,
                    read_session=self.session,
                )
                values = _decompress_prepared(
                    prepared,
                    auto_narrow=False,
                    output_dtype=self.dtype,
                    batch_bytes_target=32 * 1024**2,
                    prune_device_pool=False,
                )
            elif isinstance(self.array, cp.ndarray):
                values = self.array.reshape(-1, det_rows, det_cols)[
                    cp.asarray(selected)
                ]
            else:
                host = np.empty((stop - first, det_rows, det_cols), self.dtype)
                cursor = 0
                while cursor < len(selected):
                    row, col = divmod(int(selected[cursor]), cols)
                    length = min(col1 - col, len(selected) - cursor)
                    if len(self.array.shape) == 4:
                        host[cursor : cursor + length] = self.array[
                            row, col : col + length
                        ]
                    else:
                        host[cursor : cursor + length] = self.array[
                            row * cols + col : row * cols + col + length
                        ]
                    cursor += length
                values = cp.asarray(host)
            yield cp.ascontiguousarray(values[:, dr0:dr1, dc0:dc1])
        self.check()

    def close(self):
        self.stack.close()
        self.array = None


def _restore(values, report):
    """Restore scientific units without exposing encoded integers as counts."""
    import cupy as cp

    if report and report["storage"] == "scaled_uint16":
        return (values.astype(cp.float64) * report["scale"] + report["offset"]).astype(
            cp.float32
        )
    return values.astype(cp.float32)


def _range(source):
    """Measure the entire source range with bounded accelerator reductions."""
    import cupy as cp

    low, high = math.inf, -math.inf
    for block in source.blocks():
        values = _restore(block, source.saved)
        if bool(cp.any(~cp.isfinite(values)).get()):
            raise ValueError(
                "Precision conversion requires finite intensities; preserve this source as float32."
            )
        low = min(low, float(values.min().get()))
        high = max(high, float(values.max().get()))
    return low, high


def _encode(values, report):
    """Convert explicitly requested precision on the accelerator."""
    import cupy as cp

    if report["storage"] == "float16":
        return values.astype(cp.float16)
    return cp.clip(
        cp.rint((values.astype(cp.float64) - report["offset"]) / report["scale"]),
        0,
        65535,
    ).astype(cp.uint16)


def _new_report(source, storage):
    low, high = _range(source)
    if storage == "float16" and max(abs(low), abs(high)) > 65504:
        raise ValueError(
            "Values exceed float16's finite range; use scaled_uint16 or preserve float32."
        )
    return {
        "version": 1,
        "storage": storage,
        "source_dtype": str(source.dtype),
        "source_shape": list(source.shape),
        "intensity_min": low,
        "intensity_max": high,
        "scale": (high - low) / 65535
        if storage == "scaled_uint16" and high != low
        else 1.0,
        "offset": low if storage == "scaled_uint16" else 0.0,
        "values": 0,
        "squared_error": 0.0,
        "max_abs_error": 0.0,
        "positive_to_zero": 0,
        "changed": 0,
        "overflow": 0,
        "clipped": 0,
        "scope": "all loaded values",
        "range_scope": "complete source",
        "measurement": "GPU comparison against source",
    }


def _measure(original, restored, report):
    import cupy as cp

    difference = restored.astype(cp.float64) - original.astype(cp.float64)
    report["values"] += original.size
    report["squared_error"] += float(cp.sum(difference * difference).get())
    report["max_abs_error"] = max(
        report["max_abs_error"], float(cp.max(cp.abs(difference)).get())
    )
    report["positive_to_zero"] += int(
        cp.count_nonzero((original > 0) & (restored == 0)).get()
    )
    report["changed"] += int(cp.count_nonzero(original != restored).get())
    report["overflow"] += int(cp.count_nonzero(~cp.isfinite(restored)).get())


def _finish_report(report):
    report["rmse"] = math.sqrt(report.pop("squared_error") / report["values"])
    return report


def print_report(report, shape, resident_bytes, *, saved=False):
    """Print measured precision and residency without leaking source paths."""
    values = math.prod(shape)
    print(f"Loaded {shape[0]} × {shape[1]} scan, {shape[2]} × {shape[3]} detector")
    print(f"Precision: {report['source_dtype']} → {report['storage']}")
    print(
        f"Intensity range: {report['intensity_min']:.7g}–{report['intensity_max']:.7g}"
    )
    if report["storage"] == "scaled_uint16":
        print(f"Intensity step: {report['scale']:.7g}; offset: {report['offset']:.7g}")
    print(
        f"Storage: {values * np.dtype(report['source_dtype']).itemsize / 2**30:.3f} GiB source-equivalent dense → {resident_bytes / 2**30:.3f} GiB packed on GPU"
    )
    print("Approximate precision; source file unchanged.")
    print(
        "Saved conversion report (not remeasured against the original):"
        if saved
        else "Difference from source (GPU measured across all loaded values):"
    )
    print(
        f"  RMS error: {report['rmse']:.7g}; maximum absolute error: {report['max_abs_error']:.7g}"
    )
    print(
        f"  Positive values becoming zero: {report['positive_to_zero']:,} ({100 * report['positive_to_zero'] / report['values']:.3f}% of measured values)"
    )
    print(
        f"  Overflow: {report['overflow']}; clipped: {report['clipped']}; measured values: {report['values']:,}"
    )


def load_precision(
    path,
    *,
    dtype,
    device,
    scan_shape,
    dataset_path,
    scan_region,
    detector_region,
    verbose,
):
    """Load complete packed precision with bounded conversion and error scratch."""
    import cupy as cp

    from .backends.cuda._ans import CudaPackedResidentCounts
    from .backends.cuda.precision import PrecisionSource
    from .models import FourDSTEMData

    selected = (
        cp.cuda.Device().id
        if device is None
        else int(str(device).removeprefix("cuda:"))
    )
    with cp.cuda.Device(selected):
        source = _Source(path, scan_shape=scan_shape, dataset_path=dataset_path)
        chunks = []
        try:
            storage = precision_name(dtype) or (source.saved or {}).get("storage")
            if storage is None:
                raise ValueError(
                    "Choose dtype='float16' or 'scaled_uint16' for precision loading."
                )
            reuse = source.saved is not None and storage == source.saved["storage"]
            report = dict(source.saved) if reuse else _new_report(source, storage)
            if source.saved and not reuse:
                report["prior_conversion"] = source.saved
            row0, row1, col0, col1 = scan_region or (
                0,
                source.shape[0],
                0,
                source.shape[1],
            )
            dr0, dr1, dc0, dc1 = detector_region or (
                0,
                source.shape[2],
                0,
                source.shape[3],
            )
            shape = (row1 - row0, col1 - col0, dr1 - dr0, dc1 - dc0)
            for block in source.blocks(scan_region, detector_region):
                original = _restore(block, source.saved)
                encoded = block if reuse else _encode(original, report)
                if not reuse:
                    _measure(original, _restore(encoded, report), report)
                chunks.append(
                    CudaPackedResidentCounts.from_array(
                        encoded.view(cp.uint16), (1, encoded.shape[0], *shape[2:])
                    )
                )
            if not reuse:
                report["selection"] = {
                    "scan_region": scan_region,
                    "detector_region": detector_region,
                }
                _finish_report(report)
            resident = PrecisionSource(chunks, shape, report)
            metadata = dict(source.metadata)
            metadata.update(
                source_shape=source.shape,
                scan_shape=shape[:2],
                detector_shape=shape[2:],
                n_frames=math.prod(shape[:2]),
                selection={
                    "scan_region": scan_region,
                    "detector_region": detector_region,
                },
                precision=report,
                conversion_report_origin="saved" if reuse else "measured",
                working_shape=shape,
                working_dtype="float16" if storage == "float16" else "float32",
                representation="packed",
                residency="device",
                physical_resident_bytes=resident.nbytes,
                lossless_exact=report["changed"] == 0,
                source_dtype=report["source_dtype"],
                storage_dtype="float16" if storage == "float16" else "uint16",
            )
            if verbose:
                print_report(report, shape, resident.nbytes, saved=reuse)
            return FourDSTEMData(resident, metadata)
        except BaseException:
            for chunk in chunks:
                chunk.release()
            raise
        finally:
            source.close()


def save_precision(
    filepath,
    data,
    *,
    dtype,
    scan_shape,
    metadata,
    backend,
    format,
    compression,
    frames_per_file,
    verbose,
    wait,
    source_master,
):
    """Save approximate intensities in bounded GPU-compressed HDF5 blocks."""
    import cupy as cp

    from .backends.cuda.precision import PrecisionSource
    from .models import FourDSTEMData
    from .save import H5Writer, SaveResult, wait_for_saves

    if backend not in {"auto", "cuda"}:
        raise NotImplementedError(
            "Precision exporting currently requires CUDA; no CPU conversion is used."
        )
    if format != "arina" or compression not in {
        "auto",
        "lz4",
        "bslz4",
        "bitshuffle_lz4",
    }:
        raise NotImplementedError(
            "Precision exports currently use GPU bitshuffle/LZ4 HDF5; omit format and compression."
        )
    if source_master is not None:
        raise ValueError(
            "Converted intensities are not raw detector counts; pass calibration through metadata instead of source_master."
        )
    metadata = dict(metadata or {})
    if isinstance(data, FourDSTEMData):
        metadata = {**data.metadata, **metadata}
        payload = data.data
    else:
        payload = data
    same_precision = isinstance(payload, PrecisionSource) and (
        dtype is None or precision_name(dtype) == payload.precision["storage"]
    )
    if isinstance(payload, PrecisionSource) and not same_precision:
        raise ValueError(
            "Convert a different precision from the float32 source to avoid compounded rounding."
        )
    if not same_precision and precision_name(dtype) is None:
        raise ValueError(
            "Choose float16 or scaled_uint16, or save the original float32 source."
        )
    source = None
    writer = None
    device_id = (
        payload._device_id
        if same_precision
        else (
            payload.device.id
            if isinstance(payload, cp.ndarray)
            else cp.cuda.Device().id
        )
    )
    with cp.cuda.Device(device_id):
        try:
            if same_precision:
                report = dict(payload.precision)
                shape = payload.shape
                encoded_blocks = payload.encoded_blocks()
            else:
                source = _Source(payload, scan_shape=scan_shape)
                shape = source.shape
                report = _new_report(source, precision_name(dtype))

                def convert_blocks():
                    for block in source.blocks():
                        original = _restore(block, source.saved)
                        encoded = _encode(original, report)
                        _measure(original, _restore(encoded, report), report)
                        yield encoded

                encoded_blocks = convert_blocks()
            storage_dtype = np.float16 if report["storage"] == "float16" else np.uint16
            metadata.update(
                scan_shape=shape[:2],
                detector_shape=shape[2:],
                n_frames=math.prod(shape[:2]),
                dtype=str(np.dtype(storage_dtype)),
            )
            metadata[_PRECISION_ATTRIBUTE] = json.dumps(
                {**report, "complete": False}, allow_nan=False
            )
            writer = H5Writer(
                filepath,
                math.prod(shape[:2]),
                shape[2:],
                scan_shape=shape[:2],
                metadata=metadata,
                dtype=storage_dtype,
                frames_per_file=frames_per_file,
                compression="lz4",
            )
            for index, encoded in enumerate(encoded_blocks):
                writer.write(encoded)
                if (index + 1) % 4 == 0:
                    wait_for_saves()
            wait_for_saves()
            if not same_precision:
                _finish_report(report)
            # One JSON attribute survives HDF5 round trips without Python repr parsing.
            metadata[_PRECISION_ATTRIBUTE] = json.dumps(report, allow_nan=False)
            metadata.pop("precision", None)
            writer.close(wait=True)
            writer = None
            if verbose:
                print(
                    f"Saved {report['storage']} with conversion report: RMS error {report['rmse']:.7g}, maximum {report['max_abs_error']:.7g}"
                )
            return SaveResult(str(filepath), "cuda", complete=True)
        finally:
            if writer is not None:
                writer.close(wait=True)
            if source is not None:
                source.close()
