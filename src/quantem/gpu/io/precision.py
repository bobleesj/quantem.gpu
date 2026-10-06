"""Scaled-precision loading and export, with persistent GPU-measured conversion error.

``io.load(..., dtype="scaled_uint16")`` converts float32 intensities to uint16
codes with one scale and offset per automatically chosen region, keeps the
codes ANS encoded on the GPU, and records the conversion error measured on
the GPU. ``io.save(..., dtype=...)`` writes such codes (or float16) to HDF5
with the same report, so reopening restores the original units. Sources are
read in bounded blocks: a file, an accelerator array, or a generated block
source from an algorithm package (``shape``, ``dtype``, ``blocks()``).
"""

import bisect
import json
import math
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path

import h5py
import numpy as np

from quantem.gpu.device import select
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.device.metal_runtime import complete_command, metal_queue
from quantem.gpu.formats.hdf5.readiness import file_signature
from quantem.gpu.formats.hdf5.reads import FrameReader
from quantem.gpu.formats.precision import PRECISION_ATTRIBUTE, saved_precision
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.hdf5.cuda.decode import decompress_prepared
from quantem.gpu.io.hdf5.write import H5Writer, wait_for_saves
from quantem.gpu.io.inspect import inspect
from quantem.gpu.resident import precision as conversion
from quantem.gpu.resident.cuda import precision as cuda
from quantem.gpu.resident.mps import precision as metal
from quantem.gpu.resident.mps.arrays import MetalArray

try:
    import torch
except ImportError:  # minimal IO installs convert NumPy and CuPy sources without Torch
    torch = None


def precision_name(dtype):
    """Recognize explicit approximate storage without changing integer casts."""
    if isinstance(dtype, str):
        token = dtype.lower()
        if token == "scaled_uint16":
            return token
        if token in {"float16", "f16"}:
            return "float16"
    if dtype is None:
        return None
    try:
        return "float16" if np.dtype(dtype) == np.dtype("float16") else None
    except TypeError:
        return None


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
    backend="cuda",
):
    """Load scaled uint16 precision ANS-encoded on the GPU, with bounded conversion and error scratch.

    ``io.load`` admits only scaled uint16 here: new conversions and saved
    regional (version 2) exports go region by region; a saved single-scale
    (version 1) export keeps its codes, and a saved float16 export is converted.
    """
    context = ExitStack()
    if backend == "mps":
        if device is not None and str(device) not in {"mps", "mps:0", "0"}:
            raise ValueError("Metal precision loading uses device='mps'; omit device.")
        pack, resident_type = metal.encode_ans, metal.PrecisionSource
    else:
        selected = _cuda_device(path) if device is None else int(str(device).removeprefix("cuda:"))
        context.enter_context(cp.cuda.Device(selected))
        # Return blocks cached by earlier work before converting, and this
        # conversion's own scratch after it, so neither sits beside the result.
        cp.get_default_memory_pool().free_all_blocks()
        context.callback(cp.get_default_memory_pool().free_all_blocks)
        pack, resident_type = cuda.encode_ans, cuda.PrecisionSource

    with context:
        source = _Source(path, scan_shape=scan_shape, dataset_path=dataset_path, backend=backend)
        storage = precision_name(dtype) or (source.saved or {}).get("storage")
        if storage is None:
            # One unconverted file among saved precision exports, without a dtype.
            source.close()
            raise ValueError(
                "Choose dtype='float16' or 'scaled_uint16' for precision loading."
            )
        if source.saved is None or source.saved.get("version") == 2:
            try:
                return _load_regional(source, scan_region, detector_region, verbose, pack, resident_type)
            finally:
                source.close()
        chunks = []
        try:
            reuse = storage == source.saved["storage"]
            report = dict(source.saved) if reuse else _new_report(source, storage)
            if not reuse:
                report["prior_conversion"] = source.saved
            row0, row1, col0, col1 = scan_region or (0, source.shape[0], 0, source.shape[1])
            dr0, dr1, dc0, dc1 = detector_region or (0, source.shape[2], 0, source.shape[3])
            shape = (row1 - row0, col1 - col0, dr1 - dr0, dc1 - dc0)
            for block in source.blocks(scan_region, detector_region):
                if reuse:
                    encoded = block
                else:
                    original = conversion.restore(block, source.saved, backend)
                    encoded = conversion.encode(original, report, backend)
                    restored = (
                        None
                        if backend == "cuda"
                        else conversion.restore(encoded, report, backend)
                    )
                    conversion.measure(original, restored, report, backend, encoded=encoded)
                chunks.append(pack(encoded, (1, encoded.shape[0], *shape[2:])))
            if not reuse:
                report["selection"] = {
                    "scan_region": scan_region,
                    "detector_region": detector_region,
                }
                _finish_report(report)
            resident = resident_type(chunks, shape, report)
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
                working_dtype="float32",
                representation="encoded",
                resident_codec="ans",
                residency="device",
                physical_resident_bytes=resident.nbytes,
                lossless_exact=report["changed"] == 0,
                source_dtype=report["source_dtype"],
                storage_dtype="uint16",
            )
            if verbose:
                print_report(report, shape, resident.nbytes, saved=reuse)
            return Dataset4dstemGPU(resident, metadata)
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
    source_master,
) -> str:
    """Save approximate intensities in bounded GPU-compressed HDF5 blocks; return the backend used."""
    backend = select.resolve_backend(backend)
    if backend == "mps":
        resident_type = metal.PrecisionSource
    elif backend == "cuda":
        resident_type = cuda.PrecisionSource
    else:
        raise NotImplementedError("Precision exporting requires CUDA or Metal; no CPU conversion is used.")
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
    if isinstance(data, Dataset4dstemGPU):
        metadata = {**data.metadata, **metadata}
        payload = data.data
    else:
        payload = data
    same_precision = isinstance(payload, resident_type) and (
        dtype is None or precision_name(dtype) == payload.precision["storage"]
    )
    if isinstance(payload, resident_type) and not same_precision:
        raise ValueError(
            "Convert a different precision from the float32 source to avoid compounded rounding."
        )
    if not same_precision and precision_name(dtype) is None:
        raise ValueError(
            "Choose float16 or scaled_uint16, or save the original float32 source."
        )
    source = None
    writer = None
    context = nullcontext() if backend == "mps" else cp.cuda.Device(_cuda_device(payload))
    with context:
        try:
            if same_precision:
                report = dict(payload.precision)
                shape = payload.shape
                encoded_blocks = payload.encoded_blocks()
            elif precision_name(dtype) == "scaled_uint16":
                source = _Source(payload, scan_shape=scan_shape, backend=backend)
                shape = source.shape
                report = {"version": 2, "storage": "scaled_uint16"}

                def convert_regional_blocks():
                    reports = []
                    first = 0
                    for block in source.blocks():
                        frames = math.prod(block.shape[:-2])
                        block = _as_frames(block, shape[2:])
                        encoded, region = _convert_region(source, block)
                        region.update(first_frame=first, stop_frame=first + frames)
                        reports.append(region)
                        first += frames
                        yield encoded
                        del encoded, block
                    if first != math.prod(shape[:2]):
                        raise ValueError("The source did not produce its complete scan; repeat the export.")
                    report.update(_regional_report(shape, reports))

                encoded_blocks = convert_regional_blocks()
            elif backend == "mps" and metal.is_mps_tensor(payload):
                if len(payload.shape) == 3:
                    if scan_shape is None:
                        raise ValueError("scan_shape is required for a flat MPS tensor.")
                    shape = tuple(scan_shape) + tuple(payload.shape[-2:])
                    tensor = payload.reshape(shape)
                elif len(payload.shape) == 4:
                    shape = tuple(payload.shape)
                    tensor = payload
                else:
                    raise ValueError("Precision export needs a 4D MPS tensor or a flat tensor with scan_shape.")
                report = _new_report(tensor, precision_name(dtype), metal.tensor_range(tensor))

                def convert_tensor_blocks():
                    frames = math.prod(shape[:2])
                    pixels = math.prod(shape[2:])
                    # A large bounded batch reduces Metal launch overhead while
                    # keeping an explicit ceiling for 24 GB laptops.
                    batch = max(128, min(frames, (512 * 1024**2 // (pixels * 4) // 128) * 128))
                    flat = tensor.reshape(frames, *shape[2:])
                    for first in range(0, frames, batch):
                        original = flat[first : min(first + batch, frames)]
                        encoded = conversion.encode(original, report, "mps")
                        conversion.measure(
                            original,
                            conversion.restore(encoded, report, "mps"),
                            report,
                            "mps",
                            encoded=encoded,
                        )
                        yield encoded

                encoded_blocks = convert_tensor_blocks()
            else:
                source = _Source(payload, scan_shape=scan_shape, backend=backend)
                shape = source.shape
                if source.saved and source.saved.get("version") == 2:
                    source.restore_saved_regions()
                report = _new_report(source, precision_name(dtype))

                def convert_blocks():
                    for block in source.blocks():
                        original = conversion.restore(block, source.saved, backend)
                        encoded = conversion.encode(original, report, backend)
                        restored = (
                            None
                            if backend == "cuda" and report["storage"] == "scaled_uint16"
                            else conversion.restore(encoded, report, backend)
                        )
                        conversion.measure(original, restored, report, backend, encoded=encoded)
                        yield encoded

                encoded_blocks = convert_blocks()
            storage_dtype = np.float16 if report["storage"] == "float16" else np.uint16
            metadata.update(
                scan_shape=shape[:2],
                detector_shape=shape[2:],
                n_frames=math.prod(shape[:2]),
                dtype=str(np.dtype(storage_dtype)),
            )
            metadata[PRECISION_ATTRIBUTE] = json.dumps(
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
            for encoded in encoded_blocks:
                writer.write(encoded)
                # The bounded writer queue applies backpressure while preserving
                # overlap with the next region. File boundaries drain separately.
            wait_for_saves()
            if not same_precision and report.get("version") != 2:
                _finish_report(report)
            # A generated source may carry its own records, such as a merge summary;
            # save_metadata is an optional field of the generated-source contract.
            generated_metadata = getattr(payload, "save_metadata", None)
            if generated_metadata is not None:
                metadata.update(dict(generated_metadata))
            # One JSON attribute survives HDF5 round trips without Python repr parsing.
            metadata[PRECISION_ATTRIBUTE] = json.dumps(report, allow_nan=False)
            metadata.pop("precision", None)
            writer.close(wait=True)
            writer = None
            if verbose:
                print(
                    f"Saved {report['storage']} with conversion report: RMS error {report['rmse']:.7g}, maximum {report['max_abs_error']:.7g}"
                )
            return backend
        finally:
            if writer is not None:
                failure = sys.exception()
                try:
                    writer.close(wait=True)
                except (RuntimeError, MemoryError, OSError, ValueError) as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Incomplete writer cleanup: {cleanup_error}")
            if source is not None:
                source.close()


def print_report(report, shape, resident_bytes, *, saved=False):
    """Print measured precision and residency without leaking source paths."""
    if report.get("version") == 2:
        origin = "Saved conversion report (not remeasured)" if saved else "GPU measured across all loaded values"
        print(f"{report['source_dtype']} → scaled_uint16 | {resident_bytes / 2**30:.3f} GiB ANS | "
              f"RMSE {report['rmse']:.7g}, max {report['max_abs_error']:.7g}, "
              f"overflow {report['overflow']} | {origin}")
        return
    print(
        f"Loaded {shape[0]}×{shape[1]} scan, {shape[2]}×{shape[3]} detector | "
        f"{report['source_dtype']} → {report['storage']} | "
        f"{resident_bytes / 2**30:.3f} GiB ANS on GPU"
    )
    print(f"  scale {report['scale']:.7g}, offset {report['offset']:.7g}")
    print(
        f"  {'Saved conversion report (not remeasured)' if saved else 'GPU measured across all loaded values'} | RMSE {report['rmse']:.7g}, "
        f"max {report['max_abs_error']:.7g}, positive→zero {report['positive_to_zero']:,}, "
        f"overflow {report['overflow']}, values {report['values']:,}"
    )


class _Source:
    """Read selected data in bounded blocks; decompression stays on the GPU.

    ``source`` is a file path, an array (CuPy, Torch, NumPy or h5py), or a
    generated source from an algorithm package: any other object with
    ``shape``, ``dtype`` and ``blocks()``, as documented for ``io.save``.
    """

    def __init__(self, source, *, scan_shape=None, dataset_path=None, backend="cuda"):
        self.backend = backend
        self.decoder = None
        self.stack = ExitStack()
        self.metadata = {}
        self.saved = None
        self.signatures = {}
        self.array = None
        self.generated = False
        self.reader = None
        # Set by restore_saved_regions.
        self.restore_regions = False
        if not isinstance(source, (str, Path)):
            self.array = source
            self.shape = tuple(source.shape)
            if len(self.shape) == 3 and scan_shape is not None:
                self.shape = (*scan_shape, *self.shape[-2:])
            self.dtype = np.dtype(str(source.dtype).removeprefix("torch."))
            # Arrays are read by frame index; anything else is an algorithm's
            # generated source that yields its own blocks.
            self.generated = not (
                isinstance(source, (np.ndarray, h5py.Dataset)) or _is_tensor(source)
                or (cp is not None and isinstance(source, cp.ndarray))
            )
        elif Path(source).suffix.lower() == ".npy":
            self.array = np.load(source, mmap_mode="r", allow_pickle=False)
            self.shape, self.dtype = self.array.shape, self.array.dtype
            self._seal(source)
        else:
            info = inspect(source, scan_shape=scan_shape)
            if not info.ready:
                raise ValueError(f"{info.reason}: {info.action}")
            if info.pixel_mask is not None:
                if backend == "mps":
                    invalid = metal.has_invalid_pixels(info.pixel_mask)
                else:
                    invalid = bool(cp.any(cp.asarray(info.pixel_mask) != 0).get())
                if invalid:
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
                self.reader = FrameReader(str(source))
                self.stack.callback(self.reader.close)
                for entry in self.reader.source_infos:
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

    def restore_saved_regions(self):
        """Yield float32 intensities, each region restored with its own saved calibration.

        Converting a saved regional export again must start from intensities,
        not codes whose scale changes from region to region.
        """
        self.restore_regions = True
        self.dtype = np.dtype("float32")

    def _seal(self, path):
        """Record a source file's identity so a rewrite during conversion is caught."""
        self.signatures[str(path)] = file_signature(path)

    def check(self):
        """Refuse to continue if any sealed source file changed."""
        if any(file_signature(p) != s for p, s in self.signatures.items()):
            raise RuntimeError(
                "Source changed during conversion; retry with immutable input files."
            )

    def blocks(self, region=None, detector_region=None):
        """Yield the selected frames as ``(frames, row, col)`` GPU blocks of about 256 MiB."""
        if self.restore_regions:
            for block, report in _calibrated_blocks(self, region, detector_region):
                yield conversion.restore(block, report, self.backend)
            return
        yield from self.stored_blocks(region, detector_region)

    def stored_blocks(self, region=None, detector_region=None):
        """Yield the selected frames as stored, before any regional restore."""
        if self.generated:
            if region is not None or detector_region is not None:
                raise ValueError(
                    "Generated 4D-STEM sources are saved at their declared complete geometry."
                )
            yield from self.array.blocks()
            return
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
        # A larger bounded block amortizes CUDA decompressor launches while
        # keeping the temporary below the 24 GiB laptop workflow budget.
        target_bytes = 256 * 1024**2
        batch = max(
            128,
            (target_bytes // (det_rows * det_cols * self.dtype.itemsize) // 128) * 128,
        )
        for first in range(0, count, batch):
            stop = min(first + batch, count)
            selected = np.arange(first, stop, dtype=np.int64)
            selected = (
                (selected // (col1 - col0) + row0) * cols
                + selected % (col1 - col0)
                + col0
            )
            if self.reader is not None:
                prepared = self.reader.prepare(selected)
                if self.backend == "mps":
                    values = self._decode_metal(prepared)
                else:
                    values = decompress_prepared(prepared, batch_bytes_target=target_bytes)
            elif self.backend == "cuda" and isinstance(self.array, cp.ndarray):
                values = self.array.reshape(-1, det_rows, det_cols)[
                    cp.asarray(selected)
                ]
            elif _is_tensor(self.array) and self.array.device.type in ("cuda", "mps"):
                flat = self.array.reshape(-1, det_rows, det_cols)
                indices = torch.as_tensor(selected, device=self.array.device)
                values = flat[indices].contiguous()
                if self.backend == "cuda":
                    values = cp.from_dlpack(values.detach())
            else:
                host = np.empty((stop - first, det_rows, det_cols), self.dtype)
                cursor = 0
                while cursor < len(selected):
                    row, col = divmod(int(selected[cursor]), cols)
                    length = min(col1 - col, len(selected) - cursor)
                    if len(self.array.shape) == 4:
                        piece = self.array[row, col : col + length]
                    else:
                        piece = self.array[
                            row * cols + col : row * cols + col + length
                        ]
                    if _is_tensor(piece):
                        piece = piece.detach().cpu().numpy()
                    host[cursor : cursor + length] = piece
                    cursor += length
                values = metal.upload(host) if self.backend == "mps" else cp.asarray(host)
            if self.backend == "mps":
                yield metal.crop(values, (dr0, dr1, dc0, dc1))
            else:
                yield cp.ascontiguousarray(values[:, dr0:dr1, dc0:dc1])
            del values
        self.check()

    def _decode_metal(self, prepared):
        """Decode prepared compressed frames on Metal, reusing this source's decoder when it fits."""
        # The Metal decoder is imported only on the MPS path, like every platform decoder.
        from quantem.gpu.io.hdf5.mps.decode import MPSDecompressor

        frames, frame_bytes = prepared["total_frames"], prepared["frame_bytes"]
        compressed = len(prepared["read_buffer"])
        decoder = self.decoder
        if decoder is None or frames > decoder.max_frames or compressed > decoder.max_compressed_bytes:
            if decoder is not None:
                decoder.free()
            self.decoder = MPSDecompressor(
                max_compressed_bytes=max(compressed, frames * frame_bytes * 2),
                max_frames=frames, frame_bytes=frame_bytes,
                n_blocks_per_frame=(frame_bytes + 8191) // 8192,
            )
        decoded = self.decoder.load_prepared_frames(prepared)
        # The decoder hands over ownership of its fresh output buffer, without a copy.
        result = MetalArray(decoded.shape, decoded.dtype, buffer=decoded._mtl)
        decoded._mtl = None
        return result

    def close(self):
        """Close source files and release the decoder's Metal scratch."""
        self.stack.close()
        self.array = None
        if self.decoder is not None:
            self.decoder.free()
            self.decoder = None


def _range(source):
    """Measure the entire source range with bounded accelerator reductions."""
    if source.backend == "mps":
        return metal.source_range(source.blocks(), source.saved)
    low, high = math.inf, -math.inf
    for block in source.blocks():
        values = conversion.restore(block, source.saved, source.backend)
        if bool(cp.any(~cp.isfinite(values)).get()):
            raise ValueError(
                "Precision conversion requires finite intensities; preserve this source as float32."
            )
        low = min(low, float(values.min().get()))
        high = max(high, float(values.max().get()))
    return low, high


def _new_report(source, storage, limits=None):
    """Start a conversion report: storage, intensity range, calibration and zeroed error counters.

    ``source`` is a ``_Source``, or a Torch MPS tensor saved directly, whose
    ``limits`` the caller measures.
    """
    low, high = _range(source) if limits is None else limits
    if storage == "float16" and max(abs(low), abs(high)) > 65504:
        raise ValueError(
            "Values exceed float16's finite range; use scaled_uint16 or preserve float32."
        )
    saved = source.saved if isinstance(source, _Source) else None
    report = {
        "version": 1,
        "storage": storage,
        "source_dtype": "float32"
        if saved and saved["storage"] == "scaled_uint16"
        else str(np.dtype(str(source.dtype).removeprefix("torch."))),
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
        "selection": {"scan_region": None, "detector_region": None},
    }
    # A generated source may describe what its values are, such as a merge;
    # report_context is an optional field of the generated-source contract.
    if isinstance(source, _Source) and source.generated:
        context = getattr(source.array, "report_context", None)
        if context is not None:
            report.update(dict(context))
    return report


def _finish_report(report):
    """Turn the accumulated squared error into the reported RMSE."""
    report["rmse"] = math.sqrt(report.pop("squared_error") / report["values"])
    return report


def _calibrated_blocks(source, scan_region, detector_region):
    """Split selected stored frames at saved calibration boundaries, each with its region's report."""
    if not source.saved or source.saved.get("version") != 2:
        for block in source.stored_blocks(scan_region, detector_region):
            yield block, source.saved
            del block
        return
    rows, cols = source.shape[:2]
    r0, r1, c0, c1 = scan_region or (0, rows, 0, cols)
    selected = [row * cols + col for row in range(r0, r1) for col in range(c0, c1)]
    regions = source.saved["regions"]
    ends = [item["stop_frame"] for item in regions]
    cursor = 0
    for block in source.stored_blocks(scan_region, detector_region):
        first = 0
        while first < block.shape[0]:
            index = bisect.bisect_right(ends, selected[cursor + first])
            stop = first + 1
            while stop < block.shape[0] and selected[cursor + stop] < ends[index]:
                stop += 1
            yield _slice_frames(block, first, stop), regions[index]
            first = stop
        cursor += block.shape[0]
        del block


def _slice_frames(block, first, stop):
    """Select frames [first, stop) of a block, copying when it is a Metal staging array.

    A ``MetalArray`` is one shared buffer without views, so its frames are
    blitted into a new buffer; CuPy and Torch blocks are sliced as views.
    """
    if first == 0 and stop == block.shape[0]:
        return block
    if isinstance(block, MetalArray):
        result = MetalArray((stop - first, *block.shape[1:]), block.dtype)
        command = metal_queue().commandBuffer()
        encoder = command.blitCommandEncoder()
        frame_bytes = math.prod(block.shape[1:]) * block.dtype.itemsize
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            block._mtl, first * frame_bytes, result._mtl, 0, result.nbytes
        )
        encoder.endEncoding()
        complete_command(command, "precision frame selection")
        return result
    return block[first:stop]


def _load_regional(source, scan_region, detector_region, verbose, pack, resident_type):
    """Convert each generated or loaded region once and retain calibrated codes."""
    r0, r1, c0, c1 = scan_region or (0, source.shape[0], 0, source.shape[1])
    d0, d1, e0, e1 = detector_region or (0, source.shape[2], 0, source.shape[3])
    shape = (r1 - r0, c1 - c0, d1 - d0, e1 - e0)
    chunks, reports = [], []
    first = 0
    previous_saved = None
    try:
        for block, saved in _calibrated_blocks(source, scan_region, detector_region):
            frames = math.prod(block.shape[:-2])
            block = _as_frames(block, shape[2:])
            if saved:
                encoded = block
                report = dict(saved)
            else:
                encoded, report = _convert_region(source, block)
            report.update(first_frame=first, stop_frame=first + frames)
            if saved is not None and saved is previous_saved:
                reports[-1]["stop_frame"] = first + frames
            else:
                reports.append(report)
            previous_saved = saved
            chunks.append(pack(encoded, (1, frames, *shape[2:])))
            first += frames
            # Packing is complete; do not overlap this region with the next producer call.
            del encoded, block
        if first != math.prod(shape[:2]):
            raise ValueError(
                "The source did not produce the complete declared scan; repeat the merge."
            )
        report = _regional_report(shape, reports)
        resident = resident_type(chunks, shape, report)
        metadata = dict(source.metadata)
        # A generated source may carry its own records, such as a merge summary;
        # save_metadata is an optional field of the generated-source contract.
        generated = getattr(source.array, "save_metadata", {}) if source.generated else {}
        for key, value in generated.items():
            if key.startswith("quantem_") and key.endswith("_v1"):
                metadata[key.removeprefix("quantem_").removesuffix("_v1")] = json.loads(
                    value
                )
            metadata[key] = value
        metadata.update(
            precision=report,
            working_shape=shape,
            scan_shape=shape[:2],
            detector_shape=shape[2:],
            n_frames=first,
            working_dtype="float32",
            source_dtype=report["source_dtype"],
            storage_dtype="uint16",
            representation="encoded",
            resident_codec="ans",
            residency="device",
            physical_resident_bytes=resident.nbytes,
            lossless_exact=report["changed"] == 0,
            conversion_report_origin="saved" if source.saved else "measured",
            selection={"scan_region": scan_region, "detector_region": detector_region},
        )
        if verbose:
            print_report(report, shape, resident.nbytes, saved=bool(source.saved))
        return Dataset4dstemGPU(resident, metadata)
    except BaseException:
        for chunk in chunks:
            chunk.release()
        raise


def _convert_region(source, block):
    """Measure and encode one region on its source accelerator."""
    original = conversion.restore(block, None, source.backend)
    if source.backend == "mps":
        limits = metal.source_range([original], None)
    else:
        # One host copy for both limits; a region's calibration needs them
        # before its codes can be produced, so this sync cannot be deferred.
        low, high = (float(v) for v in cp.stack([original.min(), original.max()]).get())
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(
                "Precision conversion requires finite intensities; preserve float32."
            )
        limits = low, high
    report = _new_report(source, "scaled_uint16", limits)
    report.update(version=2, range_scope="region")
    if source.backend == "mps":
        encoded = metal.encode_measure(original, report)
    else:
        encoded = cuda.encode_measure_regional(original, report)
    _finish_report(report)
    return encoded, report


def _regional_report(shape, reports):
    """Combine only GPU-produced scalar error reports."""
    count = sum(item["values"] for item in reports)
    return {
        "version": 2,
        "storage": "scaled_uint16",
        "source_dtype": reports[0]["source_dtype"],
        "source_shape": list(shape),
        "regions": reports,
        "complete": True,
        "range_scope": "automatic regions",
        "values": count,
        "rmse": math.sqrt(
            sum(item["rmse"] ** 2 * item["values"] for item in reports) / count
        ),
        "max_abs_error": max(item["max_abs_error"] for item in reports),
        "intensity_min": min(item["intensity_min"] for item in reports),
        "intensity_max": max(item["intensity_max"] for item in reports),
        **{
            key: sum(item[key] for item in reports)
            for key in ("changed", "positive_to_zero", "overflow", "clipped")
        },
    }


def _as_frames(block, detector_shape):
    """Return a block as ``(frames, row, col)``.

    Generated sources may yield blocks with any leading scan shape. Metal
    staging arrays are already frame-first and have no reshape.
    """
    if isinstance(block, MetalArray):
        return block
    return block.reshape(math.prod(block.shape[:-2]), *detector_shape)


def _cuda_device(source) -> int:
    """The CUDA device an in-memory source already occupies; files use the current device.

    Converting on the source's own GPU avoids a cross-device copy of every block.
    A generated block source names no device in its contract, so it converts on
    the current device, where its producer runs.
    """
    if isinstance(source, cuda.PrecisionSource):
        return source._device_id
    if isinstance(source, cp.ndarray):
        return source.device.id
    if _is_tensor(source) and source.device.type == "cuda" and source.device.index is not None:
        return source.device.index
    return cp.cuda.Device().id


def _is_tensor(value) -> bool:
    """Recognize a Torch tensor while Torch stays optional for NumPy and CuPy sources."""
    return torch is not None and isinstance(value, torch.Tensor)
