"""Byte-bounded ingestion into the shared integer and float ANS residents.

Arrays already in memory, and HDF5 layouts outside the direct chunk decoder, both
reach the residents through ``load_array_resident``.
"""

import base64
import bisect
import copy
import hashlib
import json
import math
import os
import time
import zlib
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch

from quantem.gpu.device.cuda_runtime import cuda_device_index
from quantem.gpu.device.metal_runtime import shared_array
from quantem.gpu.formats.hdf5.frames import detector_sources
from quantem.gpu.formats.hdf5.master import HOT_PIXEL_CORRECTION_ATTRIBUTE
from quantem.gpu.formats.qem import metadata as qem_metadata
from quantem.gpu.formats.qem.snapshot import BIT_CODEC, BIT_DTYPES, FLOAT_CODEC, INTEGER_CODEC, encode_valid
from quantem.gpu.io.dataset import Dataset4dstem, resident_metadata
from quantem.gpu.resident import float_ans
from quantem.gpu.resident.bit_ans import BitANSResident
from quantem.gpu.resident.cuda.counts import StreamedCounts, pinned
from quantem.gpu.resident.cuda.hot_pixels import CUDAHotPixelCorrector
from quantem.gpu.resident.hot_pixels import hot_pixel_record
from quantem.gpu.resident.mps import precision as metal_precision
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.hot_pixels import MPSHotPixelCorrector
from quantem.gpu.resident.mps.spatial import build_index

MAX_INGEST_BYTES = 32 << 20


def read_frame_block(data, shape: tuple[int, ...], first: int, stop: int) -> np.ndarray | torch.Tensor:
    """Copy file-backed rows without reshaping a noncontiguous acquisition."""
    if torch.is_tensor(data):
        bits_t = data.detach().view(torch.uint8)
        block_t = torch.empty(
            (stop - first, *shape[2:]), dtype=torch.uint8, device=data.device
        )
        cursor = first
        while cursor < stop:
            row, column = divmod(cursor, shape[1])
            count = min(shape[1] - column, stop - cursor)
            block_t[cursor - first:cursor - first + count] = bits_t[row, column:column + count]
            cursor += count
        return block_t.view(data.dtype)
    if data.ndim == 3:
        return np.ascontiguousarray(data[first:stop])
    block = np.empty((stop - first, *shape[2:]), data.dtype)
    cursor = first
    while cursor < stop:
        row, column = divmod(cursor, shape[1])
        count = min(shape[1] - column, stop - cursor)
        block[cursor - first : cursor - first + count] = data[
            row, column : column + count
        ]
        cursor += count
    return block


def load_array_resident(
    shape: tuple[int, ...],
    dtype: np.dtype | str | torch.dtype,
    read_frames: Callable[[int, int], np.ndarray | torch.Tensor],
    metadata: dict,
    *,
    backend: str,
    device: int | str | None = None,
    pixel_mask: np.ndarray | None = None,
    hot_pixel_correction: str = "none",
    detector_transpose: bool = False,
    verbose: bool = False,
) -> Dataset4dstem:
    """Encode bounded input windows; never allocate a dense acquisition.

    ``shape``, ``read_frames``, ``metadata`` and ``pixel_mask`` describe the source as stored. With
    ``detector_transpose`` each block is transposed on the GPU after its flagged pixels are corrected and before it is
    encoded (``transposed_frames``), so the resident holds (scan row, scan col, detector col, detector row) of the
    source and the record says so (``transposed_record``).
    """
    started = time.perf_counter()
    shape = tuple(shape)
    if detector_transpose and backend != "cuda":
        raise NotImplementedError("detector_transpose needs CUDA; load at the stored orientation on this device.")
    if detector_transpose:
        refuse_second_transpose(metadata)
    stored_shape = (*shape[:2], shape[3], shape[2]) if detector_transpose else shape
    source_dtype = np.dtype(str(dtype).removeprefix("torch.")) if isinstance(dtype, torch.dtype) else np.dtype(dtype)
    dtype = source_dtype.newbyteorder("=")
    peak_bytes = 0
    if dtype.name not in BIT_DTYPES:
        raise NotImplementedError(
            f"ANS supports real integer and floating dtypes through 64 bits; got {dtype}. "
            "Keep the original source; no cast or upload occurred."
        )
    correction = hot_pixel_record(pixel_mask, hot_pixel_correction, backend=backend)
    if correction["applied"] and dtype.name not in ("uint8", "uint16", "uint32"):
        raise ValueError(
            f"The source flags {correction['pixel_count']} detector pixels; hot_pixel_correction="
            f"{hot_pixel_correction!r} fills them in uint8/uint16/uint32 counts only, and this acquisition is "
            f"{dtype}. Load it with hot_pixel_correction='none' to keep the stored values; flagged pixels "
            "stay excluded from detector products."
        )
    floating = dtype == np.dtype("float32")
    bit_lanes = dtype not in (np.dtype("uint8"), np.dtype("uint16"))
    # The word codec uses two bytes even for signed 8-bit measurements.
    frame_bytes = math.prod(shape[2:]) * max(2, dtype.itemsize)
    if frame_bytes > MAX_INGEST_BYTES:
        raise MemoryError("One detector frame exceeds the 32 MiB ingestion limit.")
    block_scans = min(512, MAX_INGEST_BYTES // frame_bytes)
    valid = np.ones(shape[2:], bool)
    if pixel_mask is not None and not correction["applied"]:
        valid &= np.asarray(pixel_mask) == 0
    if detector_transpose:
        # the resident's validity mask is read in its own (transposed) detector frame
        valid = np.ascontiguousarray(valid.T)
    context = nullcontext()
    if backend == "cuda":
        import cupy as cp

        selected = cuda_device_index(device)
        context = cp.cuda.Device(selected)
    elif backend != "mps":
        raise ValueError("Encoded ingestion requires backend='cuda' or 'mps'.")
    elif device not in (None, "mps"):
        raise ValueError("Metal uses device='mps'; omit numeric device selection.")

    source = corrector = None
    checksum = _Checksum() if floating else None
    with context:
        try:
            if bit_lanes:
                header_metadata = transposed_record(metadata) if detector_transpose else metadata
                source = bit_resident(stored_shape, dtype, header_metadata, valid, backend, device)
                resident = source._lanes
            elif backend == "cuda":
                source = resident = StreamedCounts(stored_shape, dtype, valid)
            else:
                source = resident = MPSStreamedCounts(shape, dtype, valid)
            corrector_type = CUDAHotPixelCorrector if backend == "cuda" else MPSHotPixelCorrector
            corrector = corrector_type(pixel_mask, hot_pixel_correction)

            for first in range(0, math.prod(shape[:2]), block_scans):
                stop = min(first + block_scans, math.prod(shape[:2]))
                block = read_frames(first, stop)
                tensor_input = torch.is_tensor(block)
                if tensor_input and detector_transpose:
                    # a Torch block reaches the encoder as its bytes, which a transpose would scramble
                    raise NotImplementedError("detector_transpose reads NumPy, file and CuPy frame blocks.")
                # frames already decoded on this GPU (a .qem encoded again with detector_transpose)
                device_input = backend == "cuda" and isinstance(block, cp.ndarray)
                block_dtype = np.dtype(str(block.dtype).removeprefix("torch.")) if tensor_input else block.dtype
                if (
                    block.shape != (stop - first, *shape[2:])
                    or block_dtype != source_dtype
                ):
                    raise ValueError(
                        "The source changed shape or dtype while loading; reopen it."
                    )
                if not tensor_input and not device_input:
                    block = np.ascontiguousarray(block)
                    if not block.dtype.isnative:
                        block = block.byteswap().view(dtype)
                peak_bytes = max(peak_bytes, block.numel() * block.element_size() if tensor_input else block.nbytes)
                if floating and not detector_transpose:
                    checksum.host(block)
                if backend == "cuda":
                    raw = (
                        cp.from_dlpack(block.view(torch.uint8).to(f"cuda:{selected}"))
                        if tensor_input else cp.ascontiguousarray(block) if device_input else cp.asarray(block)
                    )
                    if corrector is not None:
                        corrector.apply(raw)
                    if detector_transpose:
                        raw = transposed_frames(raw)
                        if floating:
                            # the float codec's checksum covers the values as stored: the transposed block
                            checksum.device(raw)
                    (source.append_device if bit_lanes else resident.append)(raw)
                    del raw
                else:
                    if tensor_input:
                        from quantem.gpu.device.metal_runtime import tensor_buffer

                        block_t = block.view(torch.uint8).to("mps")
                        torch.mps.synchronize()
                        source.append_device(shared_array(tensor_buffer(block_t), dtype, block_t.shape))
                        del block_t, block
                        continue
                    raw = metal_precision.upload(block)
                    try:
                        if corrector is not None:
                            corrector.apply(raw)
                        (source.append_device if bit_lanes else resident.append)(
                            shared_array(raw._mtl, raw.dtype, raw.shape)
                        )
                        if not bit_lanes:
                            resident.spatial_chunks.append(
                                build_index(resident, raw._mtl, len(block))
                            )
                    finally:
                        raw.release()
                del block
            if floating:
                source.header["logical_sha256"] = checksum.hexdigest()
            record = dict(metadata)
            record.update(
                resident_metadata(shape, dtype, source.nbytes, backend=backend),
                device=f"cuda:{selected}" if backend == "cuda" else "mps",
                source_dtype=source_dtype.name,
                source_logical_tensor_bytes=math.prod(shape) * source_dtype.itemsize,
                resident_profile=source.header["codec"] if bit_lanes else INTEGER_CODEC,
                **correction_metadata(correction),
                load_timings=dict(
                    resident.load_metrics,
                    peak_ingest_bytes=peak_bytes,
                    resident_ready_seconds=time.perf_counter() - started,
                ),
            )
            if detector_transpose:
                record = transposed_record(record)
            if verbose:
                print(
                    f"Loaded {source_dtype.name} measurements as {dtype.name} ANS on {backend}; "
                    "no binning or cropping"
                    + ("; detector rows and columns swapped (detector_transpose)." if detector_transpose else ".")
                )
            return Dataset4dstem(source, record)
        except BaseException:
            if source is not None:
                source.release()
            raise
        finally:
            if corrector is not None:
                corrector.close()
            if checksum is not None:
                checksum.close()


class _Checksum:
    """SHA-256 of the stored values in scan order, hashed on one worker thread while the next block is read.

    The float codec records it (``logical_sha256``) and the reference decoder checks it. Hashing 1 GB takes about a
    quarter of a second, so it runs beside the reading and encoding instead of between them. At most two blocks are
    held: a host block by its view; a device block (frames transposed on the GPU) read back into one of two
    page-locked buffers in turn, each rewritten only once the block read into it before has been hashed.
    """

    def __init__(self):
        self.digest = hashlib.sha256()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending: deque = deque()
        self.buffers: list[np.ndarray] = []
        self.blocks = 0

    def _wait(self, held: int) -> None:
        while len(self.pending) > held:
            self.pending.popleft().result()

    def _hash(self, values: np.ndarray) -> None:
        self.pending.append(self.worker.submit(self.digest.update, memoryview(values).cast("B")))
        self.blocks += 1

    def host(self, block: np.ndarray) -> None:
        self._wait(held=1)
        self._hash(block)

    def device(self, raw) -> None:
        """Copy a device block back on a stream of its own and hash it once the copy lands, while the caller encodes
        it and reads the next block. ``raw`` is held until then, so the memory pool cannot hand it out meanwhile."""
        import cupy as cp

        self._wait(held=1)           # the block still waiting was read into the other buffer
        slot = self.blocks % 2
        if len(self.buffers) <= slot:
            self.buffers.append(pinned(raw.nbytes))
        elif self.buffers[slot].nbytes < raw.nbytes:
            self.buffers[slot] = pinned(raw.nbytes)
        host = self.buffers[slot][:raw.nbytes].view(raw.dtype).reshape(raw.shape)
        if self.stream is None:
            self.stream = cp.cuda.Stream(non_blocking=True)
        ready = cp.cuda.Event()
        ready.record()                               # the transpose, queued on the caller's stream
        self.stream.wait_event(ready)
        raw.get(out=host, stream=self.stream, blocking=False)
        landed = self.stream.record()

        def digest(raw=raw, device=raw.device.id):
            with cp.cuda.Device(device):
                landed.synchronize()
            self.digest.update(memoryview(host).cast("B"))

        self.pending.append(self.worker.submit(digest))
        self.blocks += 1

    stream = None

    def hexdigest(self) -> str:
        self._wait(held=0)
        return self.digest.hexdigest()

    def close(self) -> None:
        self.worker.shutdown(wait=True)


def bit_resident(shape, dtype, metadata, valid, backend, device=None):
    """Construct an empty typed GPU word resident with acquisition metadata."""
    dtype = np.dtype(dtype)
    floating = dtype == np.dtype("float32") and bool(np.all(valid))
    codec = FLOAT_CODEC if floating else BIT_CODEC
    scientific = qem_metadata.acquisition_metadata(shape, metadata)
    header = dict(
        container="quantem.qem", container_version=1, version=1,
        profile=codec, codec=codec, shape=list(shape), dtype=dtype.name,
        metadata=qem_metadata.retained_metadata(metadata),
        scientific_metadata=scientific,
    )
    if floating:
        header["empad"] = dict(metadata.get("qem_empad") or {})
        header["empad"].setdefault("format_identifier", "float32")
        header["empad"].setdefault("format_name", "Float32")
        header["empad"].setdefault("microscope_metadata", {
            str(key): str(value) for key, value in scientific["source_metadata"].items()
        })
    else:
        header["valid"] = encode_valid(valid)
    resident_type = float_ans.FloatANSResident if floating else BitANSResident
    return resident_type(header, backend, device)


_TRANSPOSE_SOURCE = r"""
// One 32 x 32 tile of a frame per block, staged through shared memory so that both the read of the
// frame and the write of its transpose are whole rows of consecutive elements (a strided copy reads
// one element of every row, and the encoder's input is written at half the memory rate).
template <typename T>
__device__ void transpose_tile(const T* raw, T* out, unsigned rows, unsigned cols) {
    __shared__ T tile[32][33];
    const unsigned long long frame = (unsigned long long)blockIdx.z * rows * cols;
    unsigned col = blockIdx.x * 32 + threadIdx.x, row = blockIdx.y * 32 + threadIdx.y;
    for (unsigned step = 0; step < 32; step += 8)
        if (row + step < rows && col < cols) tile[threadIdx.y + step][threadIdx.x] = raw[frame + (row + step) * cols + col];
    __syncthreads();
    col = blockIdx.y * 32 + threadIdx.x; row = blockIdx.x * 32 + threadIdx.y;
    for (unsigned step = 0; step < 32; step += 8)
        if (row + step < cols && col < rows) out[frame + (row + step) * rows + col] = tile[threadIdx.x][threadIdx.y + step];
}
extern "C" __global__ void transpose_1(const unsigned char* raw, unsigned char* out, unsigned rows, unsigned cols) { transpose_tile(raw, out, rows, cols); }
extern "C" __global__ void transpose_2(const unsigned short* raw, unsigned short* out, unsigned rows, unsigned cols) { transpose_tile(raw, out, rows, cols); }
extern "C" __global__ void transpose_4(const unsigned int* raw, unsigned int* out, unsigned rows, unsigned cols) { transpose_tile(raw, out, rows, cols); }
extern "C" __global__ void transpose_8(const unsigned long long* raw, unsigned long long* out, unsigned rows, unsigned cols) { transpose_tile(raw, out, rows, cols); }
"""


def transposed_frames(raw):
    """A block of frames (frames, detector rows, detector cols) with each frame's rows and columns swapped, on its GPU.

    ``out[f, i, j] = raw[f, j, i]``, bit for bit (the elements are moved as unsigned words of their size). The block
    is already on the device, between the flagged-pixel correction and the encoder, so the swap costs one tiled
    pass over device memory (about 1.4 TB/s) and none on the host. Why: some acquisitions record the detector
    transposed relative to the scan (a reflection, which no scan-detector rotation can undo), and every later step
    reads the frames as the resident holds them. Reading them transposed inside the encoder instead was slower
    (0.80 s against 0.67 s for a 512 x 512 x 192 x 192 master): its reads of neighbouring pixels are no longer adjacent.
    """
    import cupy as cp

    from quantem.gpu.device.cuda_runtime import cuda_module

    frames, rows, cols = (int(side) for side in raw.shape)
    raw = cp.ascontiguousarray(raw)
    out = cp.empty((frames, cols, rows), raw.dtype)
    if raw.size:
        size = raw.dtype.itemsize
        kernel = cuda_module(_TRANSPOSE_SOURCE, ("transpose_1", "transpose_2", "transpose_4", "transpose_8"),
                             ("--std=c++17",))[f"transpose_{size}"]
        words = {1: cp.uint8, 2: cp.uint16, 4: cp.uint32, 8: cp.uint64}[size]
        kernel(((cols + 31) // 32, (rows + 31) // 32, frames), (32, 8),
               (raw.view(words), out.view(words), np.uint32(rows), np.uint32(cols)))
    return out


# Per-axis detector keys: (row, col) pairs and four-axis lists whose last two entries are the detector's.
_DETECTOR_PAIRS = ("detector_shape", "binned_from_detector_shape", "detector_sampling_mrad", "detector_sampling_inv_A")
_AXIS_LISTS = ("working_shape", "source_shape", "sampling", "units", "origin")


def transposed_record(metadata: dict) -> dict:
    """The record of an acquisition loaded with ``detector_transpose``: every per-axis detector value swapped.

    A value stated for detector (row, col) belongs to the other axis once the frames are transposed: the sampling
    pairs, the shape keys, the flagged-pixel mask and its coordinates, the bright-field disk centre, the last two
    entries of the four-axis lists, and in a saved schema-2 record the detector axes, their quantities and overrides.
    ``detector_transpose`` is set and declared as a processing step, so a saved copy and a calibration read it. A
    recorded ``scan_detector_rotation_flipped`` is inverted (direct ptychography transposes the frames when it is set,
    and these already are) and its angle kept. Source fields (``entry/...``, ``source_metadata``) still describe the
    file as stored. Called once, on a source record (``refuse_second_transpose`` first).
    """
    record = dict(metadata)
    for key, order in ((key, (1, 0)) for key in _DETECTOR_PAIRS):
        value = record.get(key)
        if value is not None and not np.isscalar(value) and len(value) == 2:
            record[key] = _reordered(value, order)
    for key in _AXIS_LISTS:
        value = record.get(key)
        if value is not None and not np.isscalar(value) and len(value) == 4:
            record[key] = _reordered(value, (0, 1, 3, 2))
    if "image_size_x" in record and "image_size_y" in record:
        record["image_size_x"], record["image_size_y"] = record["image_size_y"], record["image_size_x"]
    for key in ("pixel_mask", "valid_pixels"):
        if record.get(key) is not None:
            record[key] = np.ascontiguousarray(np.asarray(record[key]).T)
    # an HDF5 master keeps a saved correction under its attribute name too
    for key in ("hot_pixel_correction", HOT_PIXEL_CORRECTION_ATTRIBUTE):
        correction = record.get(key)
        if isinstance(correction, dict) and correction.get("coordinates_row_col"):
            record[key] = dict(
                correction, coordinates_row_col=[[col, row] for row, col in correction["coordinates_row_col"]]
            )
    if isinstance(record.get("scan_detector_rotation_flipped"), (bool, np.bool_)):
        record["scan_detector_rotation_flipped"] = not bool(record["scan_detector_rotation_flipped"])
    empad = record.get("qem_empad")
    if isinstance(empad, dict) and isinstance(empad.get("background"), dict) and "detector_shape" in metadata:
        # the mean-dark plane subtracted from the float32 values for detector products, one value a stored pixel
        background = empad["background"]
        rows, cols = metadata["detector_shape"]                      # the stored frame's, before the swap above
        plane = np.frombuffer(base64.b64decode(background["values_float32_le"]), "<f4").reshape(rows, cols)
        empad = dict(empad, background=dict(
            background, values_float32_le=base64.b64encode(np.ascontiguousarray(plane.T).tobytes()).decode("ascii")))
        record["qem_empad"] = empad
    disk = record.get(qem_metadata.BRIGHT_FIELD_DISK_KEY)
    text = isinstance(disk, str)
    if text:
        try:
            disk = json.loads(disk)
        except json.JSONDecodeError:
            disk = None
    if isinstance(disk, dict) and isinstance(disk.get("bright_field_center_px"), (list, tuple)):
        row, col = disk["bright_field_center_px"]
        disk = dict(disk, bright_field_center_px=[col, row])
        record[qem_metadata.BRIGHT_FIELD_DISK_KEY] = json.dumps(disk) if text else disk
    if isinstance(record.get("calibration_overrides"), dict):
        record["calibration_overrides"] = _swapped_detector_quantities(record["calibration_overrides"])
    scientific = record.get("scientific_metadata")
    text = isinstance(scientific, str)
    if text:
        try:
            scientific = json.loads(scientific)
        except json.JSONDecodeError:
            scientific = None
    if isinstance(scientific, dict):
        scientific = copy.deepcopy(scientific)
        axes = scientific.get("axes")
        if isinstance(axes, list) and len(axes) == 4:
            # the names stay in place (detector_row, detector_column); each axis takes the other's size and step
            row_axis, col_axis = axes[2], axes[3]
            axes[2] = dict(col_axis, name=row_axis.get("name"))
            axes[3] = dict(row_axis, name=col_axis.get("name"))
        for section in ("electron_microscope", "calibration_overrides"):
            if isinstance(scientific.get(section), dict):
                scientific[section] = _swapped_detector_quantities(scientific[section])
        processing = scientific.get("processing")
        if isinstance(processing, list):
            scientific["processing"] = [*processing, dict(qem_metadata.TRANSPOSE_PROCESSING)]
        record["scientific_metadata"] = json.dumps(scientific) if text else scientific
    record["detector_transpose"] = True
    return record


def refuse_second_transpose(metadata: dict) -> None:
    """Refuse ``detector_transpose`` on a record that says its frames are already transposed.

    A copy saved after a transposed load (``.qem``, or an HDF5 master whose root attributes carry the record) would be
    swapped back to the file's orientation while still saying ``detector_transpose``, and its saved calibration would
    be swapped twice. Checked before anything is read or allocated.
    """
    scientific = metadata.get("scientific_metadata")
    if isinstance(scientific, str):
        try:
            scientific = json.loads(scientific)
        except json.JSONDecodeError:
            scientific = None
    processing = scientific.get("processing") if isinstance(scientific, dict) else None
    declared = isinstance(processing, list) and any(
        isinstance(step, dict) and step.get("operation") == qem_metadata.TRANSPOSE_PROCESSING["operation"]
        for step in processing
    )
    if metadata.get("detector_transpose") or declared:
        raise ValueError(
            f"{metadata.get('source_path', 'This acquisition')} was saved with its detector rows and columns already "
            "swapped (detector_transpose); detector_transpose=True would swap them back. Omit detector_transpose."
        )


def _reordered(value, order):
    """A per-axis value with its entries in ``order``, as a list (a tuple stays a tuple, an array an array)."""
    if isinstance(value, np.ndarray):
        return value[list(order)].copy()
    entries = [value[index] for index in order]
    return tuple(entries) if isinstance(value, tuple) else entries


def _swapped_detector_quantities(quantities: dict) -> dict:
    """Quantities keyed by schema-2 path with the detector row and column pixel sizes exchanged."""
    row_path, col_path = qem_metadata.DETECTOR_PIXEL_SIZE
    result = dict(quantities)
    row, col = result.pop(row_path, None), result.pop(col_path, None)
    if col is not None:
        result[row_path] = col
    if row is not None:
        result[col_path] = row
    return result


def _inflate_frames(data, result, first) -> None:
    """Fill ``result`` with consecutive one-frame gzip chunks of ``data``, inflated on several cores.

    HDF5 inflates one chunk at a time on one core: 3.9 s of the 5.8 s load of
    10000 patterns of 256 x 256. The stored chunk bytes are read as they are
    and zlib, which releases the interpreter lock, inflates them in parallel.
    A chunk whose filter the writer skipped is copied as stored.
    """

    def inflate(offset):
        filter_mask, stored = data.id.read_direct_chunk((first + offset, 0, 0))
        result[offset] = np.frombuffer(stored if filter_mask else zlib.decompress(stored), result.dtype).reshape(result.shape[1:])

    with ThreadPoolExecutor(min(8, os.cpu_count())) as pool:
        list(pool.map(inflate, range(len(result))))


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
    dataset_path,
    backend,
    device,
    verbose,
    hot_pixel_correction,
    info,
    detector_transpose=False,
):
    """Use storage-library reads only where the direct chunk reader cannot apply."""
    if not info.ready or info.scan_shape is None or info.detector_shape is None:
        raise ValueError(f"{info.reason}: {info.action}")
    shape = (*info.scan_shape, *info.detector_shape)
    selected = dataset_path or info.metadata.get("dataset_path")
    with h5py.File(path, "r") as handle:
        if selected is not None:
            datasets = [handle[selected]]
        else:
            # Opened through the master, so external links resolve as h5py does.
            datasets = [handle[f"entry/data/{source.name}"] for source in detector_sources(handle)]
        if not datasets:
            return None
        direct = all(
            data.ndim == 3
            and data.chunks is not None
            and data.chunks[0] == 1
            # 32008 is the registered HDF5 filter id of bitshuffle+LZ4.
            and 32008
            in {
                data.id.get_create_plist().get_filter(index)[0]
                for index in range(data.id.get_create_plist().get_nfilters())
            }
            for data in datasets
        )
        # Keep the existing accelerated bitshuffle/LZ4 reader for its native layout.
        if (
            selected is None
            and direct
            and np.dtype(info.dtype)
            in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32"))
        ):
            return None
        starts = [0]
        for data in datasets:
            if (
                data.ndim not in (3, 4)
                or data.shape[-2:] != shape[2:]
                or data.dtype != np.dtype(info.dtype)
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

        # One gzip chunk per frame in the host's byte order, every chunk written: inflated in parallel.
        frame_gzip = [
            data.chunks == (1, *shape[2:])
            and [data.id.get_create_plist().get_filter(index)[0] for index in range(data.id.get_create_plist().get_nfilters())]
            == [h5py.h5z.FILTER_DEFLATE]
            and data.dtype.isnative
            and data.id.get_num_chunks() == data.shape[0]
            for data in datasets
        ]

        def read_frames(first, stop):
            result = np.empty((stop - first, *shape[2:]), datasets[0].dtype)
            cursor = first
            while cursor < stop:
                index = bisect.bisect_right(starts, cursor) - 1
                end = min(stop, starts[index + 1])
                data = datasets[index]
                local = cursor - starts[index]
                if frame_gzip[index]:
                    _inflate_frames(data, result[cursor - first : end - first], local)
                elif data.ndim == 3:
                    data.read_direct(
                        result,
                        source_sel=np.s_[local : local + end - cursor],
                        dest_sel=np.s_[cursor - first : end - first],
                    )
                else:
                    _read_four_dimensional_frames(
                        data,
                        result[cursor - first : end - first],
                        local,
                        local + end - cursor,
                    )
                cursor = end
            return result

        loaded = load_array_resident(
            shape,
            info.dtype,
            read_frames,
            info.metadata,
            backend=backend,
            device=device,
            pixel_mask=info.pixel_mask,
            hot_pixel_correction=hot_pixel_correction,
            detector_transpose=detector_transpose,
            verbose=verbose,
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


def correction_metadata(record: dict) -> dict:
    """Record how stored detector-mask pixels were handled and what that means for exactness.

    A corrected pixel no longer holds the file's count, so the working data is
    exact (its own codes decode bit for bit) but no longer the file's counts.
    Uncorrected loading keeps every original bit, including detector markers.
    """
    applied = record["applied"]
    if applied:
        policy = f"gpu-{record['method']}-corrected"
    else:
        policy = "preserve-stored-counts"
    exact = not applied
    return dict(
        lossless_exact=exact,
        file_counts_exact=exact,
        detector_mask_policy=policy,
        hot_pixel_correction=record,
        working_counts_exact=True,
        flagged_markers_stored_as_zero=0,
    )
