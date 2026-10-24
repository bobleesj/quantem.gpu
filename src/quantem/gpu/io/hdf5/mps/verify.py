"""Bounded full-count comparison of original HDF5 and saved ANS on Metal."""

import math
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, SimpleQueue

import numpy as np

from quantem.gpu.device.metal_runtime import (
    allocate_shared,
    complete_command,
    metal_module,
    metal_pipelines,
    metal_queue,
    numpy_view,
    release_buffer,
)
from quantem.gpu.formats.hdf5.reads import FrameReader
from quantem.gpu.io.hdf5.mps.decode import MPSDecompressor
from quantem.gpu.io.hdf5.verify import supports_counts  # noqa: F401

_SOURCE = (Path(__file__).parent / "kernels" / "verify_counts.msl").read_text()


def verify_counts(master: Path, resident, source_dtype: np.dtype) -> int:
    """Compare every source value with saved counts, including high bytes and mask pixels.

    A single worker reads and decodes the next source batch while the caller
    decodes ANS and compares the preceding batch. At most two source buffers
    and one saved-count buffer are live. No correction or narrowing is applied
    to the source. Only block mismatch counts cross back to the host.
    """
    scans = math.prod(resident.shape[:2])
    pixels = math.prod(resident.shape[2:])
    step = max(1, min(8192, (128 << 20) // (pixels * source_dtype.itemsize)))
    if step >= 512:
        step = step // 512 * 512
    bounds = [(first, min(first + step, scans)) for first in range(0, scans, step)]
    idle = SimpleQueue()
    decoder = None
    decoded = partials = None
    pending = None
    reader = FrameReader(str(master))

    def read(bounds):
        nonlocal decoder
        prepared = reader.prepare(np.arange(*bounds))
        compressed = max(1, len(prepared["read_buffer"]))
        if decoder is None or compressed > decoder.max_compressed_bytes:
            if decoder is not None:
                decoder.free()
            frame_bytes = int(prepared["frame_bytes"])
            decoder = MPSDecompressor(
                max_compressed_bytes=2 * compressed, max_frames=step,
                frame_bytes=frame_bytes, n_blocks_per_frame=(frame_bytes + 8191) // 8192,
            )
        try:
            buffer = idle.get_nowait()
        except Empty:
            buffer = allocate_shared(step * pixels * source_dtype.itemsize, "verification source")
        try:
            return decoder.load_prepared_frames(prepared, out_buffer=buffer)
        except BaseException as error:
            release_buffer(buffer)
            buffer = None
            # Futures retain exception frames. Drop their borrowed buffer
            # references while preserving the traceback locations and error.
            traceback.clear_frames(error.__traceback__)
            raise

    def recycle(raw):
        buffer, raw._mtl = raw._mtl, None
        idle.put(buffer)

    different = 0
    try:
        pipeline = metal_pipelines(_SOURCE, ("verify_counts",), fast_math=False)["verify_counts"]
        decoded = allocate_shared(step * pixels * resident.dtype.itemsize, "verification saved counts")
        partials = allocate_shared(((step * pixels + 4095) // 4096) * 4, "verification differences")
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                pending = pool.submit(read, bounds[0])
                for index, (first, stop) in enumerate(bounds):
                    raw = pending.result()
                    pending = None
                    try:
                        if index + 1 < len(bounds):
                            pending = pool.submit(read, bounds[index + 1])
                        count = (stop - first) * pixels
                        groups = (count + 4095) // 4096
                        command = metal_queue().commandBuffer()
                        resident._encode_scan_range_into(command, first, stop, decoded)
                        encoder = command.computeCommandEncoder()
                        encoder.setComputePipelineState_(pipeline)
                        for binding, buffer in enumerate((raw._mtl, decoded, partials)):
                            encoder.setBuffer_offset_atIndex_(buffer, 0, binding)
                        parameters = np.asarray(
                            [count, source_dtype.itemsize, resident.dtype.itemsize, 0], np.uint32
                        ).tobytes()
                        encoder.setBytes_length_atIndex_(parameters, len(parameters), 3)
                        size = metal_module().MTLSizeMake
                        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                            size(groups, 1, 1), size(256, 1, 1)
                        )
                        encoder.endEncoding()
                        complete_command(command, "full source count verification")
                        resident._check_errors()
                        different += int(numpy_view(partials, np.uint32, groups).sum(dtype=np.uint64))
                    finally:
                        command = encoder = buffer = None
                        recycle(raw)
            finally:
                if pending is not None:
                    try:
                        recycle(pending.result())
                    except BaseException:  # noqa: BLE001, S110
                        pass  # Preserve the foreground failure; failed reads free their buffer.
    finally:
        reader.close()
        if decoder is not None:
            decoder.free()
        release_buffer(decoded)
        release_buffer(partials)
        decoded = partials = None
        while not idle.empty():
            release_buffer(idle.get_nowait())
    return different
