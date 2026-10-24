"""Compare every HDF5 source count with saved ANS using bounded CUDA decoding."""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import _release_pinned, cp
from quantem.gpu.formats.hdf5.reads import FrameReader
from quantem.gpu.io.hdf5.cuda.decode import decompress_prepared
from quantem.gpu.io.hdf5.verify import supports_counts  # noqa: F401


def verify_counts(master: Path, resident, source_dtype: np.dtype) -> int:
    """Prefetch source chunks and compare all unsigned values, including flagged pixels."""
    scans = math.prod(resident.shape[:2])
    pixels = math.prod(resident.shape[2:])
    different = 0
    pending = None
    with cp.cuda.Device(resident.device), FrameReader(str(master)) as reader:
        # Larger bounded windows amortize read/decode dispatch. Reserve room
        # for raw, saved and decoder scratch while keeping the prior 128 MiB
        # target on a device with little free memory.
        free, _ = cp.cuda.runtime.memGetInfo()
        batch_bytes = max(128 << 20, min(512 << 20, free // 8))
        step = max(1, min(8192, batch_bytes // (pixels * source_dtype.itemsize)))
        if step >= 512:
            step = step // 512 * 512
        bounds = [(first, min(first + step, scans)) for first in range(0, scans, step)]

        def prepare(bounds):
            with cp.cuda.Device(resident.device):
                return reader.prepare(np.arange(*bounds))

        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                pending = pool.submit(prepare, bounds[0])
                for index, (first, stop) in enumerate(bounds):
                    prepared, pending = pending.result(), None
                    if index + 1 < len(bounds):
                        pending = pool.submit(prepare, bounds[index + 1])
                    # This decoder synchronizes before returning its pinned
                    # source buffer; no queued DMA can outlive that buffer.
                    raw = saved = None
                    try:
                        raw = decompress_prepared(prepared, batch_bytes_target=batch_bytes)
                        saved = resident.decode_scan_range_device(first, stop)
                        # Unsigned promotion preserves high source bits.
                        different += int(cp.count_nonzero(raw.reshape(saved.shape) != saved).get())
                    finally:
                        raw = saved = prepared = None
            finally:
                if pending is not None:
                    try:
                        prepared = pending.result()
                    except BaseException:  # noqa: BLE001, S110
                        pass  # Preserve the foreground failure.
                    else:
                        _release_pinned(prepared["read_buffer"])
    return different
