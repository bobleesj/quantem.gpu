"""Metal hot-pixel correction over bounded native-count batches."""

from pathlib import Path

import numpy as np

from quantem.gpu.device.metal_runtime import (
    complete_command,
    metal_module,
    metal_pipelines,
    metal_queue,
    release_buffer,
    upload_shared,
)
from quantem.gpu.resident.hot_pixels import hot_pixel_record

_SOURCE = (Path(__file__).with_name("kernels") / "hot_pixels.msl").read_text()


class MPSHotPixelCorrector:
    """Reuse detector correction metadata across streamed Metal batches."""

    def __init__(self, pixel_mask, method: str):
        self.mask = None if pixel_mask is None else np.asarray(pixel_mask)
        self.method = method
        self.record = hot_pixel_record(self.mask, method, backend="mps")
        self.valid = self.bad = None
        if self.record["applied"]:
            valid = self.mask == 0
            self.valid = upload_shared(valid.astype(np.uint8).reshape(-1), "hot-pixel valid mask")
            self.bad = upload_shared(np.flatnonzero(~valid).astype(np.int32), "hot-pixel coordinates")

    def apply(self, values) -> None:
        """Correct one Metal ``(scan, detector_row, detector_col)`` batch.

        uint32 batches are Arina counts corrected before they are stored as
        uint16: the median reads only valid neighbors, so a 0xFFFFFFFF flagged
        pixel never enters a replacement value.
        """
        if not self.record["applied"]:
            return
        dtype = np.dtype(values.dtype)
        if dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("uint32")):
            raise TypeError(
                "Metal hot-pixel correction requires native uint8/uint16/uint32 counts."
            )
        if tuple(values.shape[-2:]) != tuple(self.mask.shape):
            raise ValueError(
                "Metal hot-pixel correction requires native detector frames."
            )
        scans = int(np.prod(values.shape[:-2]))
        bad_count = int(self.record["pixel_count"])
        total = scans * bad_count
        parameters = np.asarray(
            [bad_count, values.shape[-2], values.shape[-1], total, dtype.itemsize],
            dtype=np.uint64,
        )
        metal = metal_module()
        pipeline = metal_pipelines(_SOURCE, ("hot_median", "hot_zero"), fast_math=False)[f"hot_{self.method}"]
        command = metal_queue().commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate((values._mtl, self.valid, self.bad)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBytes_length_atIndex_(parameters.tobytes(), parameters.nbytes, 3)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            metal.MTLSizeMake(total, 1, 1),
            metal.MTLSizeMake(128, 1, 1),
        )
        encoder.endEncoding()
        complete_command(command, "hot-pixel correction")

    def close(self) -> None:
        """Release the mask buffers; PyObjC never frees them on collection."""
        release_buffer(self.valid)
        release_buffer(self.bad)
        self.valid = self.bad = None
