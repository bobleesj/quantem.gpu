"""Metal hot-pixel correction over bounded native-count batches."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu.io._hot_pixels import hot_pixel_record
from quantem.gpu.io.backends.mps.packed import (
    _allocate_shared,
    _buffer_view,
    _complete,
    _metal_module,
    _release,
)


@cache
def _runtime():
    metal = _metal_module()
    device = metal.MTLCreateSystemDefaultDevice()
    options = metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(False)
    source = Path(__file__).with_name("kernels").joinpath("hot_pixels.msl").read_text()
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        raise RuntimeError(f"Metal hot-pixel kernel compilation failed: {error}")
    pipelines = {}
    for method in ("median", "zero"):
        function = library.newFunctionWithName_(f"hot_{method}")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None:
            raise RuntimeError(f"Metal hot-pixel {method} pipeline failed: {error}")
        pipelines[method] = pipeline
    return device, metal, device.newCommandQueue(), pipelines


def _upload(device, metal, values: np.ndarray, label: str):
    values = np.ascontiguousarray(values)
    buffer = _allocate_shared(device, metal, max(1, values.nbytes), label)
    if values.nbytes:
        _buffer_view(buffer, values.nbytes)[:] = memoryview(values).cast("B")
    return buffer


class MPSHotPixelCorrector:
    """Reuse detector correction metadata across streamed Metal batches."""

    def __init__(self, pixel_mask, method: str):
        self.mask = None if pixel_mask is None else np.asarray(pixel_mask)
        self.method = method
        self.record = hot_pixel_record(self.mask, method, backend="mps")
        self.device, self.metal, self.queue, self.pipelines = _runtime()
        self.valid = self.bad = None
        if self.record["applied"]:
            valid = self.mask == 0
            self.valid = _upload(
                self.device,
                self.metal,
                valid.astype(np.uint8).reshape(-1),
                "hot-pixel valid mask",
            )
            self.bad = _upload(
                self.device,
                self.metal,
                np.flatnonzero(~valid).astype(np.int32),
                "hot-pixel coordinates",
            )

    def apply(self, values) -> None:
        """Correct one Metal ``(scan, detector_row, detector_col)`` batch."""
        if not self.record["applied"]:
            return
        dtype = np.dtype(values.dtype)
        if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise TypeError(
                "Metal hot-pixel correction requires native uint8/uint16 counts."
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
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipelines[self.method])
        for index, buffer in enumerate((values._mtl, self.valid, self.bad)):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.setBytes_length_atIndex_(parameters.tobytes(), parameters.nbytes, 3)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(total, 1, 1),
            self.metal.MTLSizeMake(128, 1, 1),
        )
        encoder.endEncoding()
        _complete(command, "hot-pixel correction")

    def close(self) -> None:
        _release(self.valid)
        _release(self.bad)
        self.valid = self.bad = None
