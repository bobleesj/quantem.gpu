"""CUDA hot-pixel correction over bounded native-count batches."""

from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_module
from quantem.gpu.resident.hot_pixels import hot_pixel_record, median_plan

_SOURCE = (Path(__file__).with_name("kernels") / "hot_pixels.cu").read_text()
_NAMES = ("hot_median_u8", "hot_median_u16", "hot_zero_u8", "hot_zero_u16", "hot_median_u32", "hot_zero_u32")


class CUDAHotPixelCorrector:
    """Reuse detector correction metadata across streamed CUDA batches."""

    def __init__(self, pixel_mask, method: str):
        import cupy as cp

        self.mask = None if pixel_mask is None else np.asarray(pixel_mask)
        self.method = method
        self.record = hot_pixel_record(self.mask, method, backend="cuda")
        self.device = cp.cuda.Device().id
        self.offsets = self.neighbors = self.bad = None
        if self.record["applied"]:
            plan = median_plan(self.mask)
            self.offsets = cp.asarray(plan.offsets)
            self.neighbors = cp.asarray(plan.neighbors)
            self.bad = cp.asarray(plan.bad)

    def apply(self, values) -> None:
        """Correct one contiguous ``(scan, detector_row, detector_col)`` batch."""
        if not self.record["applied"]:
            return
        import cupy as cp

        if not isinstance(values, cp.ndarray) or values.dtype not in (
            cp.dtype("uint8"),
            cp.dtype("uint16"),
            cp.dtype("uint32"),
        ):
            raise TypeError(
                "CUDA hot-pixel correction requires native uint8/uint16/uint32 counts."
            )
        if not values.flags.c_contiguous or tuple(values.shape[-2:]) != tuple(
            self.mask.shape
        ):
            raise ValueError(
                "CUDA hot-pixel correction requires contiguous native detector frames."
            )
        total = int(np.prod(values.shape[:-2])) * int(self.bad.size)
        kernel_name = f"hot_{self.method}_u{values.dtype.itemsize * 8}"
        with cp.cuda.Device(self.device):
            cuda_module(_SOURCE, _NAMES, ("--std=c++17",))[kernel_name](
                ((total + 255) // 256,),
                (256,),
                (
                    values,
                    self.offsets,
                    self.neighbors,
                    self.bad,
                    np.int32(self.bad.size),
                    np.int32(values.shape[-2]),
                    np.int32(values.shape[-1]),
                    np.uint64(total),
                ),
            )

    def close(self) -> None:
        """Drop the device copies of the mask between acquisitions."""
        self.offsets = self.neighbors = self.bad = None
