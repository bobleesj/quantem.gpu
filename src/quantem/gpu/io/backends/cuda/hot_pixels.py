"""CUDA hot-pixel correction over bounded native-count batches."""

from __future__ import annotations

from functools import cache

import numpy as np

from quantem.gpu.io._hot_pixels import hot_pixel_record


_CUDA_SOURCE = r"""
template <typename T>
__device__ void correct_one(
    T* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long item,
    bool use_median
) {
    int bad_slot = item % bad_count;
    unsigned long long frame = item / bad_count;
    int pixel = bad[bad_slot];
    if (!use_median) {
        frames[frame * height * width + pixel] = T(0);
        return;
    }
    int row = pixel / width, column = pixel % width;
    unsigned int values[8];
    int count = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            int rr = row + dr, cc = column + dc;
            if ((dr == 0 && dc == 0) || rr < 0 || rr >= height ||
                cc < 0 || cc >= width) continue;
            int neighbor = rr * width + cc;
            if (!valid[neighbor]) continue;
            values[count++] = frames[frame * height * width + neighbor];
        }
    }
    for (int i = 1; i < count; ++i) {
        unsigned int value = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = value;
    }
    unsigned int result = 0;
    if (count & 1) result = values[count / 2];
    else if (count) result = (values[count / 2 - 1] + values[count / 2]) / 2;
    frames[frame * height * width + pixel] = T(result);
}

extern "C" __global__ void hot_median_u8(
    unsigned char* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, true);
}
extern "C" __global__ void hot_median_u16(
    unsigned short* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, true);
}
extern "C" __global__ void hot_zero_u8(
    unsigned char* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, false);
}
extern "C" __global__ void hot_zero_u16(
    unsigned short* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, false);
}
"""


@cache
def _kernels(device: int):
    import cupy as cp

    with cp.cuda.Device(device):
        module = cp.RawModule(code=_CUDA_SOURCE, options=("--std=c++17",))
        return {
            name: module.get_function(name)
            for name in (
                "hot_median_u8",
                "hot_median_u16",
                "hot_zero_u8",
                "hot_zero_u16",
            )
        }


class CUDAHotPixelCorrector:
    """Reuse detector correction metadata across streamed CUDA batches."""

    def __init__(self, pixel_mask, method: str):
        import cupy as cp

        self.mask = None if pixel_mask is None else np.asarray(pixel_mask)
        self.method = method
        self.record = hot_pixel_record(self.mask, method, backend="cuda")
        self.device = cp.cuda.Device().id
        self.valid = self.bad = None
        if self.record["applied"]:
            valid = self.mask == 0
            self.valid = cp.asarray(valid.reshape(-1), dtype=cp.uint8)
            self.bad = cp.asarray(np.flatnonzero(~valid), dtype=cp.int32)

    def apply(self, values) -> None:
        """Correct one contiguous ``(scan, detector_row, detector_col)`` batch."""
        if not self.record["applied"]:
            return
        import cupy as cp

        if not isinstance(values, cp.ndarray) or values.dtype not in (
            cp.dtype("uint8"),
            cp.dtype("uint16"),
        ):
            raise TypeError(
                "CUDA hot-pixel correction requires native uint8/uint16 counts."
            )
        if not values.flags.c_contiguous or tuple(values.shape[-2:]) != tuple(
            self.mask.shape
        ):
            raise ValueError(
                "CUDA hot-pixel correction requires contiguous native detector frames."
            )
        total = int(np.prod(values.shape[:-2])) * int(self.bad.size)
        kernel_name = f"hot_{self.method}_u{values.dtype.itemsize * 8}"
        kernel = _kernels(self.device)[kernel_name]
        kernel(
            ((total + 255) // 256,),
            (256,),
            (
                values,
                self.valid,
                self.bad,
                np.int32(self.bad.size),
                np.int32(values.shape[-2]),
                np.int32(values.shape[-1]),
                np.uint64(total),
            ),
        )

    def close(self) -> None:
        self.valid = self.bad = None
