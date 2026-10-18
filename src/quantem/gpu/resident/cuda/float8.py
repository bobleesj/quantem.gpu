"""Original FP8 byte ownership and float32 calculations on CUDA."""

from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_module
from quantem.gpu.formats.float8 import FORMAT_CODES, torch_dtype

SOURCE = (Path(__file__).with_name("kernels") / "float8.cu").read_text()


class Float8Array:
    """Own native FP8 bytes where CuPy has no corresponding numeric dtype."""

    def __init__(self, values, dtype):
        self.values = values
        self.dtype = np.dtype(dtype)

    @property
    def shape(self):
        return self.values.shape

    @property
    def device(self):
        return self.values.device

    @property
    def nbytes(self):
        return self.values.nbytes

    def reshape(self, *shape):
        return Float8Array(self.values.reshape(*shape), self.dtype)

    def get(self):
        """Copy original bits to a named FP8 NumPy array."""
        return self.values.get().view(self.dtype)

    def to_torch(self):
        """Share the original bytes as a typed Torch CUDA tensor."""
        import torch

        return torch.from_dlpack(self.values).view(torch_dtype(self.dtype))

    def release(self):
        """Release this wrapper's reference; existing tensor views stay valid."""
        self.values = None


def float32_view(values, dtype):
    """Interpret bytes or decoded words on device, retaining the original bits."""
    import cupy as cp

    if isinstance(values, Float8Array):
        values = values.values
    result = cp.empty(values.shape, cp.float32)
    invalid = cp.zeros(1, cp.uint32)
    cuda_module(SOURCE, ("float8_decode",), ("--std=c++17",))["float8_decode"](
        ((values.size + 255) // 256,),
        (256,),
        (
            values,
            result,
            invalid,
            np.uint64(values.size),
            np.uint32(values.dtype.itemsize),
            np.uint32(FORMAT_CODES[str(dtype)]),
        ),
    )
    if values.dtype.itemsize == 2 and bool(invalid[0]):
        raise ValueError("Decoded FP8 word exceeds one byte; recopy the QEM file.")
    return result


def decode_bytes(lanes, first: int, stop: int):
    """Decode and validate original byte lanes with one completion check."""
    import cupy as cp

    errors = cp.zeros(1, cp.uint32)
    words = lanes.decode_scan_range_device(first, stop, errors=errors)
    result = cp.empty(words.shape, cp.uint8)
    cuda_module(SOURCE, ("float8_bytes",), ("--std=c++17",))["float8_bytes"](
        ((words.size + 255) // 256,),
        (256,),
        (words, result, errors, np.uint64(words.size)),
    )
    if int(errors.get()[0]):
        raise ValueError("FP8 byte stream failed validation; recopy the QEM file.")
    return result


def reduce_patterns(lanes, indices, dtype, valid, mode):
    """Reduce decoded words directly, without a dense float64 window."""
    import cupy as cp

    kernels = cuda_module(
        SOURCE,
        ("float8_pattern_total", "float8_pattern_finish"),
        ("--std=c++17", "--ftz=false"),
    )
    flags = np.zeros(int(np.prod(lanes.shape[:2])), np.uint8)
    flags[indices] = 1
    selected = cp.asarray(flags)
    validity = cp.asarray(valid, dtype=cp.uint8)
    errors = cp.zeros(1, cp.uint32)
    total = cp.full(lanes.shape[2:], -cp.inf if mode == "max" else 0, cp.float64)
    for chunk in lanes.chunks:
        local = np.flatnonzero(flags[chunk.first : chunk.first + chunk.scans])
        if not local.size:
            continue
        first, stop = chunk.first + int(local[0]), chunk.first + int(local[-1]) + 1
        words = lanes.decode_scan_range_device(first, stop, errors=errors)
        kernels["float8_pattern_total"](
            ((total.size + 127) // 128,),
            (128,),
            (
                words,
                total,
                selected,
                validity,
                errors,
                np.uint32(stop - first),
                np.uint32(total.size),
                np.uint32(first),
                np.uint32(FORMAT_CODES[str(dtype)]),
                np.uint32(mode == "max"),
            ),
        )
        del words
    result = cp.empty(total.shape, cp.float32)
    kernels["float8_pattern_finish"](
        ((result.size + 255) // 256,),
        (256,),
        (
            total,
            result,
            np.uint32(result.size),
            np.uint32(len(indices) if mode == "mean" else 1),
        ),
    )
    if int(errors.get()[0]):
        raise ValueError("FP8 pattern stream failed validation; recopy the QEM file.")
    return result
