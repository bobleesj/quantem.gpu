"""Bounded float bit-lane decode with parallel literal access."""

from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cuda_module
from quantem.gpu.resident.cuda import counts

_KERNELS = Path(__file__).with_name("kernels")
# The count codec's stream reader, without its unrelated count-product kernels.
_READER_SOURCE = (_KERNELS / "streamed.cu").read_text().split('extern "C" __global__ void sc_encode')[0]
_FLOAT_SOURCE = (_KERNELS / "float_ans.cu").read_text()


def _kernels(device, lanes):
    """Return the float bit-lane kernels for ``device``; the lane count is a compile-time constant."""
    import cupy as cp

    source = _READER_SOURCE + f"\n#define FLOAT_LANES {lanes}u\n" + _FLOAT_SOURCE
    names = tuple("float_ans_" + name for name in ("direct", "entropy", "selected_entropy", "detector"))
    with cp.cuda.Device(device):
        functions = cuda_module(source, names, ("--std=c++17", "--fmad=false"))
    return tuple(functions[name] for name in names)


class CUDAFloatLanes(counts.StreamedCounts):
    """Reuse count tables/storage without exposing count-valued float products."""

    def decode_scan_range_device(self, first, stop, *, errors=None):
        """Decode scans ``[first, stop)`` as uint16 bit lanes.

        ``errors`` (one uint32 on the device) collects stream failures for the
        caller to check after several decodes, as for count residents; without
        it this decode checks its own and raises.
        """
        import cupy as cp

        owns_errors = errors is None
        with cp.cuda.Device(self.device):
            lanes = int(np.prod(self.shape[2:]))
            direct, entropy = _kernels(self.device, lanes)[:2]
            output = cp.empty((stop - first, *self.shape[2:]), cp.uint16)
            errors = counts.error_flags(errors, self.device)
            for chunk in self.chunks:
                begin, end = (
                    max(first, chunk.first),
                    min(stop, chunk.first + chunk.scans),
                )
                if begin >= end:
                    continue
                target = output[begin - first : end - first]
                local, count = np.uint32(begin - chunk.first), np.uint32(end - begin)
                direct(
                    ((int(count) * lanes + 255) // 256,),
                    (256,),
                    (*chunk.arrays, target, local, count),
                )
                entropy(
                    ((lanes + 127) // 128,),
                    (128,),
                    (
                        *chunk.arrays,
                        self.decoding,
                        target,
                        errors,
                        np.uint32(chunk.scans),
                        local,
                        count,
                    ),
                )
            if owns_errors and int(errors.get()[0]):
                raise ValueError(
                    "Float ANS stream failed reconstruction; recopy the source."
                )
            return output

    def float_detector(self, mask, background=None):
        """Read literal streams directly and decode only selected entropy lanes."""
        import cupy as cp

        with cp.cuda.Device(self.device):
            lanes = int(np.prod(self.shape[2:]))
            entropy, reduce = _kernels(self.device, lanes)[2:]
            selected = cp.asarray(mask, dtype=cp.uint8)
            scratch = cp.empty((max(chunk.scans for chunk in self.chunks), lanes), cp.uint16)
            output = cp.empty(self.shape[:2], cp.float32)
            errors = cp.zeros(1, cp.uint32)
            for chunk in self.chunks:
                entropy(
                    ((lanes + 127) // 128,),
                    (128,),
                    (
                        *chunk.arrays,
                        self.decoding,
                        scratch,
                        errors,
                        selected,
                        np.uint32(chunk.scans),
                    ),
                )
                reduce(
                    (chunk.scans,),
                    (128,),
                    (
                        *chunk.arrays,
                        scratch,
                        selected,
                        output if background is None else background,
                        output,
                        np.uint32(chunk.first),
                        np.uint32(background is not None),
                    ),
                )
            if int(errors.get()[0]):
                raise ValueError("Float ANS detector stream failed reconstruction.")
            return output
