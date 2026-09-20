"""Bounded float bit-lane decode with parallel literal access."""

from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu._compact import streamed


@cache
def _kernels(device):
    import cupy as cp

    with cp.cuda.Device(device):
        source = (
            Path(streamed.__file__)
            .with_name("kernels")
            .joinpath("streamed.cu")
            .read_text()
        )
        # The codec reader is shared; unrelated count-product kernels are omitted.
        source = source[: source.index('extern "C" __global__ void sc_encode')]
        source += Path(__file__).with_suffix(".cu").read_text()
        module = cp.RawModule(code=source, options=("--std=c++17", "--fmad=false"))
        return tuple(
            module.get_function("float_ans_" + name)
            for name in ("direct", "entropy", "selected_entropy", "detector")
        )


class CUDAFloatLanes(streamed.StreamedCounts):
    """Reuse count tables/storage without exposing count-valued float products."""

    def decode_scan_range_device(self, first, stop, *, errors=None):
        import cupy as cp

        with cp.cuda.Device(self.device):
            direct, entropy = _kernels(self.device)[:2]
            output = cp.empty((stop - first, 128, 256), cp.uint16)
            errors = cp.zeros(1, cp.uint32)
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
                    ((int(count) * 32768 + 255) // 256,),
                    (256,),
                    (*chunk.arrays, target, local, count),
                )
                entropy(
                    (256,),
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
            if int(errors.get()[0]):
                raise ValueError(
                    "Float ANS stream failed reconstruction; recopy the source."
                )
            return output

    def float_detector(self, mask, background=None):
        """Read literal streams directly and decode only selected entropy lanes."""
        import cupy as cp

        with cp.cuda.Device(self.device):
            entropy, reduce = _kernels(self.device)[2:]
            selected = cp.asarray(mask, dtype=cp.uint8)
            scratch = cp.empty((max(c.scans for c in self.chunks), 32768), cp.uint16)
            output = cp.empty(self.shape[:2], cp.float32)
            errors = cp.zeros(1, cp.uint32)
            for chunk in self.chunks:
                entropy(
                    (256,),
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
