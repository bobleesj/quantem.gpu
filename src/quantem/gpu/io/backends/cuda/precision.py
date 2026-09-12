"""Packed approximate intensities with bounded scientific-unit GPU queries."""

import bisect
import math
from functools import cache
from pathlib import Path

import cupy as cp
import numpy as np

from quantem.gpu._compact.streamed import StreamedCounts


@cache
def _precision_kernels(device_id: int):
    """Compile scaled-uint16 conversion kernels once per CUDA device."""
    with cp.cuda.Device(device_id):
        module = cp.RawModule(
            code=Path(__file__).with_suffix(".cu").read_text(),
            options=("--std=c++11", "--fmad=false"),
        )
        return {
            name: module.get_function(name)
            for name in (
                "precision_encode",
                "precision_encode_regional",
                "precision_encode_measure",
                "precision_measure",
            )
        }


def encode_scaled_uint16(values, report):
    """Encode float32 values with one CUDA kernel and reference rounding."""
    if not isinstance(values, cp.ndarray):
        raise TypeError("CUDA scaled-uint16 encoding requires a CuPy array.")
    values = cp.ascontiguousarray(values, dtype=cp.float32)
    output = cp.empty(values.shape, dtype=cp.uint16)
    count = int(values.size)
    if count:
        regional = report.get("version") == 2
        kernel = _precision_kernels(cp.cuda.Device().id)[
            "precision_encode_regional" if regional else "precision_encode"
        ]
        scalar = np.float64 if regional else np.float32
        threads = 256
        blocks = min(4096, max(1, (count + threads - 1) // threads))
        kernel(
            (blocks,),
            (threads,),
            (
                values,
                output,
                np.uint64(count),
                scalar(report["scale"]),
                scalar(report["offset"]),
            ),
        )
    return output


def measure_scaled_uint16(values, codes, report):
    """Measure a scaled-uint16 conversion with one GPU reduction pass."""
    if not isinstance(values, cp.ndarray) or not isinstance(codes, cp.ndarray):
        raise TypeError("CUDA precision measurement requires CuPy arrays.")
    values = cp.ascontiguousarray(values, dtype=cp.float32)
    codes = cp.ascontiguousarray(codes, dtype=cp.uint16)
    if values.shape != codes.shape:
        raise ValueError("Precision measurement arrays must have matching shapes.")
    count = int(values.size)
    stats = [
        cp.zeros((), dtype=cp.float64),
        cp.zeros((), dtype=cp.float64),
        cp.zeros((), dtype=cp.uint64),
        cp.zeros((), dtype=cp.uint64),
        cp.zeros((), dtype=cp.uint64),
    ]
    if count:
        kernel = _precision_kernels(cp.cuda.Device().id)["precision_measure"]
        threads = 256
        blocks = min(4096, max(1, (count + threads - 1) // threads))
        shared = threads * (8 + 8 + 8 + 8 + 8)
        kernel(
            (blocks,),
            (threads,),
            (
                values,
                codes,
                np.uint64(count),
                np.float64(report["scale"]),
                np.float64(report["offset"]),
                *stats,
            ),
            shared_mem=shared,
        )
    squared_error, max_error, positive_to_zero, changed, overflow = stats
    report["values"] += count
    report["squared_error"] += float(squared_error.get())
    report["max_abs_error"] = max(
        report["max_abs_error"], float(max_error.get())
    )
    report["positive_to_zero"] += int(positive_to_zero.get())
    report["changed"] += int(changed.get())
    report["overflow"] += int(overflow.get())


def encode_measure_scaled_uint16(values, report, stats):
    """Encode and measure one float32 block in one CUDA pass.

    ``stats`` contains five caller-owned device scalars in report order:
    squared error, maximum absolute error, positive-to-zero count, changed
    count, and overflow count.  Keeping the accumulators on the device avoids
    a host synchronization for every MAPED region.
    """
    if not isinstance(values, cp.ndarray):
        raise TypeError("CUDA scaled-uint16 encoding requires a CuPy array.")
    if len(stats) != 5:
        raise ValueError("stats must contain five CUDA accumulator scalars.")
    values = cp.ascontiguousarray(values, dtype=cp.float32)
    output = cp.empty(values.shape, dtype=cp.uint16)
    count = int(values.size)
    if not count:
        return output
    kernel = _precision_kernels(cp.cuda.Device().id)["precision_encode_measure"]
    threads = 256
    blocks = min(4096, max(1, (count + threads - 1) // threads))
    shared = threads * (8 + 8 + 8 + 8 + 8)
    kernel(
        (blocks,),
        (threads,),
        (
            values,
            output,
            np.uint64(count),
            np.float32(report["scale"]),
            np.float32(report["offset"]),
            np.float64(report["scale"]),
            np.float64(report["offset"]),
            *stats,
        ),
        shared_mem=shared,
    )
    return output


class _ANSIntensityCodes(StreamedCounts):
    """Adapt exact ANS ranges to calibrated intensity queries."""

    block_frames = 4096

    @property
    def _block_count(self):
        return math.ceil(math.prod(self.shape[:2]) / self.block_frames)

    def decode_block_device(self, index):
        first = index * self.block_frames
        return self.decode_scan_range_device(
            first, min(first + self.block_frames, math.prod(self.shape[:2]))
        )

    def extract_diffraction_device(self, acquisition, index):
        return self.decode_scan_range_device(index, index + 1)[0]

    def detector_sum_device(self, mask):
        return cp.concatenate([
            cp.sum(self.decode_block_device(index) * mask,
                   axis=(1, 2), dtype=cp.uint64)
            for index in range(self._block_count)
        ]).reshape(self.shape[:2])


def encode_ans(codes, shape):
    """Retain scaled uint16 codes exactly in the native ANS codec."""
    result = _ANSIntensityCodes(shape, cp.uint16)
    try:
        result.append(codes.view(cp.uint16))
        return result
    except BaseException:
        result.release()
        raise


class PrecisionSource:
    """Own all encoded intensities; decode only bounded query intermediates."""

    _is_gpu_frames = True
    ndim = 4
    dtype = np.dtype("float32")
    capabilities = ()

    def __init__(self, chunks, shape, precision):
        self.parts = chunks
        self.shape = tuple(shape)
        self.scan_shape, self.det_shape = self.shape[:2], self.shape[2:]
        self.n_frames = math.prod(self.scan_shape)
        self.precision = precision
        self._device_id = cp.cuda.Device().id
        self._ends = []
        count = 0
        for part in chunks:
            count += part.shape[1]
            self._ends.append(count)
        from ..._precision import part_reports

        self._reports = part_reports(precision, self._ends)
        self.is_released = False

    @property
    def device(self):
        import torch

        return torch.device("cuda", self._device_id)

    @property
    def nbytes(self):
        return sum(part.nbytes for part in self.parts)

    def numel(self):
        """Return the logical float32 element count for array-style consumers."""
        return math.prod(self.shape)

    def _restore(self, codes, report=None):
        report = self.precision if report is None else report
        if self.precision["storage"] == "float16":
            return codes.view(cp.float16).astype(cp.float32)
        return (
            codes.astype(cp.float64) * report["scale"]
            + report["offset"]
        ).astype(cp.float32)

    def encoded_blocks(self):
        if self.is_released:
            raise RuntimeError("Loaded data was closed; load it again before querying.")
        with cp.cuda.Device(self._device_id):
            for part in self.parts:
                for block in range(part._block_count):
                    codes = part.decode_block_device(block)
                    yield (
                        codes.view(cp.float16)
                        if self.precision["storage"] == "float16"
                        else codes
                    )

    def _blocks(self):
        first = 0
        for encoded in self.encoded_blocks():
            part = bisect.bisect_right(self._ends, first)
            yield self._restore(encoded, self._reports[part])
            first += encoded.shape[0]

    def frame_native(self, index, *, out=None):
        if self.is_released:
            raise RuntimeError("Loaded data was closed; load it again before querying.")
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"Scan index must be in [0, {self.n_frames}); got {index}."
            )
        with cp.cuda.Device(self._device_id):
            part = bisect.bisect_right(self._ends, index)
            start = self._ends[part - 1] if part else 0
            value = self._restore(
                self.parts[part].extract_diffraction_device(0, index - start),
                self._reports[part],
            )
            if out is not None:
                out[...] = value
                return out
            return value

    def _decode_scan_range_torch(self, first, stop):
        """Read a bounded calibrated range into independently owned Torch storage."""
        import torch

        if self.is_released or not 0 <= first < stop <= self.n_frames:
            raise ValueError("Read a nonempty scan range from an open source.")
        with cp.cuda.Device(self._device_id):
            output = cp.empty((stop - first, *self.det_shape), cp.float32)
            start = 0
            for part, end, report in zip(self.parts, self._ends, self._reports):
                low, high = max(first, start), min(stop, end)
                if low < high:
                    for block in range((low - start) // part.block_frames,
                                       (high - start - 1) // part.block_frames + 1):
                        base = start + block * part.block_frames
                        codes = part.decode_block_device(block)
                        left, right = max(low, base), min(high, base + len(codes))
                        output[left - first:right - first] = self._restore(
                            codes[left - base:right - base], report
                        )
                start = end
                if start >= stop:
                    break
            return torch.from_dlpack(output)

    def frame(self, index):
        return self.frame_native(index).get()

    def __getitem__(self, position):
        import torch

        if isinstance(position, (int, np.integer)):
            return torch.from_dlpack(self.frame_native(int(position)))
        if isinstance(position, tuple) and len(position) == 2:
            row, col = position
            if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
                raise IndexError("Scan position lies outside this loaded region.")
            return torch.from_dlpack(self.frame_native(row * self.shape[1] + col))
        raise TypeError(
            "Select one diffraction pattern with source[index] or source[row, col]."
        )

    def mean_dp(self):
        with cp.cuda.Device(self._device_id):
            if self.precision["storage"] == "scaled_uint16":
                total = cp.zeros(self.det_shape, cp.float64)
                for part, report in zip(self.parts, self._reports):
                    total += (
                        part.detector_total_device().astype(cp.float64) * report["scale"]
                        + report["offset"] * part.shape[1]
                    )
                return (total / self.n_frames).astype(cp.float32)
            total = cp.zeros(self.det_shape, cp.float64)
            for values in self._blocks():
                total += cp.sum(values, axis=0, dtype=cp.float64)
            return (total / self.n_frames).astype(cp.float32)

    def masked_sum_native(self, mask, *, out=None):
        with cp.cuda.Device(self._device_id):
            weights = cp.asarray(mask, dtype=cp.float32)
            if weights.shape != self.det_shape:
                raise ValueError(f"Detector mask must have shape {self.det_shape}.")
            if self.precision["storage"] == "scaled_uint16" and bool(
                cp.all((weights == 0) | (weights == 1))
            ):
                selected = int(cp.count_nonzero(weights).get())
                result = cp.empty(self.n_frames, cp.float32)
                first = 0
                binary = weights.astype(cp.uint8)
                for part, report in zip(self.parts, self._reports):
                    codes = part.detector_sum_device(binary).reshape(-1)
                    count = codes.size
                    result[first : first + count] = (
                        codes.astype(cp.float64) * report["scale"]
                        + report["offset"] * selected
                    ).astype(cp.float32)
                    first += count
                result = result.reshape(self.scan_shape)
                if out is not None:
                    out[...] = result
                    return out
                return result
            result = cp.empty(self.n_frames, cp.float32)
            first = 0
            for values in self._blocks():
                count = values.shape[0]
                result[first : first + count] = cp.sum(
                    values * weights, axis=(1, 2), dtype=cp.float32
                )
                first += count
            result = result.reshape(self.scan_shape)
            if out is not None:
                out[...] = result
                return out
            return result

    def masked_sum(self, mask):
        return self.masked_sum_native(mask)

    def reduce_frames(self, indices, reduce="mean"):
        if reduce not in {"mean", "sum", "max"}:
            raise ValueError("Use mean, sum or max for selected diffraction patterns.")
        indices = list(indices)
        if not indices:
            raise ValueError("Select at least one scan position.")
        with cp.cuda.Device(self._device_id):
            total = self.frame_native(indices[0]).astype(cp.float64)
            for index in indices[1:]:
                value = self.frame_native(index)
                if reduce == "max":
                    cp.maximum(total, value, out=total)
                else:
                    total += value
            if reduce == "mean":
                total /= len(indices)
            return total.astype(cp.float32).get()

    def center_of_mass(self, mask=None):
        with cp.cuda.Device(self._device_id):
            weights = (
                cp.ones(self.det_shape, cp.float32)
                if mask is None
                else cp.asarray(mask, dtype=cp.float32)
            )
            rows = cp.arange(self.det_shape[0], dtype=cp.float32)[:, None]
            cols = cp.arange(self.det_shape[1], dtype=cp.float32)[None, :]
            row_result = cp.empty(self.n_frames, cp.float32)
            col_result = cp.empty_like(row_result)
            first = 0
            for values in self._blocks():
                values *= weights
                denominator = cp.maximum(values.sum(axis=(1, 2)), 1e-10)
                count = len(values)
                row_result[first : first + count] = (values * rows).sum(
                    axis=(1, 2)
                ) / denominator
                col_result[first : first + count] = (values * cols).sum(
                    axis=(1, 2)
                ) / denominator
                first += count
            return col_result.reshape(self.scan_shape), row_result.reshape(
                self.scan_shape
            )

    def release(self):
        for part in self.parts:
            part.release()
        self.parts = []
        self.is_released = True
