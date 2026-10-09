"""Packed approximate intensities with bounded scientific-unit GPU queries."""

import bisect
import math
import struct
from pathlib import Path

import numpy as np
import torch

from quantem.gpu.device.cuda_runtime import cp, cuda_module
from quantem.gpu.formats.precision import part_reports
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.queries import DetectorQueries

_SOURCE = (Path(__file__).with_name("kernels") / "precision.cu").read_text()
_NAMES = (
    "precision_encode",
    "precision_encode_regional",
    "precision_encode_measure_regional",
    "precision_measure",
)


def _kernel(name: str):
    """Return scaled-uint16 conversion kernel ``name`` compiled for the current device."""
    return cuda_module(_SOURCE, _NAMES, ("--std=c++11", "--fmad=false"))[name]


def _regional_calibration(report):
    """Split scale and offset into the float-float pairs the exact encoder uses.

    Mirrors the Metal ``_parameters``: both scalars are rescaled by 2^exponent
    so the largest intensity sits near 1, then each is written as a float32
    high part plus a float32 remainder. Together the pair carries the float64
    calibration exactly enough for the kernel's midpoint test to reproduce
    ``nearbyint((double(value) - offset) / scale)`` code for code.
    """
    maximum = max(abs(report["intensity_min"]), abs(report["intensity_max"]))
    exponent = -math.frexp(maximum)[1] if maximum else 0
    pairs = []
    for value in (report["scale"], report["offset"]):
        scaled = math.ldexp(value, exponent)
        high = struct.unpack("f", struct.pack("f", scaled))[0]
        pairs += [np.float32(high), np.float32(scaled - high)]
    return *pairs, np.int32(exponent)


def encode_scaled_uint16(values, report):
    """Encode float32 values with one CUDA kernel and reference rounding."""
    if not isinstance(values, cp.ndarray):
        raise TypeError("CUDA scaled-uint16 encoding requires a CuPy array.")
    values = cp.ascontiguousarray(values, dtype=cp.float32)
    output = cp.empty(values.shape, dtype=cp.uint16)
    count = int(values.size)
    if count:
        regional = report.get("version") == 2
        kernel = _kernel("precision_encode_regional" if regional else "precision_encode")
        if regional:
            calibration = _regional_calibration(report)
        else:
            calibration = (np.float32(report["scale"]), np.float32(report["offset"]))
        threads = 256
        blocks = min(4096, max(1, (count + threads - 1) // threads))
        kernel(
            (blocks,),
            (threads,),
            (values, output, np.uint64(count), *calibration),
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
        kernel = _kernel("precision_measure")
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


def encode_measure_regional(values, report):
    """Encode one region with exact float64-equivalent rounding and measure it in the same pass.

    The region's codes depend on its own min/max calibration, so the range
    pass must still precede this one; everything after that (encode, restore,
    error statistics) reads the region once and ends in a single host copy of
    the five accumulators instead of five.
    """
    values = cp.ascontiguousarray(values, dtype=cp.float32)
    output = cp.empty(values.shape, dtype=cp.uint16)
    count = int(values.size)
    stats = cp.zeros(5, dtype=cp.float64)
    counters = stats[2:].view(cp.uint64)
    if count:
        kernel = _kernel("precision_encode_measure_regional")
        threads = 256
        blocks = min(4096, max(1, (count + threads - 1) // threads))
        kernel(
            (blocks,),
            (threads,),
            (
                values,
                output,
                np.uint64(count),
                np.float64(report["scale"]),
                np.float64(report["offset"]),
                *_regional_calibration(report),
                stats[0:1],
                stats[1:2],
                counters[0:1],
                counters[1:2],
                counters[2:3],
            ),
            shared_mem=threads * 40,
        )
    host = stats.get()
    report["values"] += count
    report["squared_error"] += float(host[0])
    report["max_abs_error"] = max(report["max_abs_error"], float(host[1]))
    host_counters = host[2:].view(np.uint64)
    report["positive_to_zero"] += int(host_counters[0])
    report["changed"] += int(host_counters[1])
    report["overflow"] += int(host_counters[2])
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
        # A failed encode must return its partial chunks to the GPU at once.
        result.release()
        raise


class PrecisionSource(DetectorQueries):
    """Own all encoded intensities; decode only bounded query intermediates."""

    ndim = 4
    dtype = np.dtype("float32")

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
        self._reports = part_reports(precision, self._ends)
        self.is_released = False

    @property
    def device(self):
        return torch.device("cuda", self._device_id)

    @property
    def nbytes(self):
        return sum(part.nbytes for part in self.parts)

    def numel(self):
        """Return the logical float32 element count for array-style consumers."""
        return math.prod(self.shape)

    def _restore(self, codes, report):
        """Convert stored codes back to float32 intensities with the part's calibration."""
        return (
            codes.astype(cp.float64) * report["scale"]
            + report["offset"]
        ).astype(cp.float32)

    def encoded_blocks(self):
        """Yield every decoded block of stored codes, part by part, in scan order."""
        if self.is_released:
            raise RuntimeError("Loaded data was closed; load it again before querying.")
        with cp.cuda.Device(self._device_id):
            for part in self.parts:
                for block in range(part._block_count):
                    yield part.decode_block_device(block)

    def _blocks(self):
        """Yield calibrated float32 blocks; each part has its own scale and offset."""
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
            total = cp.zeros(self.det_shape, cp.float64)
            for part, report in zip(self.parts, self._reports):
                total += (
                    part.detector_total_device().astype(cp.float64) * report["scale"]
                    + report["offset"] * part.shape[1]
                )
            return (total / self.n_frames).astype(cp.float32)

    def masked_sum_native(self, mask, *, out=None):
        with cp.cuda.Device(self._device_id):
            weights = cp.asarray(mask, dtype=cp.float32)
            if weights.shape != self.det_shape:
                raise ValueError(f"Detector mask must have shape {self.det_shape}.")
            if bool(cp.all((weights == 0) | (weights == 1))):
                # A binary mask sums stored codes exactly, then calibrates once per scan position.
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
            else:
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

    def masked_sum_exact_native(self, mask, *, out=None):
        """Restored intensities are float32, so there is no exact integer image on any output."""
        return self.masked_sum_exact(mask)

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
