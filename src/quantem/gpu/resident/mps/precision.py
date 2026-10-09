"""Metal precision conversion and direct queries over resident encoded streams."""

import bisect
import math
import struct
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from quantem.gpu.device.metal_runtime import (
    buffer_view,
    complete_command,
    metal_module,
    metal_pipelines,
    metal_queue,
    shared_array,
    tensor_buffer,
)
from quantem.gpu.formats.precision import part_reports
from quantem.gpu.resident.mps.arrays import MetalArray
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.queries import DetectorQueries

_KERNELS = Path(__file__).parent / "kernels"
_SOURCE = "#define QUANTEM_PRECISION_ANS_MEAN 1\n" + "\n".join(
    (_KERNELS / name).read_text() for name in ("streamed_counts.msl", "precision.msl")
)
_NAMES = (
    "copy", "restore", "encode", "encode_measure", "range", "range_reduce", "measure",
    "frame", "unpacked", "detector", "mean", "reduce", "range_read", "divide",
    "ans_mean", "measure_reduce",
)


def _pipelines() -> dict:
    """Return the precision pipelines by short name, compiled once per process.

    Every block of a conversion dispatches several kernels; recompiling per
    block would dominate the conversion time.
    """
    pipelines = metal_pipelines(_SOURCE, tuple(f"precision_{name}" for name in _NAMES), fast_math=False)
    return {name: pipelines[f"precision_{name}"] for name in _NAMES}


def upload(values):
    """Transfer bytes only; scientific conversion happens in Metal kernels."""
    if isinstance(values, MetalArray):
        return values
    if isinstance(values, torch.Tensor):
        if values.device.type == "mps":
            # Keep an MPS tensor on Metal. It is consumed by the direct-tensor
            # save path; converting it through NumPy would violate the GPU-only
            # contract.
            return values.contiguous()
        values = values.detach().cpu().numpy()
    values = np.ascontiguousarray(values)
    result = MetalArray(values.shape, values.dtype)
    buffer_view(result._mtl)[:values.nbytes] = memoryview(values).cast("B")
    return result


def is_mps_tensor(value):
    """Whether ``value`` is a PyTorch tensor on the Apple GPU."""
    return isinstance(value, torch.Tensor) and value.device.type == "mps"


def tensor_range(values):
    """Measure range and invalid/subnormal flags in one bounded Metal pass."""
    values = values.to(torch.float32).contiguous()
    params, scalars = _parameters(values)
    params[14] = params[15] = min(params[0], 8192)
    partial = MetalArray((params[14], 4), np.float32)
    result = MetalArray((1, 4), np.float32)
    try:
        command = metal_queue().commandBuffer()
        _dispatch("range", [values, partial], params, scalars, command=command)
        _dispatch("range_reduce", [partial, result], params, scalars, groups=1, command=command)
        complete_command(command, "precision tensor range")
        low, high, invalid, subnormal = result.get()[0].tolist()
        if invalid:
            raise ValueError("Precision conversion requires finite intensities; preserve this source as float32.")
        if subnormal:
            raise ValueError("Metal precision conversion cannot preserve float32 subnormal intensities; keep the original float32 file or use CUDA.")
        return float(low), float(high)
    finally:
        partial.release()
        result.release()


def tensor_restore(values, report):
    """Restore float32 intensities from a tensor of stored codes."""
    if report and report["storage"] == "scaled_uint16":
        return values.to(torch.float32) * report["scale"] + report["offset"]
    return values.to(torch.float32)


def tensor_encode(values, report):
    """Encode a float tensor as float16 or scaled uint16 codes."""
    if report["storage"] == "float16":
        return values.to(torch.float16)
    return torch.round((values.to(torch.float32) - report["offset"]) / report["scale"]).clamp(0, 65535).to(torch.uint16)


def tensor_measure(original, restored, report):
    """Add one tensor block's restoration error statistics to ``report``."""
    difference = restored.to(torch.float32) - original.to(torch.float32)
    errors = torch.stack(
        (torch.sum(difference * difference), torch.max(torch.abs(difference)))
    )
    counts = torch.stack(
        (
            torch.count_nonzero((original > 0) & (restored == 0)),
            torch.count_nonzero(original != restored),
            torch.count_nonzero(~torch.isfinite(restored)),
        )
    )
    squared_error, maximum = errors.cpu().tolist()
    positive_to_zero, changed, overflow = counts.cpu().tolist()
    report["values"] += original.numel()
    report["squared_error"] += squared_error
    report["max_abs_error"] = max(report["max_abs_error"], maximum)
    report["positive_to_zero"] += positive_to_zero
    report["changed"] += changed
    report["overflow"] += overflow


def _parameters(values=None, report=None):
    """Build the 16 uint64 parameters and 4 float scalars every kernel reads.

    The scale and offset are rescaled by a power of two and split into high and
    low float32 parts, so the kernels restore units with near double precision.
    """
    params = [0] * 16
    scalars = [1.0, 0.0, 0.0, 0.0]
    if values is not None:
        params[0] = values.numel() if is_mps_tensor(values) else values.size
        params[3] = {"float32": 0, "float16": 1, "uint16": 2, "uint8": 3, "bool": 3, "uint32": 4}[str(values.dtype).removeprefix("torch.")]
    if report:
        params[4] = int(report["storage"] == "scaled_uint16")
        maximum = max(abs(report["intensity_min"]), abs(report["intensity_max"]))
        exponent = -math.frexp(maximum)[1] if maximum else 0
        params[5] = exponent & 0xffffffffffffffff
        scale = math.ldexp(report["scale"], exponent)
        offset = math.ldexp(report["offset"], exponent)
        # Scalar metadata only. Detector values never undergo host arithmetic.
        for start, value in ((0, scale), (2, offset)):
            high = struct.unpack("f", struct.pack("f", value))[0]
            scalars[start:start + 2] = [high, value - high]
    return params, scalars


def _dispatch(name, buffers, params, scalars=None, *, groups=None, command=None):
    """Encode one precision kernel, waiting for it unless ``command`` is supplied.

    Buffers may be MetalArrays, raw Metal buffers, or contiguous MPS tensors,
    which are bound in place after Torch finishes its queued work.
    """
    metal = metal_module()
    owned = command is None
    command = metal_queue().commandBuffer() if owned else command
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(_pipelines()[name])
    for index, value in enumerate(buffers):
        if is_mps_tensor(value):
            if not value.is_contiguous():
                raise ValueError("Native precision requires contiguous MPS storage.")
            torch.mps.synchronize()
            encoder.setBuffer_offset_atIndex_(
                tensor_buffer(value), value.storage_offset() * value.element_size(), index
            )
        else:
            buffer = value._mtl if isinstance(value, MetalArray) else value
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    param_bytes = struct.pack("16Q", *params)
    scalar_bytes = struct.pack("4f", *(scalars or [1, 0, 0, 0]))
    encoder.setBytes_length_atIndex_(param_bytes, len(param_bytes), 6)
    encoder.setBytes_length_atIndex_(scalar_bytes, len(scalar_bytes), 7)
    if groups is not None:
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(metal.MTLSizeMake(groups, 1, 1), metal.MTLSizeMake(256, 1, 1))
    else:
        encoder.dispatchThreads_threadsPerThreadgroup_(metal.MTLSizeMake(params[15] or params[0], 1, 1), metal.MTLSizeMake(256, 1, 1))
    encoder.endEncoding()
    if owned:
        complete_command(command, f"precision {name}")


def crop(values, region):
    """Copy the ``(row0, row1, col0, col1)`` detector region of every frame."""
    row0, row1, col0, col1 = region
    if region == (0, values.shape[1], 0, values.shape[2]):
        return values
    result = MetalArray((values.shape[0], row1 - row0, col1 - col0), values.dtype)
    params, scalars = _parameters(result)
    params[1:3] = [math.prod(result.shape[1:]), math.prod(values.shape[1:])]
    params[8:13] = [col1 - col0, row0, values.shape[2], col0, values.dtype.itemsize]
    _dispatch("copy", [values, result], params, scalars)
    return result


def restore(values, report):
    """Restore float32 intensities from stored codes on Metal."""
    if values.dtype == np.float32 and not report:
        return values
    result = MetalArray(values.shape, np.float32)
    params, scalars = _parameters(values, report)
    params[13] = int(bool(report))
    _dispatch("restore", [values, result], params, scalars)
    return result


def source_range(blocks, saved):
    """Return the finite intensity range of ``blocks``, restored with the ``saved`` calibration."""
    low, high = math.inf, -math.inf
    for block in blocks:
        calibration = saved
        if calibration and calibration.get("regions"):
            if str(block.dtype).removeprefix("torch.") != "float32":
                raise ValueError("Restore regional calibration before measuring intensity range.")
            calibration = None
        if is_mps_tensor(block):
            block_low, block_high = tensor_range(
                tensor_restore(block, calibration)
            )
            low, high = min(low, block_low), max(high, block_high)
            continue
        values = restore(block, calibration)
        params, scalars = _parameters(values)
        params[14] = params[15] = min(params[0], 8192)
        stats = MetalArray((params[14], 4), np.float32)
        _dispatch("range", [values, stats], params, scalars)
        # These are small GPU-reduced statistics, never input intensities.
        rows = stats.get().tolist()
        # PyObjC frees a Metal buffer only on release; the caller's block is not ours to release.
        stats.release()
        if values is not block:
            values.release()
        for minimum, maximum, invalid, subnormal in rows:
            if invalid:
                raise ValueError("Precision conversion requires finite intensities; preserve this source as float32.")
            if subnormal:
                raise ValueError("Metal precision conversion cannot preserve float32 subnormal intensities; keep the original float32 file or use CUDA.")
            low, high = min(low, minimum), max(high, maximum)
    return low, high


def has_invalid_pixels(mask):
    """Whether the detector mask marks any pixel invalid."""
    uploaded = upload(mask)
    values = restore(uploaded, None)
    params, scalars = _parameters(values)
    params[14] = params[15] = min(params[0], 8192)
    stats = MetalArray((params[14], 4), np.float32)
    _dispatch("range", [values, stats], params, scalars)
    rows = stats.get().tolist()
    # PyObjC frees a Metal buffer only on release; release is idempotent when values is uploaded.
    for temporary in (stats, values, uploaded):
        temporary.release()
    return any(row[1] != 0 for row in rows)


def encode(values, report):
    """Encode float32 intensities as float16 or scaled uint16 codes on Metal."""
    dtype = np.float16 if report["storage"] == "float16" else np.uint16
    result = MetalArray(values.shape, dtype)
    params, scalars = _parameters(values, report)
    _dispatch("encode", [values, result], params, scalars)
    return result


def _accumulate_measurement(errors, counts, exponent, report, values, *, command=None):
    """Reduce partial statistics on the GPU before reading scalar metadata.

    With ``command`` the reduction is appended to the caller's open command
    buffer and completed here, so a fused encode pass and its reduction cost
    one host wait per region instead of two.
    """
    result = MetalArray((1, 4), np.float32)
    totals = MetalArray((1, 4), np.uint64)
    params = [0] * 16
    params[14] = errors.shape[0]
    try:
        _dispatch("measure_reduce", [errors, counts, result, totals], params, groups=1, command=command)
        if command is not None:
            complete_command(command, "precision measure_reduce")
        high, low, maximum, _ = result.get()[0].tolist()
        changed, zero, overflow, _ = totals.get()[0].tolist()
        report["squared_error"] += math.ldexp(high + low, -2 * exponent)
        report["max_abs_error"] = max(
            report["max_abs_error"], math.ldexp(maximum, -exponent)
        )
        report["changed"] += changed
        report["positive_to_zero"] += zero
        report["overflow"] += overflow
        report["values"] += values
    finally:
        result.release()
        totals.release()


# Share the partition between fused and separate measurement so their
# floating-point accumulation order (and saved error statistics) agrees.
_MEASUREMENT_PARTIALS = 65536


def measure(original, restored, report):
    """Add the restoration error statistics of one block to ``report``."""
    params, scalars = _parameters(original, report)
    params[14] = params[15] = min(original.size, _MEASUREMENT_PARTIALS)
    errors = MetalArray((params[14], 4), np.float32)
    counts = MetalArray((params[14], 4), np.uint32)
    try:
        _dispatch("measure", [original, restored, errors, counts], params, scalars)
        exponent = struct.unpack("q", struct.pack("Q", params[5]))[0]
        _accumulate_measurement(errors, counts, exponent, report, original.size)
    finally:
        errors.release()
        counts.release()


def encode_measure(values, report):
    """Encode values and measure restored-unit error in one Metal pass."""
    result = MetalArray(
        values.shape,
        np.float16 if report["storage"] == "float16" else np.uint16,
    )
    params, scalars = _parameters(values, report)
    params[14] = params[15] = min(params[0], _MEASUREMENT_PARTIALS)
    errors = MetalArray((params[14], 4), np.float32)
    counts = MetalArray((params[14], 4), np.uint32)
    try:
        command = metal_queue().commandBuffer()
        _dispatch("encode_measure", [values, result, errors, counts], params, scalars, command=command)
        exponent = struct.unpack("q", struct.pack("Q", params[5]))[0]
        _accumulate_measurement(errors, counts, exponent, report, params[0], command=command)
        return result
    except BaseException:
        result.release()
        raise
    finally:
        errors.release()
        counts.release()


class _ANSPart:
    """One calibration region's scaled codes, kept exactly in its own Metal ANS resident."""

    def __init__(self, owner, shape):
        self.owner, self.shape = owner, shape

    @property
    def nbytes(self):
        return self.owner.nbytes

    def release(self):
        self.owner.release()


def encode_ans(encoded, shape):
    """Retain scaled uint16 codes exactly in native Metal ANS storage."""
    owner = MPSStreamedCounts(shape, np.uint16)
    try:
        owner.append(shared_array(encoded._mtl, encoded.dtype, encoded.shape))
        return _ANSPart(owner, shape)
    except BaseException:
        owner.release()
        raise


@contextmanager
def _part_buffers(part, first=0, stop=None):
    """Decode only the requested scan range of a region, released when the kernel is done.

    The query kernels read three buffer slots (a layout once shared with a
    bit-packed store); the decoded codes fill all three.
    """
    codes = part.owner.decode_scan_range_device(
        first, part.shape[1] if stop is None else stop
    )
    try:
        yield [codes._mtl, codes._mtl, codes._mtl]
    finally:
        codes.release()


class PrecisionSource(DetectorQueries):
    """Keep every encoded intensity resident and restore units inside queries."""

    # quantem.widget's Show4DSTEM reads this flag to keep the source on its GPU path, without a NumPy copy.
    _is_gpu_frames = True
    ndim = 4
    dtype = np.dtype("float32")
    det_bin = 1

    def __init__(self, chunks, shape, precision):
        self.parts, self.shape, self.precision = chunks, tuple(shape), precision
        self.scan_shape, self.det_shape = self.shape[:2], self.shape[2:]
        self.n_frames = math.prod(self.scan_shape)
        self._ends = []
        count = 0
        for part in chunks:
            count += part.shape[1]
            self._ends.append(count)
        self._reports = dict(zip(map(id, chunks), part_reports(precision, self._ends)))
        self.is_released = False

    @property
    def device(self):
        return torch.device("mps")

    @property
    def nbytes(self):
        return sum(part.nbytes for part in self.parts)

    def numel(self):
        """Return the logical float32 element count for array-style consumers."""
        return math.prod(self.shape)

    def _check(self):
        if self.is_released:
            raise RuntimeError("Loaded data was closed; load it again before querying.")

    def _params(self, part):
        params, scalars = _parameters(report=self._reports[id(part)])
        params[1:3] = [math.prod(self.det_shape), part.shape[1]]
        params[12] = 1  # parts are ANS-decoded ranges, read from their first scan
        return params, scalars

    def encoded_blocks(self):
        self._check()
        for part in self.parts:
            result = MetalArray((part.shape[1], *self.det_shape), np.uint16)
            params, scalars = self._params(part)
            params[0] = result.size
            with _part_buffers(part) as buffers:
                _dispatch("unpacked", [*buffers, result], params, scalars)
            yield result

    def frame_native(self, index, *, out=None):
        self._check()
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Scan index must be in [0, {self.n_frames}); got {index}.")
        if out is not None and (not isinstance(out, MetalArray) or out.shape != self.det_shape or out.dtype != self.dtype):
            raise ValueError("out must be a native float32 Metal diffraction result with matching shape.")
        part_index = bisect.bisect_right(self._ends, index)
        part = self.parts[part_index]
        params, scalars = self._params(part)
        params[0], params[8] = params[1], index - (self._ends[part_index - 1] if part_index else 0)
        result = out if out is not None else MetalArray(self.det_shape, np.float32)
        first = params[8]
        params[8] = 0
        with _part_buffers(part, first, first + 1) as buffers:
            _dispatch("frame", [*buffers, result], params, scalars)
        return result

    def _decode_scan_range_torch(self, first, stop):
        """Restore a range directly into independently owned Torch MPS storage."""
        self._check()
        if not 0 <= first < stop <= self.n_frames:
            raise ValueError("Read a nonempty range within the scan.")
        output = torch.empty((stop - first, *self.det_shape), device="mps", dtype=torch.float32)
        start = 0
        for part, end in zip(self.parts, self._ends):
            low, high = max(first, start), min(stop, end)
            if low < high:
                params, scalars = self._params(part)
                params[0], params[8] = (high - low) * params[1], low - start
                offset = params[8]
                params[8] = 0
                with _part_buffers(part, offset, high - start) as buffers:
                    _dispatch("range_read", [*buffers, output[low - first:high - first]], params, scalars)
            start = end
            if start >= stop:
                break
        return output

    def frame(self, index):
        native = self.frame_native(index)
        values = native.get()
        native.release()
        return values

    def __getitem__(self, position):
        if isinstance(position, (int, np.integer)):
            return self.frame_native(int(position)).to_torch()
        if not isinstance(position, tuple) or len(position) != 2:
            raise TypeError(
                "Select one diffraction pattern with source[index] or "
                "source[row, col]."
            )
        row, col = position
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise IndexError("Scan position lies outside this loaded region.")
        return self.frame_native(row * self.shape[1] + col).to_torch()

    def mean_dp(self):
        self._check()
        result = MetalArray(self.det_shape, np.float32)
        # Parts retain their buffers until this ordered submission finishes.
        command = metal_queue().commandBuffer()
        for index, part in enumerate(self.parts):
            params, scalars = self._params(part)
            params[0], params[8], params[9] = params[1], int(index > 0), self.n_frames
            owner = part.owner
            owner._clear_errors()
            params[10] = owner.interval
            chunk = owner.chunks[0]
            _dispatch(
                "ans_mean",
                [*chunk.buffers, owner._decoding, owner._errors, result],
                params, scalars, command=command,
            )
        complete_command(command, "precision mean diffraction")
        for part in self.parts:
            part.owner._check_errors()
        return result

    def masked_sum_native(self, mask, *, out=None):
        self._check()
        binary = None
        values = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        if values.shape == self.det_shape and np.all((values == 0) | (values == 1)):
            binary = values.astype(bool)
        weights = None if binary is not None else upload(mask)
        if weights is not None and (weights.shape != self.det_shape or str(weights.dtype).removeprefix("torch.") not in ("float32", "uint8", "bool")):
            raise ValueError(f"Detector mask must have shape {self.det_shape} and float32/bool weights.")
        result = out if out is not None else MetalArray(self.scan_shape, np.float32)
        if not isinstance(result, MetalArray) or result.shape != self.scan_shape or result.dtype != self.dtype:
            raise ValueError("out must be a native float32 Metal scan result with matching shape.")
        if binary is not None:
            # Exact integer code sums over the mask, calibrated once per scan in
            # double precision: float32(scale * sum(codes) + offset * pixels).
            count = int(binary.sum())
            image = np.frombuffer(buffer_view(result._mtl), np.float32, count=result.size)
            first = 0
            for part in self.parts:
                region = self._reports[id(part)]
                codes = part.owner.masked_code_sums(binary)
                image[first:first + codes.size] = codes * region["scale"] + region["offset"] * count
                first += codes.size
            return result
        weights = restore(weights, None)
        first = 0
        for part in self.parts:
            params, scalars = self._params(part)
            params[8] = first
            with _part_buffers(part) as buffers:
                _dispatch("detector", [*buffers, result, weights], params, scalars, groups=part.shape[1])
            first += part.shape[1]
        return result

    def masked_sum(self, mask):
        return self.masked_sum_native(mask)

    def masked_sum_exact_native(self, mask, *, out=None):
        """Restored intensities are float32, so there is no exact integer image on any output."""
        return self.masked_sum_exact(mask)

    def reduce_frames(self, indices, reduce="mean"):
        indices = list(indices)
        if not indices or reduce not in {"mean", "sum", "max"}:
            raise ValueError("Select at least one scan position and use mean, sum or max.")
        result = MetalArray(self.det_shape, np.float32)
        buffer_view(result._mtl)[:] = b"\0" * result.nbytes
        if reduce == "max":
            self.frame_native(indices[0], out=result)
            indices = indices[1:]
        for index in indices:
            value = self.frame_native(index)
            params, scalars = _parameters(value)
            # The kernel adds each frame divided by params[9]: the frame count for a mean, 1 for a sum.
            params[0], params[8], params[9] = value.size, 2 if reduce == "max" else 0, len(indices) if reduce == "mean" else 1
            _dispatch("reduce", [value, result], params, scalars)
            value.release()
        pattern = result.get()
        result.release()
        return pattern

    def center_of_mass(self, mask=None):
        weights = torch.ones(self.det_shape, device="mps") if mask is None else torch.as_tensor(mask, device="mps", dtype=torch.float32)
        denominator = self.masked_sum_native(weights)
        row = self.masked_sum_native(weights * torch.arange(self.det_shape[0], device="mps")[:, None])
        col = self.masked_sum_native(weights * torch.arange(self.det_shape[1], device="mps")[None, :])
        params, scalars = _parameters(denominator)
        for value in (row, col):
            _dispatch("divide", [value, denominator], params, scalars)
        return col, row

    def release(self):
        for part in self.parts:
            part.release()
        self.parts = []
        self.is_released = True
