"""Metal precision conversion and direct queries over resident packed streams."""

import bisect
import math
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

from .packed import _allocate_shared, _buffer_view, _complete, _metal_module, _release


@lru_cache(maxsize=1)
def _runtime():
    metal = _metal_module()
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("Precision conversion needs an available Metal GPU.")
    options = metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(False)
    library, error = device.newLibraryWithSource_options_error_(
        (Path(__file__).parent / "kernels" / "precision.msl").read_text(), options, None
    )
    if library is None:
        raise RuntimeError(f"Could not compile Metal precision kernels: {error}")
    pipelines = {}
    for name in (
        "copy", "restore", "encode", "encode_measure", "range", "measure",
        "widths", "pack", "frame", "unpacked", "detector", "mean", "reduce",
    ):
        function = library.newFunctionWithName_(f"precision_{name}")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            raise RuntimeError(f"Could not compile Metal precision {name}: {error}")
        pipelines[name] = pipeline
    return device, metal, device.newCommandQueue(), pipelines


class MetalArray:
    """Own a bounded GPU intermediate or an explicitly requested 2D product."""

    def __init__(self, shape, dtype, buffer=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize
        device, metal, _, _ = _runtime()
        self._mtl = buffer if buffer is not None else _allocate_shared(device, metal, max(4, self.nbytes), "precision")

    def get(self):
        if self._mtl is None:
            raise RuntimeError("This GPU result was released; request it again.")
        return np.frombuffer(_buffer_view(self._mtl), self.dtype, count=self.size).reshape(self.shape).copy()

    def to_torch(self):
        """Copy this bounded shared Metal result into a Torch MPS tensor."""
        if self._mtl is None:
            raise RuntimeError("This GPU result was released; request it again.")
        import torch

        view = np.frombuffer(
            _buffer_view(self._mtl), self.dtype, count=self.size
        ).reshape(self.shape)
        return torch.from_numpy(view).to("mps")

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.get(), dtype=dtype)

    def release(self):
        buffer, self._mtl = self._mtl, None
        if buffer is not None:
            _release(buffer)

    def __del__(self):
        try:
            self.release()
        except (AttributeError, TypeError):
            pass


def upload(values):
    """Transfer bytes only; scientific conversion happens in Metal kernels."""
    if isinstance(values, MetalArray):
        return values
    if hasattr(values, "device") and str(values.device).startswith("mps"):
        # Keep an MPS tensor on Metal. It is consumed by the direct-tensor save
        # path; converting it through NumPy would violate the GPU-only contract.
        return values.contiguous()
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.ascontiguousarray(values)
    result = MetalArray(values.shape, values.dtype)
    _buffer_view(result._mtl)[:values.nbytes] = memoryview(values).cast("B")
    return result


def is_mps_tensor(value):
    return hasattr(value, "device") and str(value.device).startswith("mps")


def tensor_range(values):
    """Return a float32 tensor range using MPS reductions."""
    import torch

    # Submit all reductions before reading their three scalar results. Reading
    # each scalar separately would drain the command queue three times.
    low, high, finite = torch.stack(
        (values.amin(), values.amax(), torch.isfinite(values).all())
    ).cpu().tolist()
    if not finite:
        raise ValueError("Precision conversion requires finite intensities; preserve this source as float32.")
    return float(low), float(high)


def tensor_restore(values, report):
    import torch
    if report and report["storage"] == "scaled_uint16":
        return values.to(torch.float32) * report["scale"] + report["offset"]
    return values.to(torch.float32)


def tensor_encode(values, report):
    import torch
    if report["storage"] == "float16":
        return values.to(torch.float16)
    return torch.round((values.to(torch.float32) - report["offset"]) / report["scale"]).clamp(0, 65535).to(torch.uint16)


def tensor_measure(original, restored, report):
    import torch
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
    p = [0] * 16
    f = [1.0, 0.0, 0.0, 0.0]
    if values is not None:
        p[0] = values.size
        p[3] = {"float32": 0, "float16": 1, "uint16": 2, "uint8": 3, "bool": 3, "uint32": 4}[str(values.dtype)]
    if report:
        p[4] = int(report["storage"] == "scaled_uint16")
        maximum = max(abs(report["intensity_min"]), abs(report["intensity_max"]))
        exponent = -math.frexp(maximum)[1] if maximum else 0
        p[5] = exponent & 0xffffffffffffffff
        scale = math.ldexp(report["scale"], exponent)
        offset = math.ldexp(report["offset"], exponent)
        # Scalar metadata only. Detector values never undergo host arithmetic.
        for start, value in ((0, scale), (2, offset)):
            high = struct.unpack("f", struct.pack("f", value))[0]
            f[start:start + 2] = [high, value - high]
    return p, f


def _dispatch(name, buffers, p, f=None, *, groups=None, command=None):
    _, metal, queue, pipelines = _runtime()
    owned = command is None
    command = queue.commandBuffer() if owned else command
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipelines[name])
    for index, value in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(getattr(value, "_mtl", value), 0, index)
    params = struct.pack("16Q", *p)
    floats = struct.pack("4f", *(f or [1, 0, 0, 0]))
    encoder.setBytes_length_atIndex_(params, len(params), 6)
    encoder.setBytes_length_atIndex_(floats, len(floats), 7)
    if groups is not None:
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(metal.MTLSizeMake(groups, 1, 1), metal.MTLSizeMake(256, 1, 1))
    else:
        encoder.dispatchThreads_threadsPerThreadgroup_(metal.MTLSizeMake(p[15] or p[0], 1, 1), metal.MTLSizeMake(256, 1, 1))
    encoder.endEncoding()
    if owned:
        _complete(command, f"precision {name}")


def crop(values, region):
    row0, row1, col0, col1 = region
    if region == (0, values.shape[1], 0, values.shape[2]):
        return values
    result = MetalArray((values.shape[0], row1 - row0, col1 - col0), values.dtype)
    p, f = _parameters(result)
    p[1:3] = [math.prod(result.shape[1:]), math.prod(values.shape[1:])]
    p[8:13] = [col1 - col0, row0, values.shape[2], col0, values.dtype.itemsize]
    _dispatch("copy", [values, result], p, f)
    return result


def decode_prepared(source, prepared):
    from .dense import MPSDecompressor

    frames, frame_bytes = prepared["total_frames"], prepared["frame_bytes"]
    compressed = len(prepared["read_buffer"])
    decoder = source.decoder
    if decoder is None or frames > decoder.max_frames or compressed > decoder._comp_np.size:
        if decoder is not None:
            decoder.free()
        source.decoder = MPSDecompressor(
            max_compressed_bytes=max(compressed, frames * frame_bytes * 2),
            max_frames=frames, frame_bytes=frame_bytes,
            n_blocks_per_frame=(frame_bytes + 8191) // 8192, gpu_batch=frames,
        )
    decoded = source.decoder.load_prepared_frames(prepared, output_dtype=None)
    # The sparse decoder transfers ownership of its fresh output, without copy.
    result = MetalArray(decoded.shape, decoded.dtype, buffer=decoded._mtl)
    decoded._mtl = None
    return result


def restore(values, report):
    if values.dtype == np.float32 and not report:
        return values
    result = MetalArray(values.shape, np.float32)
    p, f = _parameters(values, report)
    p[13] = int(bool(report))
    _dispatch("restore", [values, result], p, f)
    return result


def source_range(source):
    low, high = math.inf, -math.inf
    for block in source.blocks():
        if is_mps_tensor(block):
            block_low, block_high = tensor_range(
                tensor_restore(block, source.saved)
            )
            low, high = min(low, block_low), max(high, block_high)
            continue
        values = restore(block, source.saved)
        p, f = _parameters(values)
        p[14] = p[15] = min(values.size, 8192)
        stats = MetalArray((p[14], 4), np.float32)
        _dispatch("range", [values, stats], p, f)
        # These are small GPU-reduced statistics, never input intensities.
        for minimum, maximum, invalid, subnormal in stats.get().tolist():
            if invalid:
                raise ValueError("Precision conversion requires finite intensities; preserve this source as float32.")
            if subnormal:
                raise ValueError("Metal precision conversion cannot preserve float32 subnormal intensities; keep the original float32 file or use CUDA.")
            low, high = min(low, minimum), max(high, maximum)
    return low, high


def has_invalid_pixels(mask):
    values = restore(upload(mask), None)
    p, f = _parameters(values)
    p[14] = p[15] = min(values.size, 8192)
    stats = MetalArray((p[14], 4), np.float32)
    _dispatch("range", [values, stats], p, f)
    return any(row[1] != 0 for row in stats.get().tolist())


def encode(values, report):
    dtype = np.float16 if report["storage"] == "float16" else np.uint16
    result = MetalArray(values.shape, dtype)
    p, f = _parameters(values, report)
    _dispatch("encode", [values, result], p, f)
    return result


def _accumulate_measurement(errors, counts, exponent, report, values):
    squared = 0.0
    for row in errors.get().tolist():
        squared += math.ldexp(row[0], -2 * exponent)
        report["max_abs_error"] = max(
            report["max_abs_error"], math.ldexp(row[1], -exponent)
        )
    report["squared_error"] += squared
    for changed, zero, overflow, _ in counts.get().tolist():
        report["changed"] += changed
        report["positive_to_zero"] += zero
        report["overflow"] += overflow
    report["values"] += values


def measure(original, restored, report):
    p, f = _parameters(original, report)
    p[14] = p[15] = min(original.size, 8192)
    errors = MetalArray((p[14], 4), np.float32)
    counts = MetalArray((p[14], 4), np.uint32)
    try:
        _dispatch("measure", [original, restored, errors, counts], p, f)
        exponent = struct.unpack("q", struct.pack("Q", p[5]))[0]
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
    p, f = _parameters(values, report)
    p[14] = p[15] = min(values.size, 8192)
    errors = MetalArray((p[14], 4), np.float32)
    counts = MetalArray((p[14], 4), np.uint32)
    try:
        _dispatch("encode_measure", [values, result, errors, counts], p, f)
        exponent = struct.unpack("q", struct.pack("Q", p[5]))[0]
        _accumulate_measurement(errors, counts, exponent, report, values.size)
        return result
    except BaseException:
        result.release()
        raise
    finally:
        errors.release()
        counts.release()


class _PackedPart:
    def __init__(self, words, offsets, widths, shape):
        self.buffers = [words, offsets, widths]
        self.shape = shape
        self.nbytes = sum(value.nbytes for value in self.buffers)

    def release(self):
        for value in self.buffers:
            value.release()
        self.buffers = []


def pack(encoded, shape):
    """Pack native 16-bit patterns, including float16, with GPU-measured widths."""
    frames, pixels = encoded.shape[0], math.prod(encoded.shape[1:])
    streams = ((frames + 127) // 128) * pixels
    widths = MetalArray((streams,), np.uint8)
    lengths = MetalArray((streams,), np.uint64)
    p, f = _parameters()
    p[0:3] = [streams, pixels, frames]
    _dispatch("widths", [encoded, widths, lengths], p, f)
    offsets = MetalArray((streams + 1,), np.uint64)
    # Allocation index only, matching the existing Metal packed-count owner.
    try:
        index = np.frombuffer(_buffer_view(offsets._mtl), np.uint64, count=streams + 1)
        index[0] = 0
        np.cumsum(np.frombuffer(_buffer_view(lengths._mtl), np.uint64, count=streams), out=index[1:])
        words = MetalArray((max(1, int(index[-1])),), np.uint32)
    finally:
        lengths.release()
    _dispatch("pack", [encoded, widths, offsets, words], p, f)
    return _PackedPart(words, offsets, widths, shape)


class PrecisionSource:
    """Keep every encoded intensity resident and restore units inside queries."""

    _is_gpu_frames = True
    ndim = 4
    dtype = np.dtype("float32")
    capabilities = ()
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
        self.is_released = False

    @property
    def device(self):
        import torch
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
        p, f = _parameters(report=self.precision)
        p[1:3] = [math.prod(self.det_shape), part.shape[1]]
        return p, f

    def encoded_blocks(self):
        self._check()
        for part in self.parts:
            result = MetalArray((part.shape[1], *self.det_shape), np.float16 if self.precision["storage"] == "float16" else np.uint16)
            p, f = self._params(part)
            p[0] = result.size
            _dispatch("unpacked", [*part.buffers, result], p, f)
            yield result

    def frame_native(self, index, *, out=None):
        self._check()
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Scan index must be in [0, {self.n_frames}); got {index}.")
        if out is not None and (not isinstance(out, MetalArray) or out.shape != self.det_shape or out.dtype != self.dtype):
            raise ValueError("out must be a native float32 Metal diffraction result with matching shape.")
        part_index = bisect.bisect_right(self._ends, index)
        part = self.parts[part_index]
        p, f = self._params(part)
        p[0], p[8] = p[1], index - (self._ends[part_index - 1] if part_index else 0)
        result = out if out is not None else MetalArray(self.det_shape, np.float32)
        _dispatch("frame", [*part.buffers, result], p, f)
        return result

    def frame(self, index):
        return self.frame_native(index).get()

    def __getitem__(self, position):
        import torch
        if isinstance(position, (int, np.integer)):
            return torch.from_numpy(self.frame(int(position))).to("mps")
        if not isinstance(position, tuple) or len(position) != 2:
            raise TypeError(
                "Select one diffraction pattern with source[index] or "
                "source[row, col]."
            )
        row, col = position
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise IndexError("Scan position lies outside this loaded region.")
        return torch.from_numpy(self.frame(row * self.shape[1] + col)).to("mps")

    def mean_dp(self):
        self._check()
        result = MetalArray(self.det_shape, np.float32)
        command = _runtime()[2].commandBuffer()
        for index, part in enumerate(self.parts):
            p, f = self._params(part)
            p[0], p[8], p[9] = p[1], int(index > 0), self.n_frames
            _dispatch("mean", [*part.buffers, result], p, f, command=command)
        _complete(command, "precision mean")
        return result

    def masked_sum_native(self, mask, *, out=None):
        self._check()
        weights = upload(mask)
        if weights.shape != self.det_shape or weights.dtype not in (np.float32, np.uint8, np.bool_):
            raise ValueError(f"Detector mask must have shape {self.det_shape} and float32/bool weights.")
        weights = restore(weights, None)
        result = out if out is not None else MetalArray(self.scan_shape, np.float32)
        if not isinstance(result, MetalArray) or result.shape != self.scan_shape or result.dtype != self.dtype:
            raise ValueError("out must be a native float32 Metal scan result with matching shape.")
        command = _runtime()[2].commandBuffer()
        first = 0
        for part in self.parts:
            p, f = self._params(part)
            p[8] = first
            _dispatch("detector", [*part.buffers, result, weights], p, f, groups=part.shape[1], command=command)
            first += part.shape[1]
        _complete(command, "precision detector")
        return result

    def masked_sum(self, mask):
        return self.masked_sum_native(mask)

    def reduce_frames(self, indices, reduce="mean"):
        indices = list(indices)
        if not indices or reduce not in {"mean", "sum", "max"}:
            raise ValueError("Select at least one scan position and use mean, sum or max.")
        result = MetalArray(self.det_shape, np.float32)
        _buffer_view(result._mtl)[:] = b"\0" * result.nbytes
        if reduce == "max":
            self.frame_native(indices[0], out=result)
            indices = indices[1:]
        for index in indices:
            value = self.frame_native(index)
            p, f = _parameters(value)
            p[0], p[8], p[9] = value.size, 2 if reduce == "max" else 0, len(indices) + (1 if reduce == "max" else 0)
            _dispatch("reduce", [value, result], p, f)
            value.release()
        return result.get()

    def center_of_mass(self, mask=None):
        weights = np.ones(self.det_shape, np.float32) if mask is None else np.asarray(mask, np.float32)
        denominator = self.masked_sum_native(weights)
        row = self.masked_sum_native(weights * np.arange(self.det_shape[0], dtype=np.float32)[:, None])
        col = self.masked_sum_native(weights * np.arange(self.det_shape[1], dtype=np.float32)[None, :])
        return col, row

    def release(self):
        for part in self.parts:
            part.release()
        self.parts = []
        self.is_released = True
