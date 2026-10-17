"""Typed bit-lane residents with bounded accelerator encoding and decoding."""

import copy
import math
from contextlib import nullcontext
from functools import wraps

import numpy as np
import torch

from quantem.gpu.device.cuda_runtime import cuda_device_index
from quantem.gpu.formats.qem.snapshot import decode_valid
from quantem.gpu.resident.cuda.float_ans import CUDAFloatLanes
from quantem.gpu.resident.mps.float_ans import MPSFloatLanes

MAX_DECODE_BYTES = 32 << 20


def _device_scoped(operation):
    """Keep queries on the resident's GPU even if a client switches devices."""

    @wraps(operation)
    def run(self, *args, **kwargs):
        with self.device_context():
            return operation(self, *args, **kwargs)

    return run


class BitANSResident:
    """Own original measurement bits, including wide integers, on the GPU.

    The count codec stores independent 16-bit words without interpreting the
    source measurements. Decoding reassembles their native dtype exactly.
    Floating arithmetic is distinct from this bit-exact storage contract.
    """

    interval = 512

    def __init__(self, header: dict, backend: str, device: int | str | None = None):
        self.header = copy.deepcopy(header)
        self.shape = tuple(header["shape"])
        self.dtype = np.dtype(header["dtype"])
        self.words_per_value = max(1, self.dtype.itemsize // 2)
        self.backend = backend
        self.is_released = False
        self.valid_pixels = (
            decode_valid(header["valid"], self.shape[2:])
            if "valid" in header
            else np.ones(self.shape[2:], bool)
        )
        self._background = None
        self.peak_decode_bytes = 0
        self._mean = None
        self.detector_shape = self.shape[2:]
        if self.frame_bytes > MAX_DECODE_BYTES:
            raise ValueError("One diffraction frame exceeds the 32 MiB decode budget.")
        lane_shape = (*self.shape[:3], self.shape[3] * self.words_per_value)
        if backend == "cuda":
            import cupy as cp

            selected = cuda_device_index(device)
            with cp.cuda.Device(selected):
                self._lanes = CUDAFloatLanes(lane_shape, np.uint16)
            self.device = selected
        else:
            self._lanes = MPSFloatLanes(lane_shape, np.uint16)
            self.device = self._lanes.device

    @property
    def frame_bytes(self) -> int:
        return math.prod(self.shape[2:]) * self.dtype.itemsize

    @property
    def decode_frame_bytes(self) -> int:
        """Include the 16-bit scratch word used for an 8-bit measurement."""
        return math.prod(self.shape[2:]) * max(2, self.dtype.itemsize)

    def append_device(self, raw) -> None:
        """Encode one bounded GPU window without converting its measurements."""
        if (
            tuple(raw.shape[1:]) != self.detector_shape
            or np.dtype(raw.dtype) != self.dtype
        ):
            raise ValueError(
                "Provide a GPU window with the acquisition's dtype and detector shape."
            )
        count = int(raw.shape[0])
        if not 1 <= count <= min(512, MAX_DECODE_BYTES // self.decode_frame_bytes):
            raise ValueError(
                "Typed ANS input must fit one 512-scan, 32 MiB word window."
            )
        shape = (count, self.shape[2], self.shape[3] * self.words_per_value)
        if self.backend == "cuda":
            words = (
                raw.view(np.uint8).astype(np.uint16)
                if self.dtype.itemsize == 1
                else raw.view(np.uint16)
            )
            self._lanes.append(words.reshape(shape))
        else:
            from quantem.gpu.device.metal_runtime import shared_array, tensor_buffer

            if self.dtype.itemsize == 1:
                from quantem.gpu.device.metal_runtime import copy_to_torch

                words = copy_to_torch(raw._mtl, raw.shape, np.uint8, "Byte lanes").to(
                    torch.uint16
                )
                torch.mps.synchronize()
                buffer = tensor_buffer(words)
            else:
                buffer = raw._mtl
            self._lanes.append(shared_array(buffer, np.uint16, shape))

    @property
    def nbytes(self) -> int:
        """Return resident bytes: encoded lanes plus the cached dark plane and mean."""
        if self.is_released:
            return 0
        cached = (self._background, self._mean)
        return self._lanes.nbytes + sum(
            (
                value.nbytes
                if self.backend == "cuda"
                else value.numel() * value.element_size()
            )
            for value in cached
            if value is not None
        )

    @property
    def logical_nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "ANS stays encoded; request a DP, product or bounded scan window."
        )

    def device_context(self):
        if self.backend == "cuda":
            import cupy as cp

            return cp.cuda.Device(self.device)
        return nullcontext()

    def _check(self, first=None, stop=None):
        """Refuse a released resident and decode ranges beyond the 32 MiB window."""
        if self.is_released:
            raise RuntimeError("The ANS resident was released; load it again.")
        if first is not None:
            if (
                type(first) is not int
                or type(stop) is not int
                or not 0 <= first < stop <= math.prod(self.shape[:2])
            ):
                raise ValueError("Choose a nonempty scan range within the acquisition.")
            size = (stop - first) * self.decode_frame_bytes
            if size > MAX_DECODE_BYTES:
                raise ValueError(
                    f"Decode requests are limited to 32 MiB; use at most {MAX_DECODE_BYTES // self.decode_frame_bytes} frames."
                )
            self.peak_decode_bytes = max(self.peak_decode_bytes, size)

    def decode_scan_range_device(self, first: int, stop: int):
        """Return original measurement bits in one bounded device buffer.

        The buffer is a CuPy array on CUDA and a Metal array on MPS.
        """
        self._check(first, stop)
        output = self._lanes.decode_scan_range_device(first, stop)
        if self.backend == "cuda":
            if self.dtype.itemsize == 1:
                if bool((output > 255).any()):
                    raise ValueError(
                        "Decoded byte lane exceeds its 8-bit dtype; recopy the QEM file."
                    )
                output = output.astype(np.uint8)
            return output.view(self.dtype).reshape(stop - first, *self.detector_shape)
        if self.dtype.itemsize == 1:
            from quantem.gpu.device.metal_runtime import (
                complete_command,
                metal_queue,
                tensor_buffer,
            )
            from quantem.gpu.resident.mps.arrays import MetalArray

            words = output
            try:
                tensor = words.to_torch().to(torch.int32)
                if bool((tensor > 255).any()):
                    raise ValueError(
                        "Decoded byte lane exceeds its 8-bit dtype; recopy the QEM file."
                    )
                tensor = tensor.to(torch.uint8)
            finally:
                words.release()
            output = MetalArray(tensor.shape, np.uint8)
            torch.mps.synchronize()
            command = metal_queue().commandBuffer()
            blit = command.blitCommandEncoder()
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                tensor_buffer(tensor), 0, output._mtl, 0, output.nbytes
            )
            blit.endEncoding()
            complete_command(command, "Byte lane decode")
        output.dtype = self.dtype
        output.shape = (stop - first, *self.detector_shape)
        return output

    def extract_diffraction_device(self, scan_row: int, scan_col: int):
        """Return one original DP without applying a display correction."""
        if not 0 <= scan_row < self.shape[0] or not 0 <= scan_col < self.shape[1]:
            raise IndexError("Choose a scan row and column within the acquisition.")
        first = scan_row * self.shape[1] + scan_col
        output = self.decode_scan_range_device(first, first + 1)
        if self.backend == "cuda":
            return output.reshape(self.detector_shape)
        output.shape = self.detector_shape
        return output

    @_device_scoped
    def _tensor(self, first, stop, *, corrected=True):
        """Decode a scan range as a CuPy array or Torch MPS tensor, dark-subtracted by default."""
        self._check(first, stop)
        if self.backend == "cuda":
            output = self.decode_scan_range_device(first, stop)
        else:
            if self.dtype == np.dtype("float64"):
                from quantem.gpu.resident.mps.float64_display import float32_view

                raw = self.decode_scan_range_device(first, stop)
                try:
                    output = float32_view(raw)
                finally:
                    raw.release()
            else:
                output = self._lanes.decode_scan_range_torch(first, stop)
                if self.dtype.itemsize == 1:
                    output = output.to(torch.int32)
                    if bool((output > 255).any()):
                        raise ValueError(
                            "Decoded byte lane exceeds its 8-bit dtype; recopy the QEM file."
                        )
                    output = output.to(torch.uint8)
                output = output.view(getattr(torch, self.dtype.name)).reshape(
                    stop - first, *self.detector_shape
                )
        if corrected and self._background is not None:
            output -= self._background
        if corrected and not self.valid_pixels.all():
            valid = self._array(self.valid_pixels)
            if self.backend == "mps" and self.dtype.kind == "u":
                # Torch's unsigned masking kernels are not uniformly available.
                signed = getattr(torch, f"int{self.dtype.itemsize * 8}")
                output = torch.where(valid, output.view(signed), 0).view(output.dtype)
            else:
                output = self._where(valid, output)
        return output

    def _array(self, values):
        """Upload a small host array, such as a mask or coordinates, to this resident's GPU."""
        if self.backend == "cuda":
            import cupy as cp

            with cp.cuda.Device(self.device):
                return cp.asarray(values)
        return torch.as_tensor(values, device="mps")

    def _empty(self, shape, *, zero=False):
        """Allocate a float32 product on this resident's GPU."""
        if self.backend == "cuda":
            import cupy as cp

            with cp.cuda.Device(self.device):
                return (cp.zeros if zero else cp.empty)(shape, cp.float32)
        return (torch.zeros if zero else torch.empty)(
            shape, dtype=torch.float32, device="mps"
        )

    def synchronize(self) -> None:
        """Wait for queued GPU work on this resident's device."""
        if self.backend == "cuda":
            import cupy as cp

            with cp.cuda.Device(self.device):
                cp.cuda.get_current_stream().synchronize()
        else:
            torch.mps.synchronize()

    def _sum(self, values, axes):
        """Sum over ``axes`` with the CuPy or Torch spelling."""
        return values.sum(axis=axes) if self.backend == "cuda" else values.sum(dim=axes)

    def _where(self, condition, values, other=0):
        """Select with the CuPy or Torch ``where``."""
        if self.backend == "cuda":
            import cupy as cp

            return cp.where(condition, values, other)
        return torch.where(condition, values, other)

    def _mask(self, mask):
        """Validate a binary detector selection and exclude recorded bad pixels."""
        selected = (
            np.ones(self.detector_shape, bool) if mask is None else np.asarray(mask)
        )
        if selected.shape != self.detector_shape or not np.all(
            (selected == 0) | (selected == 1)
        ):
            raise ValueError(
                f"Use a binary detector mask with shape {self.detector_shape}."
            )
        return self._array(selected.astype(bool) & self.valid_pixels)

    def _integer_sum(self, values, selected):
        """Sum integer pixels exactly; raise rather than overflow a 64-bit result.

        For 64-bit input, independently accumulate the low unsigned and high
        signed/unsigned 32-bit words. Their sums fit 64 bits under the bounded
        detector geometry. This also handles signed cancellation without an
        intermediate-overflow false alarm.
        """
        if self.backend == "cuda":
            import cupy as xp

            cast = lambda value: value.astype(xp.int64)
            words = values.view(xp.int32) if self.dtype.itemsize == 8 else None
        else:
            xp = torch
            cast = lambda value: value.to(torch.int64)
            words = values.view(torch.int32) if self.dtype.itemsize == 8 else None
        if words is None:
            result = self._sum(self._where(selected, cast(values)), (1, 2))
        else:
            words = words.reshape(*values.shape, 2)
            low = self._sum(
                self._where(selected, cast(words[..., 0]) & 0xFFFFFFFF), (1, 2)
            )
            high = cast(words[..., 1])
            if self.dtype.kind == "u":
                high &= 0xFFFFFFFF
            high = self._sum(self._where(selected, high), (1, 2)) + (low >> 32)
            result = self._join_sum(low, high)
        return result.view(xp.uint64) if self.dtype.kind == "u" else result

    def _join_sum(self, low, high):
        """Validate a two-word sum before assembling its 64-bit result."""
        minimum, maximum = (
            (0, 0xFFFFFFFF) if self.dtype.kind == "u" else (-(1 << 31), (1 << 31) - 1)
        )
        if bool(((high < minimum) | (high > maximum)).any()):
            raise OverflowError(
                "This sum exceeds the 64-bit result range. Use a smaller selection; "
                "original measurements remain unchanged."
            )
        return (low & 0xFFFFFFFF) | (high << 32)

    @_device_scoped
    def detector_total_device(self, indices=None):
        """Sum all patterns exactly to signed/unsigned 64-bit detector pixels."""
        if self.dtype.kind not in "iu":
            raise TypeError("Exact totals require integer measurements.")
        scans = math.prod(self.shape[:2])
        self._lanes.check_readable(
            0 if indices is None else int(indices.min()),
            scans if indices is None else int(indices.max()) + 1,
        )
        if scans >= 1 << 31:
            raise OverflowError("Exact totals support fewer than 2^31 scan positions.")
        if self.backend == "cuda":
            import cupy as xp

            cast = lambda value: value.astype(xp.int64)
            zeros = lambda: xp.zeros(self.detector_shape, xp.int64)
        else:
            xp = torch
            cast = lambda value: value.to(torch.int64)
            zeros = lambda: torch.zeros(
                self.detector_shape, dtype=torch.int64, device="mps"
            )
        low, high = zeros(), zeros()
        for chunk in self._lanes.chunks:
            local = (
                None
                if indices is None
                else indices[
                    (indices >= chunk.first) & (indices < chunk.first + chunk.scans)
                ]
                - chunk.first
            )
            if local is not None and not len(local):
                continue
            values = self._tensor(chunk.first, chunk.first + chunk.scans)
            if local is not None:
                # Index integer storage through a signed view on MPS.
                if self.backend == "mps" and self.dtype.kind == "u":
                    signed = getattr(torch, f"int{values.element_size() * 8}")
                    values = values.view(signed)[
                        self._array(local.astype(np.int64))
                    ].view(values.dtype)
                else:
                    values = values[self._array(local.astype(np.int64))]
            if self.dtype.itemsize == 8:
                words = values.view(xp.int32).reshape(*values.shape, 2)
                low += self._sum(cast(words[..., 0]) & 0xFFFFFFFF, (0,))
                upper = cast(words[..., 1])
                high += self._sum(
                    upper & 0xFFFFFFFF if self.dtype.kind == "u" else upper, (0,)
                )
            else:
                low += self._sum(cast(values), (0,))
            del values
        result = (
            self._join_sum(low, high + (low >> 32)) if self.dtype.itemsize == 8 else low
        )
        return result.view(xp.uint64) if self.dtype.kind == "u" else result

    @_device_scoped
    def detector_sum_device(self, mask):
        """Return an exact 64-bit integer VI, or a floating VI in calculation precision."""
        self._check()
        selected = self._mask(mask)
        if self.backend == "cuda":
            import cupy as xp

            dtype = (
                (xp.uint64 if self.dtype.kind == "u" else xp.int64)
                if self.dtype.kind in "iu"
                else (xp.float64 if self.dtype.itemsize == 8 else xp.float32)
            )
            output = xp.empty(math.prod(self.shape[:2]), dtype)
        else:
            dtype = (
                (torch.uint64 if self.dtype.kind == "u" else torch.int64)
                if self.dtype.kind in "iu"
                else torch.float32
            )
            output = torch.empty(math.prod(self.shape[:2]), dtype=dtype, device="mps")
        self._lanes.check_readable(0, math.prod(self.shape[:2]))
        for chunk in self._lanes.chunks:
            first, stop = chunk.first, chunk.first + chunk.scans
            values = self._tensor(first, stop)
            if self.dtype.kind in "iu":
                output[first:stop] = self._integer_sum(values, selected)
            else:
                values = (
                    values.astype(dtype) if self.backend == "cuda" else values.to(dtype)
                )
                output[first:stop] = self._sum(self._where(selected, values), (1, 2))
            del values
        return output.reshape(self.shape[:2])

    @_device_scoped
    def products_device(self, mask=None, *, moments=False):
        """Compute floating detector moments on the device in bounded windows."""
        if not moments:
            return (self.detector_sum_device(mask),)
        selected = self._mask(mask)
        outputs = [self._empty((math.prod(self.shape[:2]),)) for _ in range(3)]
        row = self._array(np.arange(self.detector_shape[0], dtype=np.float32)[:, None])
        column = self._array(
            np.arange(self.detector_shape[1], dtype=np.float32)[None, :]
        )
        self._lanes.check_readable(0, math.prod(self.shape[:2]))
        for chunk in self._lanes.chunks:
            first, stop = chunk.first, chunk.first + chunk.scans
            values = self._tensor(first, stop)
            values = (
                values.astype(np.float32)
                if self.backend == "cuda"
                else values.to(torch.float32)
            )
            values = self._where(selected, values)
            for output, weights in zip(outputs, (1, row, column)):
                output[first:stop] = self._sum(values * weights, (1, 2))
        return tuple(output.reshape(self.shape[:2]) for output in outputs)

    @_device_scoped
    def mean_dp_device(self):
        """Return a float32 mean DP for display; original measurements stay typed."""
        self._check()
        self._lanes.check_readable(0, math.prod(self.shape[:2]))
        if self._mean is None:
            if self.dtype.kind in "iu":
                total = self.detector_total_device()
                if self.backend == "cuda":
                    self._mean = (
                        total.astype(np.float64) / math.prod(self.shape[:2])
                    ).astype(np.float32)
                else:
                    from quantem.gpu.resident.mps.integer_mean import integer_mean

                    self._mean = integer_mean(total, math.prod(self.shape[:2]))
            else:
                self._mean = self.reduce_frames_device(range(math.prod(self.shape[:2])))
        return self._mean.copy() if self.backend == "cuda" else self._mean.clone()

    @_device_scoped
    def reduce_frames_device(self, indices, mode="mean"):
        """Reduce selected patterns to float32 on GPU, with bounded decoded storage.

        These display products use floating arithmetic; use raw reads or the
        integer detector reduction when exact integer output is required.
        """
        self._check()
        indices = np.asarray(list(indices))
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or indices.min() < 0
            or indices.max() >= math.prod(self.shape[:2])
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(
                "Select distinct integer scan positions within the acquisition."
            )
        if mode not in ("mean", "sum", "max"):
            raise ValueError("Use mean, sum or max for selected patterns.")
        self._lanes.check_readable(int(indices.min()), int(indices.max()) + 1)
        if self.dtype.kind in "iu" and mode != "max":
            result = self.detector_total_device(indices)
            if mode == "sum":
                return result
            if self.backend == "cuda":
                return (result.astype(np.float64) / len(indices)).astype(np.float32)
            from quantem.gpu.resident.mps.integer_mean import integer_mean

            return integer_mean(result, len(indices))
        result = None
        for chunk in self._lanes.chunks:
            local = (
                indices[
                    (indices >= chunk.first) & (indices < chunk.first + chunk.scans)
                ]
                - chunk.first
            )
            if not len(local):
                continue
            values = self._tensor(chunk.first, chunk.first + chunk.scans)
            if self.dtype.kind in "iu" and mode == "max":
                if self.backend == "cuda":
                    import cupy as cp

                    reduced = values[cp.asarray(local)].max(axis=0)
                    result = reduced if result is None else cp.maximum(result, reduced)
                else:
                    unsigned64 = values.dtype == torch.uint64
                    ordered = (
                        (values.view(torch.int64) ^ -(1 << 63))
                        if unsigned64
                        else values.to(torch.int64)
                    )
                    reduced = ordered[self._array(local.astype(np.int64))].amax(dim=0)
                    result = (
                        reduced if result is None else torch.maximum(result, reduced)
                    )
                continue
            if self.backend == "cuda":
                import cupy as cp

                values = values.astype(cp.float64)[cp.asarray(local)]
                reduced = values.max(axis=0) if mode == "max" else values.sum(axis=0)
                result = (
                    reduced
                    if result is None
                    else (
                        cp.maximum(result, reduced)
                        if mode == "max"
                        else result + reduced
                    )
                )
            else:
                values = values.to(torch.float32)[self._array(local.astype(np.int64))]
                reduced = values.amax(dim=0) if mode == "max" else values.sum(dim=0)
                result = (
                    reduced
                    if result is None
                    else (
                        torch.maximum(result, reduced)
                        if mode == "max"
                        else result + reduced
                    )
                )
            del values
        if mode == "mean":
            result /= len(indices)
        if self.dtype.kind in "iu":
            if self.backend == "mps":
                result = (
                    (result ^ -(1 << 63)).view(torch.uint64)
                    if self.dtype == np.dtype("uint64")
                    else result.to(getattr(torch, self.dtype.name))
                )
            return result
        return result.astype(np.float32) if self.backend == "cuda" else result

    def release_scans_before(self, stop: int) -> None:
        """Free the encoded chunks that end by scan ``stop``; reading a freed scan raises."""
        self._check()
        self._lanes.release_scans_before(stop)

    def release(self) -> None:
        """Free the encoded lanes and cached products after pending GPU work finishes."""
        if not self.is_released:
            self.synchronize()
            self._lanes.release()
            self._lanes = None
            self._background = self._mean = None
            self.is_released = True
