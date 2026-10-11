"""Portable float32 bit-lane residents with bounded accelerator decoding."""

import copy
import math
from contextlib import nullcontext
from functools import wraps

import numpy as np
import torch

from quantem.gpu.device.cuda_runtime import cuda_device_index
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


class FloatANSResident:
    """Own encoded IEEE bits; materialize at most a 32 MiB scan window.

    File measurements are never cast to counts. Raw reads preserve every bit;
    scientific products apply the recorded mean-dark subtraction exactly once.
    Use ``io.load(path, backend='mps')`` or ``backend='cuda'`` to construct one.
    """

    dtype = np.dtype("float32")
    interval = 512

    def __init__(self, header: dict, backend: str, device: int | str | None = None):
        self.header = copy.deepcopy(header)
        self.shape = tuple(header["shape"])
        self.backend = backend
        self.is_released = False
        self.valid_pixels = np.ones(self.shape[2:], bool)
        self._background = None
        self.peak_decode_bytes = 0
        self._mean = None
        self.detector_shape = self.shape[2:]
        if self.frame_bytes > MAX_DECODE_BYTES:
            raise ValueError(
                "One float32 diffraction frame exceeds the 32 MiB decode budget."
            )
        lane_shape = (*self.shape[:3], self.shape[3] * 2)
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
        return math.prod(self.shape[2:]) * 4

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
        return math.prod(self.shape) * 4

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "Float ANS stays encoded; request a DP, product or bounded scan window."
        )

    def device_context(self):
        if self.backend == "cuda":
            import cupy as cp

            return cp.cuda.Device(self.device)
        return nullcontext()

    def _check(self, first=None, stop=None):
        """Refuse a released resident and decode ranges beyond the 32 MiB window."""
        if self.is_released:
            raise RuntimeError("The float ANS resident was released; load it again.")
        if first is not None:
            if (
                type(first) is not int
                or type(stop) is not int
                or not 0 <= first < stop <= math.prod(self.shape[:2])
            ):
                raise ValueError("Choose a nonempty scan range within the acquisition.")
            size = (stop - first) * self.frame_bytes
            if size > MAX_DECODE_BYTES:
                raise ValueError(
                    f"Decode requests are limited to 32 MiB; use at most {MAX_DECODE_BYTES // self.frame_bytes} frames."
                )
            self.peak_decode_bytes = max(self.peak_decode_bytes, size)

    def decode_scan_range_device(self, first: int, stop: int):
        """Return original float32 measurement bits in one bounded device buffer.

        The buffer is a CuPy array on CUDA and a Metal array on MPS.
        """
        self._check(first, stop)
        output = self._lanes.decode_scan_range_device(first, stop)
        if self.backend == "cuda":
            return output.view(np.float32).reshape(stop - first, *self.detector_shape)
        output.dtype = self.dtype
        output.shape = (stop - first, *self.detector_shape)
        return output

    def extract_diffraction_device(self, scan_row: int, scan_column: int):
        """Return one original DP without applying a display correction."""
        if not 0 <= scan_row < self.shape[0] or not 0 <= scan_column < self.shape[1]:
            raise IndexError("Choose a scan row and column within the acquisition.")
        first = scan_row * self.shape[1] + scan_column
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
            output = self._lanes.decode_scan_range_torch(first, stop)
            output = output.view(torch.float32).reshape(
                stop - first, *self.detector_shape
            )
        if corrected and self._background is not None:
            output -= self._background
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

    @_device_scoped
    def products_device(
        self, mask: np.ndarray | None = None, *, moments: bool = False
    ):
        """Reduce bounded windows; scale weights before computing CoM moments.

        Returns ``(total,)`` or ``(total, row_moment, column_moment)`` scan
        images on the GPU.
        """
        self._check()
        mask = np.ones(self.detector_shape, bool) if mask is None else np.asarray(mask)
        if mask.shape != self.detector_shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                f"Float detector masks must be binary with shape {self.detector_shape}."
            )
        selected = self._array(mask.astype(bool))
        total = self._empty((math.prod(self.shape[:2]),))
        products = [total]
        if moments:
            products += [self._empty(total.shape), self._empty(total.shape)]
            row = self._array(
                np.arange(self.detector_shape[0], dtype=np.float32)[:, None]
            )
            column = self._array(
                np.arange(self.detector_shape[1], dtype=np.float32)[None, :]
            )
        for chunk in self._lanes.chunks:
            first, stop = chunk.first, chunk.first + chunk.scans
            values = self._tensor(first, stop)
            values = self._where(selected, values)
            if moments:
                magnitude = (
                    abs(values).max(axis=(1, 2), keepdims=True)
                    if self.backend == "cuda"
                    else values.abs().amax(dim=(1, 2), keepdim=True)
                )
                # Preserve ordinary dyadic measurements exactly; rescale only
                # extreme exponents which can overflow weighted moments or
                # underflow. Unconditional division perturbs zero signed sums.
                extreme = (magnitude > 2.0**100) | (
                    (magnitude > 0) & (magnitude < 2.0**-100)
                )
                values /= self._where(extreme, magnitude, 1)
            total[first:stop] = self._sum(values, (1, 2))
            if moments:
                products[1][first:stop] = self._sum(values * row, (1, 2))
                products[2][first:stop] = self._sum(values * column, (1, 2))
            del values
        return tuple(value.reshape(self.shape[:2]) for value in products)

    def detector_sum_device(self, mask: np.ndarray):
        """Sum each scan position over a binary detector mask, after dark subtraction."""
        self._check()
        values = np.asarray(mask)
        if values.shape != self.detector_shape or not np.all(
            (values == 0) | (values == 1)
        ):
            raise ValueError(
                f"Float detector masks must be binary with shape {self.detector_shape}."
            )
        size = max(chunk.scans for chunk in self._lanes.chunks) * self.frame_bytes
        self.peak_decode_bytes = max(self.peak_decode_bytes, size)
        return self._lanes.float_detector(values, self._background)

    def _pattern_total(self, selected=None) -> np.ndarray:
        """Sum the stored patterns per detector pixel, every scan or those flagged in ``selected``; float64 on the host.

        CUDA accumulates in float64; Metal, which has none, in compensated
        float32 that adds up to the same total. A float32 running total drifts:
        over the 16384 patterns of a 128 x 128 scan it was up to 8 units in the
        last place off the float64 mean. ``selected`` is a uint8 flag per scan.
        """
        self.peak_decode_bytes = max(self.peak_decode_bytes, max(chunk.scans for chunk in self._lanes.chunks) * self.frame_bytes)
        if self.backend == "mps":
            return self._lanes.pattern_total(selected, self._background).reshape(self.detector_shape)
        import cupy as cp

        total = cp.zeros(self.detector_shape, cp.float64)
        for chunk in self._lanes.chunks:
            if selected is None:
                total += self._tensor(chunk.first, chunk.first + chunk.scans).sum(axis=0, dtype=cp.float64)
                continue
            local = np.flatnonzero(selected[chunk.first : chunk.first + chunk.scans])
            if local.size:
                window = self._tensor(chunk.first + int(local[0]), chunk.first + int(local[-1]) + 1)
                total += window[cp.asarray(local - local[0])].sum(axis=0, dtype=cp.float64)
        return total.get()

    @_device_scoped
    def mean_dp_device(self):
        """Compute and cache the full-scan mean on the accelerator.

        The float64 total divided by the scan count and rounded once to
        float32, the rule of every mean pattern. It covers every scan
        position, so it raises once rows were freed: the rows left, divided by
        the full scan count, are not a mean.
        """
        self._check()
        self._lanes.check_readable(0, math.prod(self.shape[:2]))
        if self._mean is None:
            self._mean = self._array((self._pattern_total() / math.prod(self.shape[:2])).astype(np.float32))
        return self._mean.copy() if self.backend == "cuda" else self._mean.clone()

    @_device_scoped
    def reduce_frames_device(self, indices, mode="mean"):
        """Reduce selected scan positions with bounded GPU working storage.

        ``sum`` and ``mean`` take the float64 total and round once to float32,
        so the mean over every scan position equals ``mean_dp_device``. A
        selected scan that was released raises: its chunk is gone, and
        skipping it gave a mean over fewer patterns than it was divided by.
        """
        self._check()
        indices = np.asarray(list(indices))
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or indices.min() < 0
            or indices.max() >= math.prod(self.shape[:2])
        ):
            raise ValueError("Select integer scan positions within the acquisition.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Select each scan position once for a region reduction.")
        if mode not in ("sum", "mean", "max"):
            raise ValueError("Use mean, sum or max for the selected DPs.")
        self._lanes.check_readable(int(indices.min()), int(indices.max()) + 1)
        if mode != "max":
            selected = np.zeros(math.prod(self.shape[:2]), np.uint8)
            selected[indices] = 1
            total = self._pattern_total(selected)
            return self._array((total / len(indices) if mode == "mean" else total).astype(np.float32))
        maximum = None
        for chunk in self._lanes.chunks:
            local = indices[(indices >= chunk.first) & (indices < chunk.first + chunk.scans)] - chunk.first
            if not len(local):
                continue
            first, stop = int(local.min()), int(local.max()) + 1
            values = self._tensor(chunk.first + first, chunk.first + stop)
            selected = values[self._array((local - first).astype(np.int64))]
            reduced = selected.max(axis=0) if self.backend == "cuda" else selected.amax(dim=0)
            if maximum is None:
                maximum = reduced
            elif self.backend == "cuda":
                import cupy as cp

                maximum = cp.maximum(maximum, reduced)
            else:
                maximum = torch.maximum(maximum, reduced)
        return maximum

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
