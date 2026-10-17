"""Float32 products from exact GPU bit-lane residents."""

import math

import numpy as np
import torch

from quantem.gpu.resident.bit_ans import MAX_DECODE_BYTES, BitANSResident, _device_scoped


class FloatANSResident(BitANSResident):
    """Preserve IEEE float32 bits and apply a recorded mean-dark plane in products."""

    dtype = np.dtype("float32")

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
