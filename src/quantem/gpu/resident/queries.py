"""The detector queries answered by every object that ``detector.prepare`` computes on.

The detector backends and the scaled-precision residents, which answer detector
queries on their own encoded data, share this surface. Each defines
``scan_shape``, ``det_shape`` and ``n_frames`` and implements ``frame``,
``mean_dp``, ``masked_sum``, ``reduce_frames`` and ``center_of_mass``; the
optional operations below raise until a source overrides them, so
``DetectorSession`` calls every operation directly and an unsupported request
says what is missing.
"""

from types import MappingProxyType

import numpy as np


class DetectorQueries:
    """Default answers for the optional detector operations."""

    # A single 4D acquisition has no series axis.
    series_shape: tuple[int, ...] = ()
    # Detector pixels the source counts, bool ``(*series_shape, row, col)``: every exact
    # product reads a flagged pixel as 0. None when the data carry no flags (dense arrays).
    valid_pixels: np.ndarray | None = None
    # Read-only defaults; sources that record their identity or timings assign their own dicts.
    backend_metadata = MappingProxyType({})
    last = MappingProxyType({})

    def frame_native(self, index, *, out=None, wait=True):
        raise NotImplementedError("This backend has no native frame output; use output='numpy'.")

    def masked_sum_native(self, mask, *, out=None, wait=True, block_stride=1):
        raise NotImplementedError("This backend has no native detector output; use output='numpy'.")

    def mean_dp_native(self):
        """Return the mean pattern without a host copy.

        A source whose ``mean_dp`` already stays on its device returns that
        result; a source that reduces on the host has no native output.
        """
        result = self.mean_dp()
        if isinstance(result, np.ndarray):
            raise NotImplementedError("This backend has no native mean output; use output='numpy'.")
        return result

    def masked_sum_exact(self, mask):
        raise TypeError("Exact detector sums require integer detector data; this source holds float intensities.")

    def masked_sum_exact_native(self, mask, *, out=None):
        """Exact integer image in the source's sum dtype on its device.

        Only CUDA count series keep exact sums on the device; a source holding
        float intensities raises ``TypeError`` instead, as its host exact sum does.
        """
        raise NotImplementedError("This backend has no native exact detector output; use output='numpy'.")

    def reduce_frames_exact(self, indices):
        raise NotImplementedError("This compute backend has no exact selected-frame reducer.")

    def reduce_frames_max(self, indices):
        raise NotImplementedError("This compute backend has no exact selected-frame maximum.")

    def weighted_sum_exact(self, weights):
        raise NotImplementedError("This backend has no exact weighted detector sum.")

    def detector_total(self):
        raise NotImplementedError(
            "This backend has no exact detector total; use reduce_frames_exact over every scan index."
        )

    def finish(self) -> dict:
        """Timings of the last query; only streamed series queue queries to finish later."""
        return dict(self.last)


# ---


def weight_digits(weights: np.ndarray, count_max: int):
    """Split nonnegative integer detector weights into binary digits that sum exactly.

    The CUDA and Metal residual decoders add ``weight * count`` over 32 pixels
    in int32, so ``32 * weight * count`` must stay below 2^31: a weight up to
    1023 for uint16 counts, which a column weight on a detector wider than 1024
    pixels exceeds. Each digit holds the most bits within that bound for counts
    up to ``count_max`` (10 for uint16, 18 for uint8), and
    ``weights = sum(digit << shift)``, so the exact sums of one pass per digit,
    shifted left and added, give the exact weighted sum.

    Yields ``(shift, digit)`` for every digit with a nonzero pixel, digits as
    int64 arrays shaped like ``weights``.
    """
    bits = ((2**31 - 1) // (32 * int(count_max)) + 1).bit_length() - 1
    values = np.asarray(weights, dtype=np.int64)
    for shift in range(0, max(int(values.max(initial=0)), 1).bit_length(), bits):
        digit = (values >> shift) & ((1 << bits) - 1)
        if digit.any():
            yield shift, digit
