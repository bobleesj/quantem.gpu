"""Contract tests for the multi-acquisition MPS ANS detector adapter."""

import numpy as np
import pytest

from quantem.gpu import detector
from quantem.gpu.io.backends.mps._ans import MPSANSResidentCounts


class _Output:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.released = False

    def to_numpy(self):
        return self.values.copy()

    def release(self):
        self.released = True


class _Source(MPSANSResidentCounts):
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.uint16)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.is_released = False
        self.outputs = []

    def extract_diffraction_device(self, row, column):
        output = _Output(self.values[row, column])
        self.outputs.append(output)
        return output

    def detector_sums_device(self, masks):
        values = np.asarray(masks, dtype=np.uint8)
        images = np.stack(
            [(self.values * mask).sum(axis=(2, 3), dtype=np.uint64) for mask in values]
        )
        output = _Output(images)
        self.outputs.append(output)
        return output


def test_mps_ans_series_overlaps_sources_and_preserves_axes():
    values = np.arange(2 * 3 * 2 * 3, dtype=np.uint16).reshape(2, 3, 2, 3)
    first, second = _Source(values), _Source(values + 100)
    session = detector.prepare([first, second])

    assert session.series_shape == (2,)
    np.testing.assert_array_equal(
        session.frame(4), np.stack([values[1, 1], (values + 100)[1, 1]])
    )
    masks = np.stack([np.ones((2, 3), dtype=np.uint8), np.eye(2, 3, dtype=np.uint8)])
    expected = np.stack(
        [
            np.stack([(source.values * mask).sum(axis=(2, 3), dtype=np.uint64) for mask in masks])
            for source in (first, second)
        ],
        axis=1,
    )
    np.testing.assert_array_equal(session.masked_sums_exact(masks), expected)
    assert all(output.released for source in (first, second) for output in source.outputs)


def test_mps_ans_series_rejects_different_geometry():
    first = _Source(np.zeros((2, 3, 2, 3), dtype=np.uint16))
    second = _Source(np.zeros((2, 3, 3, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="same shape"):
        detector.prepare([first, second])
