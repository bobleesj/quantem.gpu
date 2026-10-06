"""Bounded views of encoded acquisitions reduce on the native session, equal to reducing their reads.

quantem.widget's live viewer hands ``detector.prepare`` a bounded reader over
an acquisition from ``io.load``; ``View`` below follows that reader's contract
(``read(scan_region=...)``, ``valid``, ``_detector_source``, ``_detector_region``).
The encoded session answers the mean pattern and frame reductions without
reading every scan position; the reads give the reference.
"""

import os

import numpy as np
import pytest
import torch

from quantem.gpu import detector
from quantem.gpu.detector.bounded import BoundedDetectorCompute
from tests.hardware.test_encoded_detector_moments import _counts, _load, _valid


class View:
    """A scan region of a loaded acquisition, read with flagged detector pixels set to 0."""

    _bounded_detector_source = True

    def __init__(self, source, region):
        self._detector_source = source
        self._detector_region = region
        row_start, row_stop, col_start, col_stop = region
        self.shape = (row_stop - row_start, col_stop - col_start, *source.shape[2:])
        self.device = torch.device(source.device)
        self.valid = torch.as_tensor(source.data.valid_pixels, device=self.device).reshape(self.shape[2:])
        self.reads = 0

    def read(self, *, scan_region):
        self.reads += 1
        row_start, row_stop, col_start, col_stop = scan_region
        row_offset, _, col_offset, _ = self._detector_region
        values_t = self._detector_source.read(
            scan_region=(row_start + row_offset, row_stop + row_offset, col_start + col_offset, col_stop + col_offset)
        )
        return values_t.float().masked_fill(~self.valid, 0)


@pytest.mark.parametrize("region", [(0, 32, 0, 32), (4, 20, 9, 30)], ids=["whole", "region"])
def test_native_reductions_equal_reductions_of_reads(tmp_path, region):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    counts = _counts(31)
    row_start, row_stop, col_start, col_stop = region
    viewed = (counts[row_start:row_stop, col_start:col_stop] * _valid()).reshape(-1, *counts.shape[2:])
    with _load(tmp_path, counts, backend) as loaded:
        view = View(loaded, region)
        native = detector.prepare(view)
        indices = [0, 7, 7, 40, len(viewed) - 1]
        mean_dp = native.mean_dp()
        reduced = {mode: native.reduce_frames(indices, mode) for mode in ("mean", "sum", "max")}
        assert view.reads == 0
        reads = BoundedDetectorCompute(view, None)
        np.testing.assert_array_equal(mean_dp, reads.mean_dp())
        # The exact total divided in float64 and rounded once, like every mean pattern.
        np.testing.assert_array_equal(mean_dp, (viewed.sum(0) / len(viewed)).astype(np.float32))
        for mode, values in reduced.items():
            np.testing.assert_array_equal(values, reads.reduce_frames(indices, mode))
        np.testing.assert_array_equal(reduced["max"], viewed[indices].max(0))


class FloatView(View):
    """A scan region of a saved float acquisition (a MAPED merge), which has no flagged pixels."""

    def __init__(self, source, region):
        self._detector_source = source
        self._detector_region = region
        row_start, row_stop, col_start, col_stop = region
        self.shape = (row_stop - row_start, col_stop - col_start, *source.shape[2:])
        self.device = torch.device(source.device)
        self.valid = None
        self.reads = 0

    def read(self, *, scan_region):
        self.reads += 1
        row_start, row_stop, col_start, col_stop = scan_region
        row_offset, _, col_offset, _ = self._detector_region
        return self._detector_source.read(
            scan_region=(row_start + row_offset, row_stop + row_offset, col_start + col_offset, col_stop + col_offset)
        ).float()


@pytest.mark.parametrize("region", [(0, 8, 0, 9), (1, 7, 2, 9)], ids=["whole", "region"])
def test_float_precision_views_reduce_on_the_session(tmp_path, region):
    """Show4DSTEM opens MAPED merges: saved float intensities have no exact integer total."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    from quantem.gpu import io

    intensities = np.random.default_rng(5).gamma(2.0, 3.0, (8, 9, 12, 10)).astype(np.float32)
    path = tmp_path / "merged_master.h5"
    io.save(path, intensities, dtype="scaled_uint16", backend=backend, verbose=False)
    with io.load(path, backend=backend, verbose=False) as loaded:
        view = FloatView(loaded, region)
        native = detector.prepare(view)
        indices = [0, 3, 3, 10]
        mean_dp = native.mean_dp()
        reduced = {mode: native.reduce_frames(indices, mode) for mode in ("mean", "sum", "max")}
        assert view.reads == 0
        reads = BoundedDetectorCompute(view, None)
        np.testing.assert_allclose(mean_dp, reads.mean_dp(), rtol=1e-6)
        for mode, values in reduced.items():
            np.testing.assert_allclose(values, reads.reduce_frames(indices, mode), rtol=1e-6)
