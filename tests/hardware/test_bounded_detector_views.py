"""Bounded views reduce on the session of the acquisition they view, never by reading every scan position.

quantem.widget's live viewer hands ``detector.prepare`` a bounded reader over
an acquisition from ``io.load`` or over a GPU tensor; ``View`` below follows
that reader's contract (``read(scan_region=...)``, ``valid``,
``_detector_source``, ``_detector_region``) and counts its reads. Whatever the
acquisition holds (encoded counts, float intensities, scaled precision, paired
counts, a dense tensor), the mean pattern, virtual images and frame reductions
of the whole scan or of one scan region come from its own session with zero
reads, and equal NumPy on the viewed values with the flagged pixels zeroed.
"""

import os

import numpy as np
import pytest
import torch

from quantem.gpu import detector, io
from quantem.gpu.io import Dataset4dstemGPU
from tests.hardware.test_encoded_detector_moments import DETECTOR_SHAPE, FLAGGED, SCAN_SHAPE, _counts, _load, _valid
from tests.hardware.test_screening_products import write_master


class View:
    """A scan region of an acquisition or GPU tensor, read as float32 with flagged detector pixels set to 0."""

    _bounded_detector_source = True

    def __init__(self, source, region, valid=None):
        self._detector_source = source
        self._detector_region = region
        row_start, row_stop, col_start, col_stop = region
        self.shape = (row_stop - row_start, col_stop - col_start, *source.shape[2:])
        self.device = torch.device(source.device)
        if valid is None and isinstance(source, Dataset4dstemGPU) and source.representation == "encoded":
            valid = source.data.valid_pixels
        self.valid = None if valid is None else torch.as_tensor(np.asarray(valid, dtype=bool), device=self.device).reshape(self.shape[2:])
        self.reads = 0

    def read(self, *, scan_region):
        self.reads += 1
        row_start, row_stop, col_start, col_stop = scan_region
        row_offset, _, col_offset, _ = self._detector_region
        rows, cols = slice(row_start + row_offset, row_stop + row_offset), slice(col_start + col_offset, col_stop + col_offset)
        if torch.is_tensor(self._detector_source):
            values_t = self._detector_source[rows, cols].float()
        else:
            values_t = self._detector_source.read(scan_region=(rows.start, rows.stop, cols.start, cols.stop)).float()
        return values_t if self.valid is None else values_t.masked_fill(~self.valid, 0)


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return backend


def _assert_session_reductions(view, values, *, exact_images=True):
    """Every reduction of ``view`` equals NumPy on ``values`` (the whole scan) and costs no bounded read."""
    row_start, row_stop, col_start, col_stop = view._detector_region
    valid = np.ones(values.shape[2:], bool) if view.valid is None else view.valid.cpu().numpy()
    viewed = (values[row_start:row_stop, col_start:col_stop] * valid).reshape(-1, *values.shape[2:])
    wide = viewed.astype(np.float64)
    session = detector.prepare(view)
    indices = np.array([0, 7, 40, len(viewed) // 2, len(viewed) - 1])
    mask = np.zeros(values.shape[2:], bool)
    mask[2:9, 1:8] = True
    mean_dp = session.mean_dp()
    image = session.masked_sum(mask)
    reduced = {mode: session.reduce_frames(indices, mode) for mode in ("mean", "sum", "max")}
    assert view.reads == 0
    # The total divided in float64 and rounded once, like every mean pattern.
    np.testing.assert_array_equal(mean_dp, (wide.sum(axis=0) / len(wide)).astype(np.float32))
    np.testing.assert_array_equal(reduced["mean"], (wide[indices].sum(axis=0) / len(indices)).astype(np.float32))
    np.testing.assert_array_equal(reduced["sum"], wide[indices].sum(axis=0).astype(np.float32))
    np.testing.assert_array_equal(reduced["max"], viewed[indices].max(axis=0).astype(np.float32))
    expected_image = wide[:, mask].sum(axis=1).astype(np.float32).reshape(view.shape[:2])
    if exact_images:
        np.testing.assert_array_equal(image, expected_image)
    else:
        np.testing.assert_allclose(image, expected_image, rtol=1e-6)
    assert all(result.dtype == np.float32 for result in (mean_dp, image, *reduced.values()))
    # One pattern is the one thing a bounded reader is read for.
    np.testing.assert_array_equal(session.frame(int(indices[2])), viewed[indices[2]])
    assert view.reads == 1


@pytest.mark.parametrize("region", [(0, 32, 0, 32), (4, 20, 9, 30)], ids=["whole", "region"])
def test_encoded_counts(tmp_path, region):
    backend = _backend()
    counts = _counts(31)
    with _load(tmp_path, counts, backend) as loaded:
        _assert_session_reductions(View(loaded, region), counts)


@pytest.mark.parametrize("region", [(0, 32, 0, 32), (4, 20, 9, 30)], ids=["whole", "region"])
def test_bright_counts_keep_exact_totals_above_float32(tmp_path, region):
    """1024 patterns near full scale sum past 2^24: a float32 total of the reads loses counts, the session does not."""
    backend = _backend()
    counts = np.random.default_rng(37).integers(52_000, 65_536, (*SCAN_SHAPE, *DETECTOR_SHAPE)).astype(np.uint16)
    row_start, row_stop, col_start, col_stop = region
    viewed = (counts[row_start:row_stop, col_start:col_stop] * _valid()).reshape(-1, *DETECTOR_SHAPE)
    total = viewed.sum(axis=0, dtype=np.uint64)
    running = np.zeros(DETECTOR_SHAPE, np.float32)
    for pattern in viewed:
        running += pattern
    assert int(total[total > 0].min()) > 2**24
    assert np.any(running != total)
    with _load(tmp_path, counts, backend) as loaded:
        _assert_session_reductions(View(loaded, region), counts)


@pytest.mark.parametrize("region", [(0, 40, 0, 37), (3, 31, 5, 36)], ids=["whole", "region"])
@pytest.mark.parametrize("masked", [False, True], ids=["all-valid", "flagged"])
def test_float32_intensities(tmp_path, region, masked):
    """Float files (EMPAD, simulations) load as float ANS: their means come out at float64 accuracy."""
    backend = _backend()
    rng = np.random.default_rng(41)
    intensities = (rng.gamma(4.0, 2.5e4, (40, 37, 12, 10)) + rng.normal(0.0, 3.0, (40, 37, 12, 10))).astype(np.float32)
    np.save(tmp_path / "intensities.npy", intensities)
    valid = np.ones((12, 10), bool)
    if masked:
        valid[3, 4] = valid[8, 1] = False
    with io.load(tmp_path / "intensities.npy", backend=backend) as loaded:
        assert loaded.dtype == np.float32
        _assert_session_reductions(View(loaded, region, valid), intensities)


@pytest.mark.parametrize("region", [(0, 8, 0, 9), (1, 7, 2, 9)], ids=["whole", "region"])
def test_scaled_precision_intensities(tmp_path, region):
    """Show4DSTEM opens MAPED merges: saved scaled intensities reduce from their codes, close to reducing the reads."""
    backend = _backend()
    intensities = np.random.default_rng(5).gamma(2.0, 3.0, (8, 9, 12, 10)).astype(np.float32)
    path = tmp_path / "merged_master.h5"
    io.save(path, intensities, dtype="scaled_uint16", backend=backend, verbose=False)
    row_start, row_stop, col_start, col_stop = region
    with io.load(path, backend=backend, verbose=False) as loaded:
        restored = loaded.read().cpu().numpy()[row_start:row_stop, col_start:col_stop].reshape(-1, 12, 10).astype(np.float64)
        view = View(loaded, region)
        session = detector.prepare(view)
        indices = [0, 3, 3, 10]
        mean_dp = session.mean_dp()
        reduced = {mode: session.reduce_frames(indices, mode) for mode in ("mean", "sum", "max")}
        assert view.reads == 0
        np.testing.assert_allclose(mean_dp, restored.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(reduced["mean"], restored[indices].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(reduced["sum"], restored[indices].sum(axis=0), rtol=1e-6)
        np.testing.assert_allclose(reduced["max"], restored[indices].max(axis=0), rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("region", [(0, 32, 0, 32), (4, 20, 9, 30)], ids=["whole", "region"])
@pytest.mark.parametrize("wrapped", [True, False], ids=["acquisition", "tensor"])
def test_dense_tensors(dtype, region, wrapped):
    """A tensor already on the GPU (a strided preview, a processed scan) reduces in Torch on its device."""
    backend = _backend()
    values = np.random.default_rng(43).integers(52_000, 65_536, (*SCAN_SHAPE, *DETECTOR_SHAPE)).astype(np.uint16)
    values = values if dtype == np.uint16 else (values / 7.0).astype(np.float32)
    values_t = torch.as_tensor(values, device=backend)
    source = Dataset4dstemGPU(values_t, {"name": "preview"}) if wrapped else values_t
    valid = np.ones(DETECTOR_SHAPE, bool)
    valid[FLAGGED] = False
    # A float virtual image is a float32 sum over the mask pixels; counts stay exact.
    _assert_session_reductions(View(source, region, valid), values, exact_images=dtype == np.uint16)


@pytest.mark.parametrize("region", [(0, 32, 0, 32), (4, 20, 9, 30)], ids=["whole", "region"])
def test_paired_counts(tmp_path, region):
    if _backend() != "cuda":
        pytest.skip("The paired layout is a CUDA resident.")
    counts = _counts(47)
    pixel_mask = np.zeros(DETECTOR_SHAPE, dtype=np.uint32)
    pixel_mask[FLAGGED] = 1
    master = write_master(tmp_path, counts, pixel_mask)
    with io.load(master, representation="paired", hot_pixel_correction="none", verbose=False) as loaded:
        _assert_session_reductions(View(loaded, region, _valid().astype(bool)), counts)
