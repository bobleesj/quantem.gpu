"""quantem.gpu.drift reproduces myptycho's translation drift exactly on real data: MoS2 bilayer frames 151-191
streamed through SSB, against myptycho's outputs on the same phases (time/tests/drift_baseline.py). Skipped where that
frozen baseline folder is absent (every host but the one that holds it) and marked slow, so run it with -m slow: it is the only parity
check where consistent rejects pairs (10 of 114), the synthetic series keeping all of its pairs."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.gpu.drift import drift, register, running

BASELINE = Path(os.environ.get("QUANTEM_DRIFT_BASELINE", "/nonexistent"))

pytestmark = [pytest.mark.skipif(not (BASELINE / "baseline.npz").exists() or not torch.cuda.is_available(), reason="needs the frozen drift baseline and a GPU"),
              pytest.mark.slow]


@pytest.fixture(scope="module")
def series():
    baseline = np.load(BASELINE / "baseline.npz")
    phases = np.load(BASELINE / "series_phases.npz")["phase"]
    return torch.as_tensor(phases, device="cuda"), tuple(float(value) for value in baseline["series_sampling_A"]), baseline


def test_series_drift_matches_myptycho(series):
    images, sampling_A, baseline = series
    found = drift(images, sampling_A=sampling_A)
    np.testing.assert_array_equal(found.displacement_A, baseline["series_drift_displacement"])
    # refine counts its offsets in whole 1/100 px where myptycho stepped a float arange: σ and closure move in the
    # last bit (3e-18 Å), the displacements not at all
    np.testing.assert_allclose(found.sigma_A, baseline["series_drift_sigma"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(found.closure_A, baseline["series_drift_closure"], rtol=0, atol=1e-12)


def test_series_running_matches_myptycho(series):
    """running correlates by quantem.gpu.parallax's cross_correlation_shift_batch_cp, as the baseline did at quantem.gpu
    a26c62fa: a failure here after a change there is that change, not drift's."""
    images, sampling_A, baseline = series
    np.testing.assert_array_equal(running(images, sampling_A=sampling_A).displacement_A, baseline["series_running_displacement"])


def test_series_register_first_to_last_matches_myptycho(series):
    images, sampling_A, baseline = series
    np.testing.assert_array_equal(np.array(register(images[0], images[-1], sampling_A=sampling_A).shift_px), baseline["series_register_151_191_shift"])
