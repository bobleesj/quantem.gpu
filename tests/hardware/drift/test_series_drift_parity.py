"""quantem.gpu.drift reproduces myptycho's translation drift exactly on real data: MoS2 bilayer frames 151-191
streamed through SSB, against myptycho's outputs on the same phases (time/tests/drift_baseline.py). Skipped where that
frozen baseline folder is absent."""

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
    np.testing.assert_array_equal(found.sigma_A, baseline["series_drift_sigma"])
    np.testing.assert_array_equal(found.closure_A, baseline["series_drift_closure"])


def test_series_running_matches_myptycho(series):
    images, sampling_A, baseline = series
    np.testing.assert_array_equal(running(images, sampling_A=sampling_A).displacement_A, baseline["series_running_displacement"])


def test_series_register_first_to_last_matches_myptycho(series):
    images, sampling_A, baseline = series
    np.testing.assert_array_equal(np.array(register(images[0], images[-1], sampling_A=sampling_A).shift_px), baseline["series_register_151_191_shift"])
