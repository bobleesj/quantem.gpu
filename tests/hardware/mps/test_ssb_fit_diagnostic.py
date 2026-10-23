"""Model fitting can omit its separate diagnostic without changing the fit."""

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps
from .test_ssb_upsampling import _disk_session


@pytest.mark.parametrize("tilt", [False, True])
def test_fit_without_diagnostic_retains_fit_phase_and_later_loss(tilt):
    """Fit for interactive review, then request the identical diagnostic later."""
    _require_mps()
    options = dict(tilt=tilt, num_trials=8, refinement="newton", seed=11,
                   check_rotation=False, compare_standard=False, verbose=False)
    with _disk_session(n=32) as reference, _disk_session(n=32) as session:
        expected = reference.find_aberrations(search="standard", **options)
        actual = session.find_aberrations(search="standard", **options, compute_loss=False)
        assert actual.loss is None
        assert actual.aberrations == expected.aberrations
        assert actual.sample_tilt_mrad == expected.sample_tilt_mrad
        assert actual.depth_spread_nm == expected.depth_spread_nm
        assert actual.sample_agreement == expected.sample_agreement
        assert actual.trial_records == expected.trial_records
        assert actual.optimizer == expected.optimizer
        np.testing.assert_array_equal(actual.phase, expected.phase)
        later = session.reconstruct(actual, compute_loss=True)
        assert later.loss == expected.loss
        np.testing.assert_array_equal(later.phase, expected.phase)
