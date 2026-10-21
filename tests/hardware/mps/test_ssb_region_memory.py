"""Regional correction preserves the overview under both memory policies."""

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps
from .test_ssb_upsampling import _disk_session


@pytest.mark.parametrize("retain", [False, True])
def test_region_fit_preserves_overview_with_memory_budget(monkeypatch, retain):
    """Fit a region and return to the identical full-field correction."""
    _require_mps()
    from quantem.gpu.ssb.mps.hardware import require_mlx

    mx = require_mlx()
    info = mx.device_info()
    monkeypatch.setattr(mx, "device_info", lambda: {
        **info, "max_recommended_working_set_size": (128 << 30) if retain else 0,
    })
    optics = {"C10_nm": 8.0, "C12_nm": 1.0, "phi12_deg": 30.0}
    with _disk_session(n=128) as session:
        before, before_loss = session.preview(
            optics, sample_tilt_mrad=(2.0, -1.0), depth_spread_nm=12.0,
            upsampling_factor=2, compute_loss=True,
        )
        prepared = session._compute._prepared
        with session.select_region((16, 80, 24, 104)) as regional:
            assert (session._compute._prepared is prepared) == retain
            fit = regional.find_aberrations(
                tilt=True, num_trials=8, refinement="newton", seed=11,
                check_rotation=False, compare_standard=False, verbose=False,
            )
            assert np.isfinite(fit.phase).all()
        after, after_loss = session.preview(
            optics, sample_tilt_mrad=(2.0, -1.0), depth_spread_nm=12.0,
            upsampling_factor=2, compute_loss=True,
        )
        np.testing.assert_array_equal(after, before)
        assert after_loss == before_loss


def test_full_field_region_shares_evidence_but_keeps_fit_state_independent():
    """A whole-scan region needs one Fourier stack, with unchanged fit results."""
    _require_mps()
    optics = {"C10_nm": 8.0, "C12_nm": 1.0, "phi12_deg": 30.0}
    with _disk_session(n=64) as session:
        before, before_loss = session.preview(optics, compute_loss=True)
        prepared = session._compute._prepared
        with session.select_region((0, 64, 0, 64)) as regional:
            other = regional._compute._prepared
            assert other is not prepared
            assert other.g_qk is prepared.g_qk
            options = dict(tilt=True, num_trials=8, refinement="newton", seed=11,
                           check_rotation=False, compare_standard=False, verbose=False)
            shared = regional.find_aberrations(**options)
            regional._compute.release_prepared()
            rebuilt = regional.find_aberrations(**options)
            assert shared.aberrations == rebuilt.aberrations
            assert shared.sample_tilt_mrad == rebuilt.sample_tilt_mrad
            assert shared.depth_spread_nm == rebuilt.depth_spread_nm
            np.testing.assert_array_equal(shared.phase, rebuilt.phase)
            regional.set_rotation(20.0)
            regional.preview(optics, compute_loss=False)
        after, after_loss = session.preview(optics, compute_loss=True)
        assert session._compute._prepared is prepared
        np.testing.assert_array_equal(after, before)
        assert after_loss == before_loss


def test_rectangular_region_reserves_its_actual_padded_grid(monkeypatch):
    """A 129 by 128 selection must budget the 256-square FFT it will build."""
    _require_mps()
    from quantem.gpu.ssb.mps.hardware import require_mlx

    mx = require_mlx()
    with _disk_session(n=256) as session:
        session.preview({"C10_nm": 8.0, "C12_nm": 1.0, "phi12_deg": 30.0},
                        compute_loss=False)
        backend = session._compute
        # Enough for the measured rectangle, but not four padded FFT stacks.
        limit = mx.get_active_memory() + 2 * backend._prepared.g_qk.nbytes + (2 << 30)
        info = mx.device_info()
        monkeypatch.setattr(mx, "device_info", lambda: {
            **info, "max_recommended_working_set_size": 2 * limit,
        })
        backend.release_for_region((129, 128))
        assert backend._prepared is None
