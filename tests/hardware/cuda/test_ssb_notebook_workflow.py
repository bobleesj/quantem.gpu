"""Scientist workflow: carry correction into sampling changes and saved reloads."""
from dataclasses import replace

import numpy as np
import pytest

from .test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def test_auto_grid_matches_explicit_grid_without_refitting(monkeypatch):
    import cupy as cp
    from quantem.gpu.optics.physics import ssb_upsampling_factor

    ssb, _ = _synthetic_session()
    try:
        ssb.tilt_mrad = (2.0, -3.0)
        ssb.depth_spread_nm = 12.0
        factor = ssb_upsampling_factor(
            voltage_kV=ssb.voltage_kV, semiangle_mrad=ssb.semiangle_mrad,
            scan_sampling_A=ssb.scan_sampling_A,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Output sampling must not search aberrations")

        monkeypatch.setattr(ssb, "find_aberrations", forbidden)
        native = ssb.reconstruct(upsample=1)
        explicit = ssb.reconstruct(upsample=factor)
        automatic = ssb.reconstruct(upsample="auto")
        assert automatic.upsample == factor
        assert automatic.loss == native.loss
        assert automatic.aberrations == native.aberrations
        assert automatic.tilt_mrad == native.tilt_mrad
        assert automatic.depth_spread_nm == native.depth_spread_nm
        np.testing.assert_array_equal(cp.asnumpy(automatic.phase), cp.asnumpy(explicit.phase))
    finally:
        ssb.close()


@pytest.mark.parametrize('depth', [0., 12.])
@pytest.mark.parametrize('estimator', ['mean_phase', 'phase_of_mean'])
def test_saved_sampling_preserves_correction_and_native_loss(tmp_path, estimator, depth, monkeypatch):
    import cupy as cp

    ssb, _ = _synthetic_session()
    source = tmp_path / 'counts.npy'
    np.save(source, cp.asnumpy(ssb._data))
    ssb.source_path = str(source)
    try:
        correction = ssb.find_aberrations(trials=0, refinement=None,
                                          check_rotation=False, verbose=False)
        correction = replace(correction, tilt_mrad=(2., -3.), depth_spread_nm=depth)
        native, loss = ssb.preview(correction.aberrations, tilt_mrad=correction.tilt_mrad,
                                  depth_spread_nm=correction.depth_spread_nm,
                                  phase_estimator=estimator)
        results = {}
        for factor in (1, 2, 3, 4, 1):
            result = ssb.reconstruct(correction, upsample=factor, phase_estimator=estimator,
                                     save_to=tmp_path / f'x{factor}')
            assert result.phase.shape == (128 * factor, 128 * factor)
            assert result.scan_sampling_A == .3 / factor
            assert result.loss == loss
            assert result.aberrations == correction.aberrations
            assert result.tilt_mrad == correction.tilt_mrad
            assert result.depth_spread_nm == depth
            assert not result.amplitude_estimated
            assert result.upsample == factor
            assert result.phase_estimator == estimator
            assert np.isfinite(cp.asnumpy(result.phase)).all()
            results[factor] = cp.asnumpy(result.phase)
        np.testing.assert_allclose(results[1], native, atol=1e-7)
        def forbidden(*args, **kwargs):
            raise AssertionError('Reconstruction must not refit')
        monkeypatch.setattr(ssb, 'find_aberrations', forbidden)
        for factor in (4, 3, 2, 1):
            saved = ssb.reconstruct(correction, upsample=factor, phase_estimator=estimator,
                                    save_to=tmp_path / f'x{factor}')
            # A source path cannot identify the direct array's crop or edits.
            # The saved image is refreshed; immutable SSB.open sources have
            # separate cache-reuse checks in test_ssb_persistence.
            assert not saved.reused
            np.testing.assert_array_equal(cp.asnumpy(saved.phase), results[factor])
    finally:
        ssb.close()


def test_default_correction_and_changed_sample_do_not_reuse_stale_phase():
    import cupy as cp

    ssb, _ = _synthetic_session()
    try:
        correction = ssb.reconstruct(aberrations={'C10': -8., 'C12': 4., 'phi12': -.7})
        assert correction.phase_estimator == 'phase_of_mean'
        expected, _ = ssb.preview(correction.aberrations)
        native = ssb.reconstruct(correction)
        assert native.phase_estimator == 'phase_of_mean'
        np.testing.assert_allclose(cp.asnumpy(native.phase), expected, atol=1e-7)
        ssb.depth_spread_nm = 12.
        ssb.tilt_mrad = (2., -3.)
        changed = ssb.reconstruct(phase_estimator='mean_phase')
        assert changed is not native
        expected, _ = ssb.preview(correction.aberrations, tilt_mrad=(2., -3.), depth_spread_nm=12., phase_estimator='mean_phase')
        np.testing.assert_allclose(cp.asnumpy(changed.phase), expected, atol=1e-7)
    finally:
        ssb.close()
