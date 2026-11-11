"""Scientist workflow: carry correction into sampling changes and saved reloads."""
import math
from dataclasses import replace

import numpy as np
import pytest

from .test_direct_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def test_auto_grid_matches_explicit_grid_without_refitting(monkeypatch):
    import cupy as cp
    from quantem.gpu.optics.physics import direct_auto_upsample

    direct, _ = _synthetic_session()
    try:
        direct.sample_tilt_mrad = (2.0, -3.0)
        direct.depth_spread_nm = 12.0
        factor = direct_auto_upsample(
            voltage_kV=direct.voltage_kV, semiangle_mrad=direct.semiangle_mrad,
            scan_sampling_A=direct.scan_sampling_A,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("Output sampling must not search aberrations")

        monkeypatch.setattr(direct, "aberrations", forbidden)
        native = direct.run(upsample=1)
        explicit = direct.run(upsample=factor)
        automatic = direct.run(upsample="auto")
        assert automatic.upsample == factor
        assert automatic.loss == native.loss
        assert automatic.aberrations == native.aberrations
        assert automatic.sample_tilt_mrad == native.sample_tilt_mrad
        assert automatic.depth_spread_nm == native.depth_spread_nm
        np.testing.assert_array_equal(cp.asnumpy(automatic.phase), cp.asnumpy(explicit.phase))
    finally:
        direct.close()


@pytest.mark.parametrize('depth', [0., 12.])
def test_saved_sampling_preserves_correction_and_native_loss(tmp_path, depth, monkeypatch):
    import cupy as cp

    direct, _ = _synthetic_session()
    source = tmp_path / 'counts.npy'
    np.save(source, cp.asnumpy(direct._data))
    direct.source_path = str(source)
    try:
        correction = direct.aberrations(search="primary", tilt=False, trials=0, refinement=None,
                                          check_rotation=False, verbose=False)
        correction = replace(correction, sample_tilt_mrad=(2., -3.), depth_spread_nm=depth)
        native, loss = direct.preview(correction.aberrations, sample_tilt_mrad=correction.sample_tilt_mrad,
                                  depth_spread_nm=correction.depth_spread_nm)
        results = {}
        for factor in (1, 2, 3, 4, 1):
            result = direct.run(correction, upsample=factor, save_dir=tmp_path / f'x{factor}')
            assert result.phase.shape == (128 * factor, 128 * factor)
            assert result.object_sampling_A == (.3 / factor, .3 / factor)
            assert result.loss == loss
            assert result.aberrations == correction.aberrations
            assert result.sample_tilt_mrad == correction.sample_tilt_mrad
            assert result.depth_spread_nm == depth
            assert not result.amplitude_estimated
            assert result.upsample == factor
            assert result.reconstruction_method == 'phase_of_mean'
            assert np.isfinite(cp.asnumpy(result.phase)).all()
            results[factor] = cp.asnumpy(result.phase)
        np.testing.assert_allclose(results[1], native, atol=1e-7)
        def forbidden(*args, **kwargs):
            raise AssertionError('Reconstruction must not refit')
        monkeypatch.setattr(direct, 'aberrations', forbidden)
        for factor in (4, 3, 2, 1):
            saved = direct.run(correction, upsample=factor, save_dir=tmp_path / f'x{factor}')
            # A source path cannot identify the direct array's crop or edits.
            # The saved image is refreshed; immutable DirectPtycho.open sources have
            # separate cache-reuse checks in test_direct_persistence.
            assert not saved.reused
            np.testing.assert_array_equal(cp.asnumpy(saved.phase), results[factor])
    finally:
        direct.close()


def test_default_correction_and_changed_sample_do_not_reuse_stale_phase():
    import cupy as cp

    direct, _ = _synthetic_session()
    try:
        correction = direct.run(aberrations={'C10_nm': -8., 'C12_nm': 4., 'phi12_deg': math.degrees(-.7)})
        assert correction.reconstruction_method == 'phase_of_mean'
        expected, _ = direct.preview(correction.aberrations)
        native = direct.run(correction)
        assert native.reconstruction_method == 'phase_of_mean'
        np.testing.assert_allclose(cp.asnumpy(native.phase), expected, atol=1e-7)
        direct.depth_spread_nm = 12.
        direct.sample_tilt_mrad = (2., -3.)
        changed = direct.run()
        assert changed is not native
        expected, _ = direct.preview(correction.aberrations, sample_tilt_mrad=(2., -3.), depth_spread_nm=12.)
        np.testing.assert_allclose(cp.asnumpy(changed.phase), expected, atol=1e-7)
    finally:
        direct.close()
