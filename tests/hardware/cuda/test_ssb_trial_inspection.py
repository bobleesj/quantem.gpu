"""Replay scientific search history without mutating the active reconstruction."""

from copy import deepcopy
import math

import numpy as np
import pytest

from tests.hardware.cuda.test_ssb_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def test_fixed_parameters_survive_trial_replay_and_saved_reload(tmp_path):
    import cupy as cp
    from quantem.gpu.ssb.persistence import load_result, result_paths

    ssb, _ = _synthetic_session()
    source = tmp_path / 'counts.npy'
    np.save(source, cp.asnumpy(ssb._data))
    ssb.source_path = str(source)
    settings = dict(trials=8, refinement=None, check_rotation=False, verbose=False,
                    search_ranges={'C10_nm': (-10., 10.), 'C12_nm': 4., 'phi12_deg': 30.},
                    save_to=tmp_path / 'search')
    try:
        aberrations = ssb.find_aberrations(**settings)
        assert not hasattr(ssb, 'fit')
        assert len(aberrations.trials) == 8
        assert aberrations["C12"] == pytest.approx(4.)
        assert aberrations["phi12"] == pytest.approx(math.radians(30.))
        native = ssb.reconstruct(aberrations)
        expected, _ = ssb.preview(dict(aberrations))
        np.testing.assert_allclose(cp.asnumpy(native.phase), expected, atol=1e-7)
        records = deepcopy(aberrations.trial_records)
        replay = ssb.reconstruct(aberrations, trials=[7, 0])
        assert replay.phase.shape == (2, 128, 128)
        for frame, trial_id in enumerate((7, 0)):
            record = records[trial_id]
            values = record['params']
            assert values['C12_nm'] == pytest.approx(4.)
            assert values['phi12_deg'] == pytest.approx(30.)
            phase, loss = ssb.preview({'C10': values['C10_nm'], 'C12': values['C12_nm'],
                                       'phi12': math.radians(values['phi12_deg'])})
            np.testing.assert_allclose(cp.asnumpy(replay.phase[frame]), phase, atol=1e-7)
            assert loss == pytest.approx(record['loss'], rel=2e-5, abs=1e-7)
        assert ssb._reconstruction is native
        assert aberrations.trial_records == records
        figure = ssb.show_trials(best=3)
        image_axes = [axis for axis in figure.axes if axis.images and axis.get_visible()]
        best_records = sorted(records, key=lambda record: record['loss'])[:3]
        assert [axis.get_title() for axis in image_axes] == [
            f"Trial {record['trial']}" for record in best_records
        ]
        assert image_axes[0].get_position().x0 == pytest.approx(image_axes[2].get_position().x0)
        assert image_axes[0].get_position().y0 > image_axes[2].get_position().y0
        assert ssb._reconstruction is native
        loaded = load_result(paths=result_paths(settings['save_to'], 'find_aberrations'),
                             signature=aberrations.metadata['signature'], backend='cuda')
        assert loaded.reused
        assert loaded.trial_records == records
        override = ssb.reconstruct(loaded, aberrations={'C10': 0.}, upsample=2)
        assert override.phase.shape == (256, 256)
        assert override.scan_sampling_A == .15
        assert override['C10'] == 0.
        assert override['C12'] == loaded['C12']
        assert dict(loaded) == dict(aberrations)
    finally:
        ssb.close()


def test_rotation_check_keeps_original_trial_geometry(monkeypatch):
    import cupy as cp
    from quantem.gpu.ssb import workflow

    ssb, _ = _synthetic_session()
    signs = iter((-1., 1.))
    monkeypatch.setattr(workflow, 'column_sign', lambda phase: next(signs))
    try:
        aberrations = ssb.find_aberrations(trials=8, refinement=None, verbose=False)
        assert aberrations.rotation_flipped
        assert ssb.com_reversed
        assert not any(record['com_reversed'] for record in aberrations.trial_records)
        original = (ssb.rotation_angle_deg, ssb.com_reversed)
        replay = ssb.reconstruct(aberrations, trials=[0])
        assert (ssb.rotation_angle_deg, ssb.com_reversed) == original
        assert ssb._reconstruction is aberrations
        record = aberrations.trial_records[0]
        values = record['params']
        ssb.set_rotation(record['rotation_angle_deg'], record['com_reversed'])
        expected, _ = ssb.preview({'C10': values['C10_nm'], 'C12': values['C12_nm'],
                                  'phi12': math.radians(values['phi12_deg'])})
        np.testing.assert_allclose(cp.asnumpy(replay.phase[0]), expected, atol=1e-7)
    finally:
        ssb.close()


def test_joint_history_carries_tilt_depth_and_objective():
    import cupy as cp

    ssb, _ = _synthetic_session()
    try:
        aberrations = ssb.find_aberrations(tilt=True, trials=8, refinement=None,
                                           check_rotation=False, verbose=False)
        table = aberrations.trials
        assert len(table) == 8
        assert set(table.objective) == {'negative_thick_agreement'}
        assert {'tilt_row_mrad', 'tilt_col_mrad', 'depth_spread_nm', 'band_inv_A'} <= set(table)
        record = aberrations.trial_records[3]
        values = record['params']
        expected, _ = ssb.preview({'C10': values['C10_nm'], 'C12': values['C12_nm'],
                                  'phi12': math.radians(values['phi12_deg'])},
                                 tilt_mrad=(record['tilt_row_mrad'], record['tilt_col_mrad']),
                                 depth_spread_nm=record['depth_spread_nm'])
        replay = ssb.reconstruct(aberrations, trials=[3])
        np.testing.assert_allclose(cp.asnumpy(replay.phase[0]), expected, atol=1e-7)
        for invalid in ({}, {'best': 0}, {'first': -1}, {'best': 2, 'last': 2}):
            with pytest.raises(ValueError, match='one positive count'):
                ssb.show_trials(**invalid)
        with pytest.raises(ValueError, match='Unknown trial IDs'):
            ssb.reconstruct(aberrations, trials=[1000])
    finally:
        ssb.close()


def test_explicit_complex_estimator_preserves_wave_amplitude():
    import cupy as cp
    from quantem.gpu.ssb.units import aberrations_to_engine

    ssb, _ = _synthetic_session()
    coefs = {'C10': -8., 'C12': 4., 'phi12': -.7}
    try:
        expected = ssb._backend_protocol.reconstruct_result(aberrations_to_engine(coefs))
        result = ssb.reconstruct(aberrations=coefs, phase_estimator='complex_wave')
        np.testing.assert_array_equal(cp.asnumpy(result.object_wave), cp.asnumpy(expected.object_wave))
        assert result.amplitude_estimated
        assert result.aberrations == coefs
        with pytest.raises(ValueError, match='requires upsample=1'):
            ssb.reconstruct(aberrations=coefs, phase_estimator='complex_wave', upsample=2)
    finally:
        ssb.close()


def test_zero_trial_search_does_not_relabel_previous_search():
    ssb, _ = _synthetic_session()
    try:
        first = ssb.find_aberrations(trials=4, refinement='nelder-mead',
                                      check_rotation=False, verbose=False)
        history = deepcopy(first.trial_records)
        assert first.refine_nfev > 0
        ssb.set_rotation(37.)
        second = ssb.find_aberrations(trials=0, refinement=None,
                                       check_rotation=False, verbose=False)
        assert second.n_trials == 0
        assert second.trials.empty
        assert second.refine_nfev is None
        assert second.refine_method is None
        assert first.trial_records == history
        assert all(record['rotation_angle_deg'] == 0. for record in history)
        with pytest.raises(ValueError, match='No completed trials'):
            ssb.show_trials(best=5)
    finally:
        ssb.close()


@pytest.mark.parametrize('estimator', ['mean_phase', 'phase_of_mean', 'complex_wave'])
def test_search_starts_from_the_current_public_coefficients(estimator):
    """Changing coefficients before a refinement-only search must take effect."""
    ssb, _ = _synthetic_session()
    try:
        wanted = {'C10': 9.0, 'C12': 3.0, 'phi12': 0.25}
        ssb.reconstruct(aberrations=wanted, phase_estimator=estimator, compute_loss=False)
        result = ssb.find_aberrations(trials=0, refinement=None,
                                      check_rotation=False, verbose=False)
        assert result.aberrations == wanted
        assert ssb.aberrations == wanted
    finally:
        ssb.close()


def test_zero_trial_tilt_search_fails_before_backend_work(monkeypatch):
    """An empty tilt search must not first spend 200 trials on its control fit."""
    ssb, _ = _synthetic_session()
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError('An invalid tilt search reached the backend')
        monkeypatch.setattr(ssb, '_fit_tilt', forbidden)
        with pytest.raises(ValueError, match='tilt=True requires at least one trial'):
            ssb.find_aberrations(tilt=True, trials=0, verbose=False)
    finally:
        ssb.close()


def test_search_retains_anisotropic_scan_calibration():
    """Search and reconstruction results retain both physical pixel spacings."""
    from quantem.gpu import SSB

    original, _ = _synthetic_session()
    try:
        with SSB(original._data, backend='cuda', voltage_kV=300.0,
                 semiangle_mrad=30.0, scan_sampling_A=(0.3, 0.45),
                 det_sampling=3.0) as ssb:
            result = ssb.find_aberrations(trials=0, refinement=None,
                                          check_rotation=False, verbose=False)
            assert result.scan_sampling_A == (0.3, 0.45)
            assert ssb.reconstruct(result, upsample=2).scan_sampling_A == (0.15, 0.225)
    finally:
        original.close()
