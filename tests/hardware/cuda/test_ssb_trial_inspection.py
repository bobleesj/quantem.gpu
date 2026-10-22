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
    settings = dict(tilt=False, num_trials=8, refinement=None, check_rotation=False, verbose=False,
                    search_ranges={'C10_nm': (-10., 10.), 'C12_nm': 4., 'phi12_deg': 30.},
                    save_dir=tmp_path / 'search')
    try:
        aberrations = ssb.find_aberrations(**settings)
        assert not hasattr(ssb, 'fit')
        assert len(aberrations.trials) == 8
        assert aberrations["C12_nm"] == pytest.approx(4.)
        assert math.radians(aberrations["phi12_deg"]) == pytest.approx(math.radians(30.))
        native = ssb.reconstruct(aberrations)
        expected, _ = ssb.preview(dict(aberrations))
        np.testing.assert_allclose(cp.asnumpy(native.phase), expected, atol=1e-7)
        records = deepcopy(aberrations.trial_records)
        replay = ssb.reconstruct(aberrations, trial_ids=[7, 0])
        assert replay.phase.shape == (2, 128, 128)
        for frame, trial_id in enumerate((7, 0)):
            record = records[trial_id]
            values = record['params']
            assert values['C12_nm'] == pytest.approx(4.)
            assert values['phi12_deg'] == pytest.approx(30.)
            phase, _ = ssb.preview({'C10_nm': values['C10_nm'], 'C12_nm': values['C12_nm'],
                                    'phi12_deg': values['phi12_deg']})
            np.testing.assert_allclose(cp.asnumpy(replay.phase[frame]), phase, atol=1e-7)
            # a standard trial's loss is its negative model agreement at sample tilt and depth spread 0
            point = np.array([[10.0 * values['C10_nm'], 10.0 * values['C12_nm'], math.radians(values['phi12_deg']),
                               0.0, 0.0, 0.0]])
            agreement = float(ssb._backend().sample_agreement(point, tuple(record['band_inv_A']))[0])
            assert record['objective'] == 'negative_thick_agreement'
            assert -agreement == pytest.approx(record['loss'], rel=1e-5)
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
        row = ssb.show_trials(best=3, rows=1, size="small")
        row_axes = [axis for axis in row.axes if axis.images and axis.get_visible()]
        assert len(row_axes) == 3
        assert len({round(axis.get_position().y0, 6) for axis in row_axes}) == 1
        for old, new in zip(image_axes, row_axes, strict=True):
            np.testing.assert_array_equal(old.images[0].get_array(), new.images[0].get_array())
        assert ssb._reconstruction is native
        loaded = load_result(paths=result_paths(settings['save_dir'], 'find_aberrations'),
                             signature=aberrations.metadata['signature'], backend='cuda')
        assert loaded.reused
        assert loaded.trial_records == records
        override = ssb.reconstruct(loaded, aberrations={'C10_nm': 0.}, upsampling_factor=2)
        assert override.phase.shape == (256, 256)
        assert override.object_sampling_A == (.15, .15)
        assert override['C10_nm'] == 0.
        assert override['C12_nm'] == loaded['C12_nm']
        assert dict(loaded) == dict(aberrations)
    finally:
        ssb.close()


def test_branch_comparison_keeps_original_trial_geometry(monkeypatch):
    import cupy as cp
    from quantem.gpu.ssb import workflow

    ssb, _ = _synthetic_session()
    compare = workflow.compare_branches

    def second_branch(fits, **kwargs):
        branches = compare(fits, **kwargs)
        branches.chosen = branches.kept = 1
        return branches

    monkeypatch.setattr(workflow, 'compare_branches', second_branch)
    first = ssb.scan_detector_rotation_deg
    try:
        ssb.compare_rotation_branches(num_trials=8, refinement=None, verbose=False, apply=True)
        aberrations = ssb._aberration_search
        assert aberrations.scan_detector_rotation_deg == ssb.scan_detector_rotation_deg != first
        first_search = [record for record in aberrations.trial_records if record['search'] == 0]
        assert first_search and all(record['scan_detector_rotation_deg'] == first for record in first_search)
        original = ssb.scan_detector_rotation_deg
        replay = ssb.reconstruct(aberrations, trial_ids=[0])
        assert ssb.scan_detector_rotation_deg == original
        assert ssb._reconstruction is aberrations
        record = aberrations.trial_records[0]
        values = record['params']
        ssb.set_rotation(record['scan_detector_rotation_deg'])
        # the branch comparison fits the joint model by default: the trial carries its tilt and depth spread
        expected, _ = ssb.preview({'C10_nm': values['C10_nm'], 'C12_nm': values['C12_nm'],
                                  'phi12_deg': values['phi12_deg']},
                                 sample_tilt_mrad=tuple(record['sample_tilt_mrad']),
                                 depth_spread_nm=record['depth_spread_nm'])
        np.testing.assert_allclose(cp.asnumpy(replay.phase[0]), expected, atol=1e-7)
    finally:
        ssb.close()


def test_joint_history_carries_tilt_depth_and_objective():
    import cupy as cp

    ssb, _ = _synthetic_session()
    try:
        aberrations = ssb.find_aberrations(tilt=True, num_trials=8, refinement=None,
                                           check_rotation=False, verbose=False)
        table = aberrations.trials
        assert len(table) == 8
        assert set(table.objective) == {'negative_thick_agreement'}
        assert {'sample_tilt_mrad', 'depth_spread_nm', 'band_inv_A'} <= set(table)
        record = aberrations.trial_records[3]
        values = record['params']
        expected, _ = ssb.preview({'C10_nm': values['C10_nm'], 'C12_nm': values['C12_nm'],
                                  'phi12_deg': values['phi12_deg']},
                                 sample_tilt_mrad=tuple(record['sample_tilt_mrad']),
                                 depth_spread_nm=record['depth_spread_nm'])
        replay = ssb.reconstruct(aberrations, trial_ids=[3])
        np.testing.assert_allclose(cp.asnumpy(replay.phase[0]), expected, atol=1e-7)
        for invalid in ({}, {'best': 0}, {'first': -1}, {'best': 2, 'last': 2}):
            with pytest.raises(ValueError, match='one positive count'):
                ssb.show_trials(**invalid)
        with pytest.raises(ValueError, match='Unknown trial IDs'):
            ssb.reconstruct(aberrations, trial_ids=[1000])
    finally:
        ssb.close()


def test_explicit_complex_estimator_preserves_wave_amplitude():
    import cupy as cp
    from quantem.gpu.ssb.units import aberrations_to_engine

    ssb, _ = _synthetic_session()
    coefs = {'C10_nm': -8., 'C12_nm': 4., 'phi12_deg': math.degrees(-.7)}
    try:
        expected = ssb._backend().reconstruct_result(aberrations_to_engine(coefs))
        result = ssb.reconstruct(aberrations=coefs, phase_estimator='complex_wave')
        np.testing.assert_array_equal(cp.asnumpy(result.object_wave), cp.asnumpy(expected.object_wave))
        assert result.amplitude_estimated
        assert result.aberrations == coefs
        with pytest.raises(ValueError, match='requires upsampling_factor=1'):
            ssb.reconstruct(aberrations=coefs, phase_estimator='complex_wave', upsampling_factor=2)
    finally:
        ssb.close()


def test_zero_trial_search_does_not_relabel_previous_search():
    ssb, _ = _synthetic_session()
    try:
        first = ssb.find_aberrations(tilt=False, num_trials=4, refinement='nelder-mead',
                                      check_rotation=False, verbose=False)
        history = deepcopy(first.trial_records)
        assert first.refine_nfev > 0
        ssb.set_rotation(37.)
        second = ssb.find_aberrations(tilt=False, num_trials=0, refinement=None,
                                       check_rotation=False, verbose=False)
        assert second.num_trials == 0
        assert second.trials.empty
        assert second.refine_nfev is None
        assert second.refinement is None
        assert first.trial_records == history
        assert all(record['scan_detector_rotation_deg'] == 0. for record in history)
        with pytest.raises(ValueError, match='No completed trials'):
            ssb.show_trials(best=5)
    finally:
        ssb.close()


@pytest.mark.parametrize('estimator', ['mean_phase', 'phase_of_mean', 'complex_wave'])
def test_search_starts_from_the_current_public_coefficients(estimator):
    """Changing coefficients before a refinement-only search must take effect."""
    ssb, _ = _synthetic_session()
    try:
        wanted = {'C10_nm': 9.0, 'C12_nm': 3.0, 'phi12_deg': math.degrees(0.25)}
        ssb.reconstruct(aberrations=wanted, phase_estimator=estimator, compute_loss=False)
        result = ssb.find_aberrations(tilt=False, num_trials=0, refinement=None,
                                      check_rotation=False, verbose=False)
        assert result.aberrations == wanted
        assert ssb.aberrations == wanted
    finally:
        ssb.close()


def test_zero_trial_tilt_search_fails_before_backend_work(monkeypatch):
    """A tilt search with no trials and no previous fit to refine fails before spending 200 trials on its control fit."""
    ssb, _ = _synthetic_session()
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError('An invalid tilt search reached the backend')
        monkeypatch.setattr(ssb, '_fit_tilt', forbidden)
        with pytest.raises(ValueError, match='num_trials=0'):
            ssb.find_aberrations(tilt=True, num_trials=0, verbose=False)
    finally:
        ssb.close()


def test_search_retains_anisotropic_scan_calibration():
    """Search and reconstruction results retain both physical pixel spacings."""
    from quantem.gpu import SSB

    original, _ = _synthetic_session()
    try:
        with SSB(original._data, backend='cuda', voltage_kV=300.0,
                 semiangle_mrad=30.0, scan_sampling_A=(0.3, 0.45),
                 detector_sampling_mrad=(3.0, 3.0)) as ssb:
            result = ssb.find_aberrations(tilt=False, num_trials=0, refinement=None,
                                          check_rotation=False, verbose=False)
            assert result.object_sampling_A == (0.3, 0.45)
            assert ssb.reconstruct(result, upsampling_factor=2).object_sampling_A == (0.15, 0.225)
    finally:
        original.close()
