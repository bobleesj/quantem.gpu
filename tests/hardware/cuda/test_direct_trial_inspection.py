"""Replay scientific search history without mutating the active reconstruction."""

from copy import deepcopy
import math

import numpy as np
import pytest

from tests.hardware.cuda.test_direct_thick_sample import _synthetic_session

pytestmark = pytest.mark.slow


def test_fixed_parameters_survive_trial_replay_and_saved_reload(tmp_path):
    import cupy as cp
    from quantem.gpu.direct_ptycho.persistence import load_result, result_paths

    direct, _ = _synthetic_session()
    source = tmp_path / 'counts.npy'
    np.save(source, cp.asnumpy(direct._data))
    direct.source_path = str(source)
    settings = dict(tilt=False, trials=8, refinement=None, check_rotation=False, verbose=False,
                    search_ranges={'C10_nm': (-10., 10.), 'C12_nm': 4., 'phi12_deg': 30.},
                    save_dir=tmp_path / 'search')
    try:
        aberrations = direct.aberrations(search="primary", **settings)
        assert not hasattr(direct, 'fit')
        assert len(aberrations.trials) == 8
        assert aberrations["C12_nm"] == pytest.approx(4.)
        assert math.radians(aberrations["phi12_deg"]) == pytest.approx(math.radians(30.))
        native = direct.run(aberrations)
        expected, _ = direct.preview(dict(aberrations))
        np.testing.assert_allclose(cp.asnumpy(native.phase), expected, atol=1e-7)
        records = deepcopy(aberrations.trial_records)
        replay = direct.run(aberrations, trial_ids=[7, 0])
        assert replay.phase.shape == (2, 128, 128)
        for frame, trial_id in enumerate((7, 0)):
            record = records[trial_id]
            values = record['params']
            assert values['C12_nm'] == pytest.approx(4.)
            assert values['phi12_deg'] == pytest.approx(30.)
            phase, _ = direct.preview({'C10_nm': values['C10_nm'], 'C12_nm': values['C12_nm'],
                                    'phi12_deg': values['phi12_deg']})
            np.testing.assert_allclose(cp.asnumpy(replay.phase[frame]), phase, atol=1e-7)
            # a standard trial's loss is its negative model agreement at sample tilt and depth spread 0
            point = np.array([[10.0 * values['C10_nm'], 10.0 * values['C12_nm'], math.radians(values['phi12_deg']),
                               0.0, 0.0, 0.0]])
            agreement = float(direct._backend().sample_agreement(point, tuple(record['band_inv_A']))[0])
            assert record['objective'] == 'negative_thick_agreement'
            assert -agreement == pytest.approx(record['loss'], rel=1e-5)
        assert direct._reconstruction is native
        assert aberrations.trial_records == records
        figure = direct.show("trials", best=3)
        image_axes = [axis for axis in figure.axes if axis.images and axis.get_visible()]
        best_records = sorted(records, key=lambda record: record['loss'])[:3]
        assert [axis.get_title() for axis in image_axes] == [
            f"Trial {record['trial']}" for record in best_records
        ]
        assert image_axes[0].get_position().x0 == pytest.approx(image_axes[2].get_position().x0)
        assert image_axes[0].get_position().y0 > image_axes[2].get_position().y0
        row = direct.show("trials", best=3, rows=1, size="small")
        row_axes = [axis for axis in row.axes if axis.images and axis.get_visible()]
        assert len(row_axes) == 3
        assert len({round(axis.get_position().y0, 6) for axis in row_axes}) == 1
        for old, new in zip(image_axes, row_axes, strict=True):
            np.testing.assert_array_equal(old.images[0].get_array(), new.images[0].get_array())
        assert direct._reconstruction is native
        loaded = load_result(paths=result_paths(settings['save_dir'], 'aberrations'),
                             signature=aberrations.metadata['signature'], backend='cuda')
        assert loaded.reused
        assert loaded.trial_records == records
        override = direct.run(loaded, aberrations={'C10_nm': 0.}, upsample=2)
        assert override.phase.shape == (256, 256)
        assert override.object_sampling_A == (.15, .15)
        assert override['C10_nm'] == 0.
        assert override['C12_nm'] == loaded['C12_nm']
        assert dict(loaded) == dict(aberrations)
    finally:
        direct.close()


def test_branch_comparison_keeps_original_trial_geometry(monkeypatch):
    import cupy as cp
    from quantem.gpu.direct_ptycho import workflow

    direct, _ = _synthetic_session()
    compare = workflow.compare_branches

    def second_branch(fits, **kwargs):
        branches = compare(fits, **kwargs)
        branches.chosen = branches.kept = 1
        return branches

    monkeypatch.setattr(workflow, 'compare_branches', second_branch)
    first = direct.scan_detector_rotation_deg
    try:
        direct.scan_rotations(report=False, trials=8, refinement=None, verbose=False, apply=True)
        aberrations = direct._aberration_search
        assert aberrations.scan_detector_rotation_deg == direct.scan_detector_rotation_deg != first
        first_search = [record for record in aberrations.trial_records if record['search'] == 0]
        assert first_search and all(record['scan_detector_rotation_deg'] == first for record in first_search)
        original = direct.scan_detector_rotation_deg
        replay = direct.run(aberrations, trial_ids=[0])
        assert direct.scan_detector_rotation_deg == original
        assert direct._reconstruction is aberrations
        record = aberrations.trial_records[0]
        values = record['params']
        direct.set_rotation(record['scan_detector_rotation_deg'])
        # the branch comparison fits the joint model by default: the trial carries its tilt and depth spread
        expected, _ = direct.preview({'C10_nm': values['C10_nm'], 'C12_nm': values['C12_nm'],
                                  'phi12_deg': values['phi12_deg']},
                                 sample_tilt_mrad=tuple(record['sample_tilt_mrad']),
                                 depth_spread_nm=record['depth_spread_nm'])
        np.testing.assert_allclose(cp.asnumpy(replay.phase[0]), expected, atol=1e-7)
    finally:
        direct.close()


def test_joint_history_carries_tilt_depth_and_objective():
    import cupy as cp

    direct, _ = _synthetic_session()
    try:
        aberrations = direct.aberrations(search="primary", tilt=True, trials=8, refinement=None,
                                           check_rotation=False, verbose=False)
        table = aberrations.trials
        assert len(table) == 8
        assert set(table.objective) == {'negative_thick_agreement'}
        assert {'sample_tilt_mrad', 'depth_spread_nm', 'band_inv_A'} <= set(table)
        record = aberrations.trial_records[3]
        values = record['params']
        expected, _ = direct.preview({'C10_nm': values['C10_nm'], 'C12_nm': values['C12_nm'],
                                  'phi12_deg': values['phi12_deg']},
                                 sample_tilt_mrad=tuple(record['sample_tilt_mrad']),
                                 depth_spread_nm=record['depth_spread_nm'])
        replay = direct.run(aberrations, trial_ids=[3])
        np.testing.assert_allclose(cp.asnumpy(replay.phase[0]), expected, atol=1e-7)
        for invalid in ({}, {'best': 0}, {'first': -1}, {'best': 2, 'last': 2}):
            with pytest.raises(ValueError, match='one positive count'):
                direct.show("trials", **invalid)
        with pytest.raises(ValueError, match='Unknown trial IDs'):
            direct.run(aberrations, trial_ids=[1000])
    finally:
        direct.close()


def test_zero_trial_search_does_not_relabel_previous_search():
    direct, _ = _synthetic_session()
    try:
        first = direct.aberrations(search="primary", tilt=False, trials=4, check_rotation=False,
                                     verbose=False)
        history = deepcopy(first.trial_records)
        assert first.refine_nfev > 0
        direct.set_rotation(37.)
        second = direct.aberrations(search="primary", tilt=False, trials=0, refinement=None,
                                       check_rotation=False, verbose=False)
        assert second.trials.empty
        assert second.refine_nfev is None
        assert second.refinement is None
        assert first.trial_records == history
        assert all(record['scan_detector_rotation_deg'] == 0. for record in history)
        with pytest.raises(ValueError, match='No completed trials'):
            direct.show("trials", best=5)
    finally:
        direct.close()


def test_search_starts_from_the_current_public_coefficients():
    """Changing coefficients before a refinement-only search must take effect."""
    direct, _ = _synthetic_session()
    try:
        wanted = {'C10_nm': 9.0, 'C12_nm': 3.0, 'phi12_deg': math.degrees(0.25)}
        direct.run(aberrations=wanted, compute_loss=False)
        result = direct.aberrations(search="primary", tilt=False, trials=0, refinement=None,
                                      check_rotation=False, verbose=False)
        assert result.aberrations == wanted
        assert direct.known_aberrations == wanted
    finally:
        direct.close()


def test_zero_trial_tilt_search_fails_before_backend_work(monkeypatch):
    """A tilt search with no trials and no previous fit to refine fails before spending 200 trials on its control fit."""
    direct, _ = _synthetic_session()
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError('An invalid tilt search reached the backend')
        monkeypatch.setattr(direct, '_fit_tilt', forbidden)
        with pytest.raises(ValueError, match='trials=0'):
            direct.aberrations(search="primary", tilt=True, trials=0, verbose=False)
    finally:
        direct.close()


def test_search_retains_anisotropic_scan_calibration():
    """Search and reconstruction results retain both physical pixel spacings."""
    from quantem.gpu import DirectPtycho

    original, _ = _synthetic_session()
    try:
        with DirectPtycho(original._data, backend='cuda', voltage_kV=300.0,
                 semiangle_mrad=30.0, scan_sampling_A=(0.3, 0.45),
                 detector_sampling_mrad=(3.0, 3.0)) as direct:
            result = direct.aberrations(search="primary", tilt=False, trials=0, refinement=None,
                                          check_rotation=False, verbose=False)
            assert result.object_sampling_A == (0.3, 0.45)
            assert direct.run(result, upsample=2).object_sampling_A == (0.15, 0.225)
    finally:
        original.close()
