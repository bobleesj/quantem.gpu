"""SSB computation stays on CUDA until an explicit viewer export."""

import numpy as np
import pytest

from tests.hardware.cuda.test_ssb_thick_sample import _synthetic_session


@pytest.mark.parametrize("estimator", ["mean_phase", "phase_of_mean", "complex_wave"])
def test_reconstruction_does_not_export_phase(monkeypatch, estimator):
    import cupy as cp

    ssb, _ = _synthetic_session()
    values = {"C10": -8.0, "C12": 4.0, "phi12": -0.7}
    try:
        expected = ssb.reconstruct(aberrations=values, phase_estimator=estimator)
        # Preparation may export tiny geometry metadata; repeated scientific
        # reconstruction must not export the image and upload it again.
        original_asnumpy = cp.asnumpy

        def no_host(value, *args, **kwargs):
            if value.ndim > 1:
                raise AssertionError("Reconstruction exported a phase image")
            return original_asnumpy(value, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(cp, "asnumpy", no_host)
            actual = ssb.reconstruct(aberrations=values, phase_estimator=estimator, force=True)
        assert isinstance(actual.object_wave, cp.ndarray)
        cp.testing.assert_array_equal(actual.object_wave, expected.object_wave)
        phase, _ = ssb.preview(values, phase_estimator="mean_phase")
        assert isinstance(phase, np.ndarray)
    finally:
        ssb.close()


def test_trial_replay_and_tilt_stay_on_device(monkeypatch):
    import cupy as cp

    ssb, _ = _synthetic_session()
    try:
        fitted = ssb.find_aberrations(trials=4, refinement=None, check_rotation=False, verbose=False)
        ssb.tilt_mrad = (2.0, -1.0)
        ssb.depth_spread_nm = 5.0
        ssb.reconstruct(upsample=2, force=True)

        def no_host(*args, **kwargs):
            raise AssertionError("Reconstruction exported a CUDA array")

        with monkeypatch.context() as patch:
            patch.setattr(cp, "asnumpy", no_host)
            result = ssb.reconstruct(upsample=2, force=True)
            replay = ssb.reconstruct(fitted, trials=[0, 2])
        assert isinstance(result.object_wave, cp.ndarray)
        assert isinstance(replay.object_wave, cp.ndarray)
        assert replay.object_wave.shape == (2, 128, 128)
    finally:
        ssb.close()


def test_mean_phase_trial_replay_owns_each_trial_wave():
    """A later mean-phase kernel must not overwrite an earlier replayed image."""
    import cupy as cp

    ssb, _ = _synthetic_session()
    try:
        fitted = ssb.find_aberrations(trials=4, refinement=None, check_rotation=False, verbose=False)
        expected = []
        for record in fitted.trial_records[:3]:
            values = record["params"]
            result = ssb.reconstruct(aberrations={
                "C10": values["C10_nm"], "C12": values["C12_nm"],
                "phi12": np.deg2rad(values["phi12_deg"]),
            }, phase_estimator="mean_phase", compute_loss=False)
            expected.append(result.object_wave.copy())
        assert any(not cp.array_equal(expected[0], frame) for frame in expected[1:])
        replay = ssb.reconstruct(fitted, trials=[0, 1, 2], phase_estimator="mean_phase")
        cp.testing.assert_array_equal(replay.object_wave, cp.stack(expected))
    finally:
        ssb.close()
