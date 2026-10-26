"""Reconstruct, review and restore direct methods on a completed acquisition.

Set QUANTEM_DIRECT_SOURCE to a public gold acquisition, optionally supplying
its calibration as QUANTEM_DIRECT_CALIBRATION (a JSON object). These tests
preserve every detector pixel and use a 48 by 48 measured scan region.
"""

import json
import os

import numpy as np
import pytest

from quantem.gpu import SSB, DirectPtycho

pytestmark = pytest.mark.slow


@pytest.fixture
def acquisition():
    source = os.environ.get("QUANTEM_DIRECT_SOURCE")
    if source is None:
        pytest.skip(
            "Set QUANTEM_DIRECT_SOURCE to the public gold acquisition for CUDA acceptance."
        )
    calibration = json.loads(os.environ.get("QUANTEM_DIRECT_CALIBRATION", "{}"))
    return source, calibration


def test_compare_methods_and_change_only_regularization(acquisition):
    """Keep the measured region and optics fixed while comparing direct methods."""
    import cupy as cp

    source, calibration = acquisition
    with DirectPtycho.open(source, backend="cuda", **calibration) as full, full.select_region((64, 112, 64, 112)) as ptycho:
        optics = ptycho.find_aberrations(
            tilt=True,
            trials=12,
            search="primary",
            check_rotation=False,
            show=False,
            verbose=False,
        )
        ssb = ptycho.run(optics, upsampling_factor="auto", compute_loss=False)
        with pytest.warns(DeprecationWarning):
            old = ptycho.reconstruct(
                optics, upsampling_factor="auto", compute_loss=False
            )
        assert SSB is DirectPtycho
        cp.testing.assert_array_equal(ssb.object_wave, old.object_wave)
        phase = ptycho.run(optics, method="phase", upsampling_factor="auto")
        original = phase.object_wave.copy()
        prepared = ptycho._prepared_direct[1]["phase"]
        changed = ptycho.run(
            optics,
            method="phase",
            phase_regularization=0.1,
            upsampling_factor="auto",
        )
        assert ptycho._prepared_direct[1]["phase"] is prepared
        cp.testing.assert_array_equal(phase.object_wave, original)
        assert not cp.array_equal(changed.object_wave, original)
        preview, loss = ptycho.preview(
            dict(optics),
            method="phase",
            phase_regularization=0.1,
            sample_tilt_mrad=optics.sample_tilt_mrad,
            depth_spread_nm=optics.depth_spread_nm,
            upsampling_factor="auto",
        )
        cp.testing.assert_array_equal(cp.asarray(preview), changed.phase)
        assert (
            loss is None
        )  # Do not relabel prediction error as SSB phase variance.
        complex_result = ptycho.run(
            optics,
            method="complex",
            amplitude_regularization=0.1,
            upsampling_factor="auto",
        )
        assert complex_result.amplitude_estimated
        assert (
            complex_result.reconstruction_options["amplitude_regularization"] == 0.1
        )
        assert (
            complex_result.reconstruction_diagnostics["training_relative_sse"] >= 0
        )
        before = ptycho._prepared_direct[1]["complex"]
        ptycho.run(
            optics,
            method="complex",
            aberrations={"C10_nm": optics["C10_nm"] + 1},
            upsampling_factor="auto",
        )
        assert ptycho._prepared_direct[1]["complex"] is not before
        restored = ptycho.run(
            optics,
            method="complex",
            amplitude_regularization=0.1,
            upsampling_factor="auto",
        )
        cp.testing.assert_array_equal(
            restored.object_wave, complex_result.object_wave
        )


def test_saved_alternatives_survive_and_reload_without_solving(
    acquisition, tmp_path, monkeypatch
):
    """Reopen saved phase and complex results and preserve earlier strengths."""
    from quantem.gpu.direct_ptycho.cuda.direct import PreparedDirect

    source, calibration = acquisition
    options = dict(backend="cuda", **calibration)
    with DirectPtycho.open(source, **options) as full, full.select_region((64, 112, 64, 112)) as ptycho:
        optics = ptycho.find_aberrations(
            tilt=True,
            trials=12,
            search="primary",
            check_rotation=False,
            show=False,
            verbose=False,
        )
        a = ptycho.run(optics, method="phase", save_dir=tmp_path)
        b = ptycho.run(
            optics, method="phase", phase_regularization=0.1, save_dir=tmp_path
        )
        c = ptycho.run(optics, method="complex", save_dir=tmp_path)
        assert len({a.saved_path, b.saved_path, c.saved_path}) == 3
        assert all(result.saved_path.is_file() for result in (a, b, c))
        expected = a.phase.get()

    def unexpected(*args, **kwargs):
        raise AssertionError("An exact saved result must not prepare or solve again.")

    monkeypatch.setattr(PreparedDirect, "prepare", unexpected)
    with DirectPtycho.open(source, **options) as full, full.select_region((64, 112, 64, 112)) as ptycho:
        restored = ptycho.run(optics, method="phase", save_dir=tmp_path)
        assert restored.reused
        np.testing.assert_array_equal(restored.phase.get(), expected)
        assert restored.reconstruction_options == a.reconstruction_options
        assert restored.reconstruction_diagnostics == a.reconstruction_diagnostics
