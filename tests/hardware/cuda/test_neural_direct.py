"""Reconstruct one native scan with its validated frozen model, without a series.

Supply QEM_NEURAL_SOURCE, QEM_NEURAL_CALIBRATION and
QEM_NEURAL_CHECKPOINT. Keep measured inputs and weights outside the repo.
The calibration JSON has session constructor arguments, sample_tilt_mrad,
depth_spread_nm and scan_basis. Missing inputs skip this acceptance test.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import cupy as cp
import numpy as np
import pytest

from quantem.gpu import DirectPtycho, io
from quantem.gpu.direct_ptycho.neural_phase import NeuralPhaseModel
from quantem.gpu.direct_ptycho.cuda.direct import PreparedDirect

pytestmark = pytest.mark.slow


def test_reconstruct_compare_damping_and_restore(tmp_path):
    """One scan retains its direct phase and the same frozen correction at different baseline dampings."""
    paths = [os.environ.get("QEM_NEURAL_" + key)
             for key in ("SOURCE", "CALIBRATION", "CHECKPOINT")]
    if not all(paths):
        pytest.skip("Supply the native scan, calibration and validated checkpoint.")
    source, calibration_path, checkpoint = paths
    calibration = json.loads(Path(calibration_path).read_text())
    model = NeuralPhaseModel.load(checkpoint)
    with io.load(source, device="cuda:0", verbose=False) as data:
        with DirectPtycho(data, **calibration["session"]) as ptycho:
            ptycho.sample_tilt_mrad = tuple(calibration["sample_tilt_mrad"])
            ptycho.depth_spread_nm = calibration["depth_spread_nm"]
            settings = dict(method="phase", upsampling="auto", compute_loss=False,
                            scan_basis=calibration["scan_basis"])
            direct = ptycho.reconstruct(**settings)
            neural = ptycho.reconstruct(**settings, refinement=model)
            cp.testing.assert_array_equal(direct.phase, neural.phase)
            assert neural.neural_phase is not None and neural.annular_counts is not None
            assert neural.neural_phase.shape == direct.phase.shape
            assert neural.annular_counts.shape == data.shape[:2]

            tuned = ptycho.reconstruct(**settings, refinement=model, phase_damping=0.02,
                                       save_dir=tmp_path)
            cp.testing.assert_array_equal(tuned.neural_phase, neural.neural_phase)
            assert not cp.array_equal(tuned.phase, neural.phase)
            assert tuned.reconstruction_options["phase_damping"] == 0.02
            assert tuned.reconstruction_options["neural_phase_model"]["input_phase_damping"] == 0.03

            with patch.object(PreparedDirect, "prepare", side_effect=AssertionError("Unexpected solve")):
                restored = ptycho.reconstruct(**settings, refinement=model, phase_damping=0.02,
                                              save_dir=tmp_path)
            assert restored.reused
            cp.testing.assert_array_equal(restored.phase, tuned.phase)
            cp.testing.assert_array_equal(restored.neural_phase, tuned.neural_phase)
            cp.testing.assert_array_equal(restored.annular_counts, tuned.annular_counts)
            assert restored.reconstruction_options == tuned.reconstruction_options

            # An incomplete archive must be recomputed, never returned as a complete neural result.
            np.savez(restored.saved_path, object_wave=cp.asnumpy(restored.object_wave))
            repaired = ptycho.reconstruct(**settings, refinement=model, phase_damping=0.02,
                                          save_dir=tmp_path)
            assert not repaired.reused
            cp.testing.assert_array_equal(repaired.neural_phase, tuned.neural_phase)


def test_fit_staged_in_memory_and_score_unused_region(tmp_path):
    """Train from counts without a checkpoint, then evaluate separate scan data."""
    source = os.environ.get("QEM_NEURAL_SOURCE")
    calibration_path = os.environ.get("QEM_NEURAL_CALIBRATION")
    if not source or not calibration_path:
        pytest.skip("Supply the native scan and optical calibration.")
    calibration = json.loads(Path(calibration_path).read_text())
    with io.load(source, device="cuda:0", verbose=False) as data:
        with DirectPtycho(data, **calibration["session"]) as ptycho:
            ptycho.sample_tilt_mrad = tuple(calibration["sample_tilt_mrad"])
            ptycho.depth_spread_nm = calibration["depth_spread_nm"]
            settings = dict(method="phase", upsampling="auto", compute_loss=False,
                            scan_basis=calibration["scan_basis"])
            optics = ptycho.reconstruct(**settings)
            model = NeuralPhaseModel.fit(ptycho, optics, steps=2, direction_steps=1,
                                        scan_basis=calibration["scan_basis"])
            assert not list(tmp_path.iterdir())
            assert model.checkpoint_sha256 is None
            assert len(model._net["heads"]) == 2
            assert model.training_record["recipe"] == "staged"
            reconstruction = ptycho.reconstruct(optics, **settings, refinement=model)
            assert bool(cp.isfinite(reconstruction.neural_phase).all())
            model.save(tmp_path / "optional.pt")
            restored = NeuralPhaseModel.load(tmp_path / "optional.pt")
            replay = ptycho.reconstruct(optics, **settings, refinement=restored)
            cp.testing.assert_array_equal(replay.neural_phase, reconstruction.neural_phase)
            scores = model.score(ptycho, optics, partition="detector",
                                 scan_basis=calibration["scan_basis"])
            assert np.isfinite(scores.to_numpy()).all()
            assert scores.shape == (2, 3)
            # Compare cached differentiable training/score math with the
            # canonical detector residual and unchanged inference solver.
            import torch
            from quantem.gpu.direct_ptycho._annular_detector import annular_counts
            from quantem.gpu.direct_ptycho._phase_networks import Direction
            from quantem.gpu.direct_ptycho._phase_refinement import refine
            from quantem.gpu.direct_ptycho._phase_staged_training import _pair

            guide = torch.from_dlpack(annular_counts(ptycho, data, model.annulus_mrad)).float()
            cases = _pair(ptycho, optics, scores.attrs["scan_region"], model._factor,
                          model._sampling, guide, calibration["scan_basis"])
            net = model._on_device(cases[0].base.device)
            zero_head = Direction().to(cases[0].base.device)
            torch.nn.init.zeros_(zero_head.decode[-1].weight)
            torch.nn.init.zeros_(zero_head.decode[-1].bias)
            with torch.no_grad():
                for case in cases:
                    learned = case.unet_phase(net["unet"])
                    canonical = refine(case.source, case.base, case.guide, model._sampling,
                                       net["unet"], [zero_head], torch.zeros_like(net.readout),
                                       net.kernels, noise_level=1.0)
                    torch.testing.assert_close(learned, canonical, atol=2e-6, rtol=2e-5)
                    direct_error = case.target.phase_error(cp.from_dlpack(case.base))
                    learned_error = case.target.phase_error(cp.from_dlpack(learned))
                    assert direct_error - float(case.gain(learned)) == pytest.approx(learned_error, abs=1e-8)
            with pytest.raises(ValueError, match="disjoint"):
                model.score(ptycho, optics, partition="detector",
                            scan_region=model.training_record["train_region"])
