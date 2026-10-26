"""Measure actual GPU redraws without mutating a calibrated session."""

import hashlib
import json
import sys

import numpy as np
import pytest

from quantem.gpu import SSB
from quantem.gpu.benchmark import run_direct_ptycho_preview


@pytest.mark.parametrize("sample", [False, True])
def test_changed_defocus_benchmark_preserves_calibration(sample):
    """Timing covers changed images; higher-order and sample terms survive."""
    backend = "mps" if sys.platform == "darwin" else "cuda"
    if backend == "mps":
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("Requires physical Metal hardware")
    else:
        cp = pytest.importorskip("cupy")
        if not cp.cuda.is_available():
            pytest.skip("Requires physical CUDA hardware")
    counts = np.random.default_rng(7).poisson(30, (128, 128, 16, 16)).astype(np.uint16)
    optics = {
        "C10_nm": 5.0,
        "C12_nm": 1.2,
        "phi12_deg": 12.0,
        "C21_nm": 30.0,
        "phi21_deg": 23.0,
        "C50_nm": 500000.0,
    }
    with SSB(
        counts,
        backend=backend,
        voltage_kV=300,
        semiangle_mrad=20,
        scan_sampling_A=0.5,
        detector_sampling_mrad=4,
        scan_detector_rotation_deg=11,
        aberrations=optics,
    ) as ssb:
        if sample:
            ssb.sample_tilt_mrad = (2.0, -3.0)
            ssb.depth_spread_nm = 12.0
        before = (dict(ssb.aberrations), ssb.sample_tilt_mrad, ssb.depth_spread_nm)
        reference, _ = ssb.preview(
            ssb.aberrations,
            upsampling_factor=1,
            compute_loss=False,
            sample_tilt_mrad=ssb.sample_tilt_mrad or (0, 0),
            depth_spread_nm=ssb.depth_spread_nm or 0,
        )
        if backend == "cuda":
            reference = cp.asnumpy(reference)
        report = run_direct_ptycho_preview(ssb, upsampling=(1, 2), runs=3)
        assert before == (ssb.aberrations, ssb.sample_tilt_mrad, ssb.depth_spread_nm)
        assert report["bright_field_pixels"] == ssb.num_bright_field_pixels
        assert report["browser_paint_measured"] is False
        for factor, result in report["measurements"].items():
            assert result["output_shape"] == [128 * int(factor)] * 2
            assert result["phase_delivery"]["warm_n"] == 2
            assert len({s["phase_sha256"] for s in result["samples"]}) == 3
        assert (
            report["measurements"]["1"]["samples"][0]["phase_sha256"]
            == hashlib.sha256(reference.tobytes()).hexdigest()
        )
        json.dumps(report)
