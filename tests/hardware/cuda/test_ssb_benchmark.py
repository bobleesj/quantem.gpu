"""Run the public CUDA benchmark through loading, fitting, and phase export."""

import json

import numpy as np
import pytest

from .test_ssb_thick_sample import _require_cuda

pytestmark = pytest.mark.slow


def test_benchmark_exports_cuda_phases_for_full_and_regional_fits(tmp_path):
    """The complete report includes device phases without implicit NumPy conversion."""
    from quantem.gpu.benchmark import run_ssb

    _require_cuda()
    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 24, 24)).astype(np.uint16)
    row, col = np.meshgrid(
        np.arange(24) - 11.5, np.arange(24) - 11.5, indexing="ij"
    )
    counts[:, :, np.hypot(row, col) < 8] += 200
    source = tmp_path / "scan.npy"
    np.save(source, counts)

    report = run_ssb(
        source,
        backend="cuda",
        runs=2,
        region_sizes=(64,),
        calibration={
            "voltage_kV": 300.0,
            "semiangle_mrad": 30.0,
            "scan_sampling_A": (0.3, 0.3),
            "detector_sampling_mrad": (3.75, 3.75),
        },
    )

    assert report["acquisition"]["shape"] == [128, 128, 24, 24]
    assert set(report["measurements"]) == {
        "full_field_fit",
        "full_field_reconstruction",
        "region_64",
        "move_64",
        "resize_63x64",
    }
    for measurement in report["measurements"].values():
        assert measurement["warm_n"] == 1
        assert all(check["bitwise_equal"] for check in measurement["repeat_parity"])
        for fit in measurement["fits"]:
            assert len(fit["phase_sha256"]) == 64
            assert fit["completed_trials"] == 200 * fit["rotation_branches"]
            assert fit["optimizer"]["device"] == "cuda"
            assert fit["optimizer"]["host_numerical_search"] is False
    assert str(source) not in json.dumps(report)
