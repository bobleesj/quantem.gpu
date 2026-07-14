from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_SSB_MASTER")
    or not os.environ.get("QUANTEM_GPU_SSB_REFERENCE_NPZ"),
    reason=(
        "set QUANTEM_GPU_SSB_MASTER and QUANTEM_GPU_SSB_REFERENCE_NPZ "
        "for real-data CUDA/MPS SSB reference parity"
    ),
)
def test_mps_ssb_fixed_aberration_matches_cuda_reference() -> None:
    """Compare MPS SSB fixed-aberration output against a CUDA reference artifact."""
    from quantem.gpu.io import load
    from quantem.gpu.ssb.mps import ssb_preview

    master = Path(os.environ["QUANTEM_GPU_SSB_MASTER"])
    reference_path = Path(os.environ["QUANTEM_GPU_SSB_REFERENCE_NPZ"])
    if not master.exists():
        pytest.skip(f"master not available: {master}")
    if not reference_path.exists():
        pytest.skip(f"reference not available: {reference_path}")

    reference = np.load(reference_path)
    reference_meta = json.loads(str(reference["meta"]))
    result = ssb_preview(
        load(str(master), backend="mps", verbose=False).data,
        voltage_kV=300,
        semiangle_mrad=21.4,
        scan_sampling_A=1.0,
        C10=0.0,
        C12=50.0,
        phi12=0.0,
        bf_radius=5,
        chunk_bf=16,
        compute_loss=True,
        verbose=False,
    )

    assert result.num_bf == int(reference_meta["num_bf"])
    assert np.allclose(result.bf_center, reference_meta["bf_center"], atol=1e-4)
    if reference_meta.get("loss_full") is not None:
        assert abs(float(result.loss) - float(reference_meta["loss_full"])) < 0.01

    phase_diff = np.abs(reference["phase"] - result.phase)
    amp_diff = np.abs(reference["amplitude"] - result.amplitude)
    assert float(np.mean(phase_diff)) < 0.02
    assert float(np.quantile(phase_diff.reshape(-1), 0.99)) < 0.05
    assert float(np.mean(amp_diff)) < 0.02
