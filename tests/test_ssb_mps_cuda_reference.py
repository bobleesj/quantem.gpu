from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_mps_sparse_row_indices_match_cuda_supported_sizes() -> None:
    """Pin MPS sparse objective row masks to the CUDA optimizer kernels."""
    from quantem.gpu.ssb.mps import _cuda_sparse_row_indices

    rows_128 = _cuda_sparse_row_indices((128, 128))
    assert rows_128.shape == (128,)
    assert rows_128[:12].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert rows_128[-8:].tolist() == [120, 121, 122, 123, 124, 125, 126, 127]

    rows_256 = _cuda_sparse_row_indices((256, 256))
    assert rows_256.shape == (128,)
    assert rows_256[:12].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
    assert rows_256[-4:].tolist() == [248, 249, 250, 251]

    rows_512 = _cuda_sparse_row_indices((512, 512))
    assert rows_512.shape == (128,)
    assert rows_512[:10].tolist() == [0, 1, 8, 9, 16, 17, 24, 25, 32, 33]
    assert rows_512[-2:].tolist() == [504, 505]

    with pytest.raises(ValueError, match="128x128, 256x256, or 512x512"):
        _cuda_sparse_row_indices((64, 64))


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
    phase_corr = np.corrcoef(reference["phase"].reshape(-1), result.phase.reshape(-1))[
        0, 1
    ]
    # MPS returns the same mean-of-per-BF-phase image as the CUDA fixed preview.
    assert float(phase_corr) > 0.999
    assert float(np.mean(phase_diff)) < 0.01
    assert float(np.quantile(phase_diff.reshape(-1), 0.99)) < 0.03


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_SSB_MASTER")
    or not os.environ.get("QUANTEM_GPU_SSB_SPARSE_REFERENCE_NPZ"),
    reason=(
        "set QUANTEM_GPU_SSB_MASTER and QUANTEM_GPU_SSB_SPARSE_REFERENCE_NPZ "
        "for real-data CUDA/MPS sparse optimizer parity"
    ),
)
def test_mps_ssb_sparse_optimizer_loss_matches_cuda_reference() -> None:
    """Compare MPS optimizer objective against pinned CUDA sparse losses."""
    from quantem.gpu.io import load
    from quantem.gpu.ssb.mps import (
        _as_chunked_frames,
        _as_sampling,
        _bf_pixels,
        _prepare_selection,
        _reconstruct_prepared_batch_cuda_sparse,
        _scan_shape,
    )

    master = Path(os.environ["QUANTEM_GPU_SSB_MASTER"])
    reference_path = Path(os.environ["QUANTEM_GPU_SSB_SPARSE_REFERENCE_NPZ"])
    if not master.exists():
        pytest.skip(f"master not available: {master}")
    if not reference_path.exists():
        pytest.skip(f"reference not available: {reference_path}")

    reference = np.load(reference_path)
    reference_meta = json.loads(str(reference["meta"]))
    assert reference_meta["objective"] == "cuda_sparse_variance_loss_batch"

    frames = _as_chunked_frames(load(str(master), backend="mps", verbose=False).data)
    scan_shape = _scan_shape(frames)
    if "scan_shape" in reference_meta:
        assert tuple(reference_meta["scan_shape"]) == scan_shape
    det_shape = tuple(int(x) for x in frames.shape[-2:])
    threshold = float(reference_meta.get("bf_intensity_threshold", 0.5))
    bf_radius = reference_meta.get("bf_radius_arg", 5)
    center_override = reference_meta.get("bf_center")
    if center_override is not None:
        center_override = (float(center_override[0]), float(center_override[1]))
    bf_row, bf_col, center, _radius, detected_radius = _bf_pixels(
        frames,
        threshold,
        bf_radius,
        center_override=center_override,
    )
    if "num_bf" in reference_meta:
        assert int(bf_row.size) == int(reference_meta["num_bf"])
    semiangle_mrad = float(reference_meta.get("semiangle_mrad", 21.4))
    det_px = (2.0 * semiangle_mrad) / detected_radius
    prepared = _prepare_selection(
        frames,
        scan_shape=scan_shape,
        det_shape=det_shape,
        bf_row=bf_row,
        bf_col=bf_col,
        center=center,
        voltage_kV=float(reference_meta.get("voltage_kV", 300)),
        semiangle_mrad=semiangle_mrad,
        scan_sampling=_as_sampling(float(reference_meta.get("scan_sampling_A", 0.405))),
        det_sampling=(det_px, det_px),
        rotation_angle_deg=0.0,
        chunk_bf=16,
    )

    params = reference["params"].astype(np.float32)
    losses = _reconstruct_prepared_batch_cuda_sparse(
        prepared,
        C10=params[:, 0],
        C12=params[:, 1],
        phi12=params[:, 2],
        chunk_bf=16,
    )
    assert np.allclose(losses, reference["losses"], atol=1e-4)
