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


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_SSB_MASTER"),
    reason="set QUANTEM_GPU_SSB_MASTER for real-data MPS SSB batch parity",
)
def test_mps_ssb_batched_loss_matches_scalar() -> None:
    """Keep batched MPS optimizer losses numerically pinned to scalar losses."""
    from quantem.gpu.io import load
    from quantem.gpu.ssb.mps import (
        _as_chunked_frames,
        _as_sampling,
        _bf_pixels,
        _prepare_selection,
        _reconstruct_prepared,
        _reconstruct_prepared_batch,
        _scan_shape,
    )

    master = Path(os.environ["QUANTEM_GPU_SSB_MASTER"])
    if not master.exists():
        pytest.skip(f"master not available: {master}")

    frames = _as_chunked_frames(load(str(master), backend="mps", verbose=False).data)
    scan_shape = _scan_shape(frames)
    det_shape = tuple(int(x) for x in frames.shape[-2:])
    bf_row, bf_col, center, _radius, detected_radius = _bf_pixels(frames, 0.5, 5)
    det_px = (2.0 * 21.4) / detected_radius
    prepared = _prepare_selection(
        frames,
        scan_shape=scan_shape,
        det_shape=det_shape,
        bf_row=bf_row,
        bf_col=bf_col,
        center=center,
        voltage_kV=300,
        semiangle_mrad=21.4,
        scan_sampling=_as_sampling(0.405),
        det_sampling=(det_px, det_px),
        rotation_angle_deg=0.0,
        chunk_bf=16,
    )
    params = np.asarray(
        [
            [-24.5, 0.65, -0.824],
            [0.0, 50.0, 0.0],
            [25.0, 20.0, 0.5],
            [-100.0, 80.0, -0.3],
        ],
        dtype=np.float32,
    )
    scalar_losses = []
    for c10, c12, phi12 in params:
        _object_wave, loss = _reconstruct_prepared(
            prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=16,
            compute_loss=True,
            compute_object=False,
        )
        scalar_losses.append(float(loss))

    batch_losses = _reconstruct_prepared_batch(
        prepared,
        C10=params[:, 0],
        C12=params[:, 1],
        phi12=params[:, 2],
        chunk_bf=16,
    )
    assert np.allclose(batch_losses, np.asarray(scalar_losses), atol=1e-6)


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
    det_shape = tuple(int(x) for x in frames.shape[-2:])
    bf_row, bf_col, center, _radius, detected_radius = _bf_pixels(frames, 0.5, 5)
    det_px = (2.0 * 21.4) / detected_radius
    prepared = _prepare_selection(
        frames,
        scan_shape=scan_shape,
        det_shape=det_shape,
        bf_row=bf_row,
        bf_col=bf_col,
        center=center,
        voltage_kV=300,
        semiangle_mrad=21.4,
        scan_sampling=_as_sampling(0.405),
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
    assert np.allclose(losses, reference["losses"], atol=6e-5)
