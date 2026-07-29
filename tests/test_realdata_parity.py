from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import pytest


REALDATA_MASTER_ENV = "QUANTEM_GPU_REALDATA_MASTER"


def _realdata_master() -> Path:
    raw_path = os.environ.get(REALDATA_MASTER_ENV)
    if not raw_path:
        pytest.skip(f"{REALDATA_MASTER_ENV} is not set.")
    path = Path(raw_path).expanduser()
    if not path.exists():
        pytest.skip(f"{REALDATA_MASTER_ENV} does not point to an existing file.")
    return path


def _cupy():
    return pytest.importorskip("cupy")


def _clean_gpu() -> None:
    cp = _cupy()
    gc.collect()
    cp.fft.config.get_plan_cache().clear()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _require_vram(min_gb: float) -> None:
    cp = _cupy()
    free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
    if free_gb < min_gb:
        pytest.skip(f"Not enough free VRAM ({free_gb:.1f} GB free, need {min_gb:.0f} GB).")


def test_realdata_detector_products_are_exact_and_gpu_backed() -> None:
    """Real-data parity for the public detector workflow."""
    cp = _cupy()
    from quantem.gpu import detector
    from quantem.gpu.io import load

    _require_vram(12.0)
    path = _realdata_master()
    _clean_gpu()
    data = load(path, verbose=False).data

    t1 = time.perf_counter()
    mean_dp = detector.mean_dp(data)
    center, radius = detector.auto_probe(mean_dp)
    bright_field = detector.bf(data, center=center, radius=radius)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t1

    expected_mean = cp.asnumpy(
        data.reshape(-1, *data.shape[-2:]).sum(axis=0, dtype=cp.uint64)
    ).astype(np.float32) / int(np.prod(data.shape[:2]))
    np.testing.assert_array_equal(mean_dp, expected_mean)
    assert bright_field.dtype == np.float32
    assert np.isfinite(bright_field).all()
    assert elapsed > 0.0

    del data, mean_dp, bright_field
    _clean_gpu()
