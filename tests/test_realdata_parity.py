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


def test_realdata_detector_products_match_legacy_widget_and_not_slower() -> None:
    """Real-data parity for the migrated detector helpers."""
    cp = _cupy()
    legacy_detector = pytest.importorskip("quantem.widget.detector")
    from quantem.gpu.detector import detect_bf_radius, dp_mean, virtual_image
    from quantem.gpu.io.hdf5 import load

    _require_vram(12.0)
    path = _realdata_master()
    _clean_gpu()
    data = load(path, verbose=False).data

    t0 = time.perf_counter()
    old_mean = legacy_detector.dp_mean(data)
    (old_row_col, old_radius) = legacy_detector.detect_bf_radius(old_mean)
    old_bf = legacy_detector.virtual_image(
        data, old_row_col[0], old_row_col[1], radius=old_radius,
    )
    cp.cuda.Stream.null.synchronize()
    old_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    new_mean = dp_mean(data)
    (new_row_col, new_radius) = detect_bf_radius(new_mean)
    new_bf = virtual_image(
        data, new_row_col[0], new_row_col[1], radius=new_radius,
    )
    cp.cuda.Stream.null.synchronize()
    new_s = time.perf_counter() - t1

    cp.testing.assert_array_equal(new_mean, old_mean)
    assert new_row_col == old_row_col
    assert new_radius == old_radius
    cp.testing.assert_array_equal(new_bf, old_bf)
    assert new_s <= old_s * 1.10 + 0.01

    del data, old_mean, old_bf, new_mean, new_bf
    _clean_gpu()
