from __future__ import annotations

import numpy as np


def test_calibration_products_cache_roundtrip(tmp_path) -> None:
    from quantem.gpu.calibration import (
        CalibrationProducts,
        _load_calibration_products_cache,
        _save_calibration_products,
        _source_fingerprint,
        calibration_products_cache_path,
    )

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = calibration_products_cache_path(master, tmp_path / "cache")
    metadata = {
        "version": 1,
        "source": _source_fingerprint(master),
        "parameters": {
            "center": [3.5, 4.5],
            "radius_px": 2.0,
            "rotation_deg": 17.0,
            "use_transpose": False,
        },
        "timing": {"elapsed_s": 1.0},
    }
    products = CalibrationProducts(
        mean_dp=np.arange(9, dtype=np.float32).reshape(3, 3),
        bf=np.ones((4, 4), dtype=np.float32),
        df=np.full((4, 4), 2.0, dtype=np.float32),
        com_row=np.full((4, 4), 3.0, dtype=np.float32),
        com_col=np.full((4, 4), 4.0, dtype=np.float32),
        center=(3.5, 4.5),
        radius_px=2.0,
        rotation_deg=17.0,
        use_transpose=False,
        metadata=metadata,
    )

    _save_calibration_products(products, cache_path)
    loaded = _load_calibration_products_cache(cache_path, master)

    assert loaded is not None
    assert loaded.loaded_from_cache is True
    assert loaded.cache_path == cache_path
    assert loaded.center == (3.5, 4.5)
    assert loaded.radius_px == 2.0
    assert loaded.rotation_deg == 17.0
    assert loaded.use_transpose is False
    np.testing.assert_array_equal(loaded.mean_dp, products.mean_dp)
    np.testing.assert_array_equal(loaded.bf, products.bf)
    np.testing.assert_array_equal(loaded.df, products.df)
    np.testing.assert_array_equal(loaded.com_row, products.com_row)
    np.testing.assert_array_equal(loaded.com_col, products.com_col)


def test_calibration_products_cache_rejects_stale_source(tmp_path) -> None:
    from quantem.gpu.calibration import (
        CalibrationProducts,
        _load_calibration_products_cache,
        _save_calibration_products,
        _source_fingerprint,
        calibration_products_cache_path,
    )

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"first")
    cache_path = calibration_products_cache_path(master, tmp_path / "cache")
    metadata = {
        "version": 1,
        "source": _source_fingerprint(master),
        "parameters": {
            "center": [0.0, 0.0],
            "radius_px": 1.0,
            "rotation_deg": 0.0,
            "use_transpose": False,
        },
    }
    products = CalibrationProducts(
        mean_dp=np.zeros((2, 2), dtype=np.float32),
        bf=np.zeros((2, 2), dtype=np.float32),
        df=np.zeros((2, 2), dtype=np.float32),
        com_row=np.zeros((2, 2), dtype=np.float32),
        com_col=np.zeros((2, 2), dtype=np.float32),
        center=(0.0, 0.0),
        radius_px=1.0,
        rotation_deg=0.0,
        use_transpose=False,
        metadata=metadata,
    )

    _save_calibration_products(products, cache_path)
    master.write_bytes(b"changed")

    assert _load_calibration_products_cache(cache_path, master) is None


def test_load_calibration_products_cache_hit_is_backend_neutral(tmp_path) -> None:
    from quantem.gpu.calibration import (
        CalibrationProducts,
        _save_calibration_products,
        _source_fingerprint,
        calibration_products_cache_path,
        load_calibration_products,
    )

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = calibration_products_cache_path(master, tmp_path / "cache")
    metadata = {
        "version": 1,
        "source": _source_fingerprint(master),
        "parameters": {
            "center": [8.0, 9.0],
            "radius_px": 4.0,
            "rotation_deg": 23.0,
            "use_transpose": True,
        },
    }
    products = CalibrationProducts(
        mean_dp=np.ones((4, 4), dtype=np.float32),
        bf=np.ones((8, 8), dtype=np.float32),
        df=np.full((8, 8), 2.0, dtype=np.float32),
        com_row=np.full((8, 8), 3.0, dtype=np.float32),
        com_col=np.full((8, 8), 4.0, dtype=np.float32),
        center=(8.0, 9.0),
        radius_px=4.0,
        rotation_deg=23.0,
        use_transpose=True,
        metadata=metadata,
    )
    _save_calibration_products(products, cache_path)

    loaded = load_calibration_products(
        master,
        backend="mps",
        cache_dir=tmp_path / "cache",
    )

    assert loaded.loaded_from_cache is True
    assert loaded.use_transpose is True
    assert loaded.rotation_deg == 23.0
    np.testing.assert_array_equal(loaded.bf, products.bf)


def test_calibration_memory_plan_scales_to_one_full_chunk() -> None:
    from quantem.gpu.calibration import _calibration_memory_plan_for_shapes

    plan = _calibration_memory_plan_for_shapes(
        (1024, 1024),
        (192, 192),
        np.dtype(np.uint16).itemsize,
        24.0,
    )
    assert plan.memory_budget_source == "user"
    assert plan.chunk_rows == 170
    assert plan.chunk_count == 7
    assert plan.chunk_rows_source == "budget"

    mid_plan = _calibration_memory_plan_for_shapes(
        (1024, 1024),
        (192, 192),
        np.dtype(np.uint16).itemsize,
        48.0,
    )
    assert mid_plan.chunk_rows == 341
    assert mid_plan.chunk_count == 4

    large_plan = _calibration_memory_plan_for_shapes(
        (1024, 1024),
        (192, 192),
        np.dtype(np.uint16).itemsize,
        96.0,
    )
    assert large_plan.chunk_rows == 1024
    assert large_plan.chunk_count == 1

    huge_plan = _calibration_memory_plan_for_shapes(
        (1024, 1024),
        (192, 192),
        np.dtype(np.uint16).itemsize,
        2048.0,
    )
    assert huge_plan.chunk_rows == 1024
    assert huge_plan.chunk_count == 1


def test_calibration_memory_plan_user_chunk_rows_override() -> None:
    from quantem.gpu.calibration import (
        _calibration_memory_plan_for_shapes,
        _calibration_memory_plan_with_chunk_rows,
    )

    plan = _calibration_memory_plan_for_shapes(
        (1024, 1024),
        (192, 192),
        np.dtype(np.uint16).itemsize,
        24.0,
    )
    override = _calibration_memory_plan_with_chunk_rows(plan, 64)

    assert override.chunk_rows == 64
    assert override.chunk_rows_source == "user"
    assert override.chunk_count == 16


def test_calibration_memory_plan_auto_uses_large_free_vram(monkeypatch) -> None:
    import quantem.gpu.calibration as calibration

    monkeypatch.setattr(calibration, "_cuda_memory_info_gb", lambda: (97.0, 98.0))

    plan = calibration._calibration_memory_plan_for_shapes(
        (1024, 1024),
        (192, 192),
        np.dtype(np.uint16).itemsize,
        None,
    )

    assert plan.memory_budget_source == "auto_cuda"
    assert plan.chunk_rows == 1024
    assert plan.chunk_count == 1
    assert plan.cuda_free_gb == 97.0
