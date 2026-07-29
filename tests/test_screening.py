from __future__ import annotations

import numpy as np


def _result(master, workflow):
    metadata = {
        "version": workflow._CACHE_VERSION,
        "source": workflow._source_fingerprint(master),
        "parameters": {
            "center": [3.5, 4.5],
            "radius_px": 2.0,
            "rotation_deg": 17.0,
            "transposed": False,
        },
    }
    zeros = np.zeros((4, 4), dtype=np.float32)
    return workflow.ScreeningResult(
        mean_dp=np.arange(9, dtype=np.float32).reshape(3, 3),
        bright_field=np.ones((4, 4), dtype=np.float32),
        dark_field=np.full((4, 4), 2.0, dtype=np.float32),
        dpc_phase=zeros.copy(),
        com_row=zeros.copy(),
        com_col=zeros.copy(),
        probe_center=(3.5, 4.5),
        probe_radius=2.0,
        rotation_deg=17.0,
        transposed=False,
        metadata=metadata,
    )


def test_screening_cache_roundtrip(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    expected = _result(master, workflow)

    workflow._save_cache(expected, cache_path)
    actual = workflow._prepare_cache(cache_path, master)

    assert actual is not None
    assert actual.from_cache is True
    assert actual.cache_path == cache_path
    assert actual.probe_center == (3.5, 4.5)
    assert actual.probe_radius == 2.0
    np.testing.assert_array_equal(actual.mean_dp, expected.mean_dp)
    np.testing.assert_array_equal(actual.bright_field, expected.bright_field)
    np.testing.assert_array_equal(actual.dark_field, expected.dark_field)
    assert actual.dpc_phase.dtype == np.float32


def test_screening_cache_rejects_changed_source(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"first")
    cache_path = workflow._cache_path(master, tmp_path / "cache")
    workflow._save_cache(_result(master, workflow), cache_path)

    master.write_bytes(b"changed")

    assert workflow._prepare_cache(cache_path, master) is None


def test_screening_forced_rotation_recomputes_phase(tmp_path) -> None:
    from quantem.gpu.screening import workflow

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    result = _result(master, workflow)
    rows, cols = np.indices((4, 4), dtype=np.float32)
    result.com_row = rows
    result.com_col = cols

    rotated = workflow._with_rotation(result, 35.0)

    assert rotated.rotation_deg == 35.0
    assert rotated.transposed is False
    assert rotated.dpc_phase.dtype == np.float32
    assert rotated.dpc_phase.shape == (4, 4)


def test_screening_rejects_packed_uint4_products() -> None:
    from quantem.gpu.screening.workflow import _screening_output_dtype

    try:
        _screening_output_dtype("u4")
    except ValueError as error:
        assert "derived screening products" in str(error)
    else:
        raise AssertionError("packed uint4 must not be accepted for derived products")
