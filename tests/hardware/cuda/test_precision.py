"""Precision exports, packed reopening and selected detector workflows."""

import os

import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)

from quantem.gpu import io
from quantem.gpu.detector import prepare


@pytest.mark.parametrize("dtype", ["float16", "scaled_uint16"])
def test_float_archive_loads_selected_packed_intensities(tmp_path, dtype, capsys):
    values = cp.linspace(0, 1716, 12 * 16 * 16 * 16, dtype=cp.float32).reshape(
        12, 16, 16, 16
    )
    values[:2] *= cp.float32(0.000001)
    path = tmp_path / "result_master.h5"
    io.save(path, values, dtype="float32")
    loaded = io.load(
        path, dtype=dtype, scan_region=(1, 7, 2, 11), detector_region=(1, 15, 0, 16)
    )
    report = loaded.metadata["precision"]
    selected = values[1:7, 2:11, 1:15]
    if dtype == "float16":
        expected = selected.astype(cp.float16).astype(cp.float32)
    else:
        expected = (
            cp.rint(selected.astype(cp.float64) / report["scale"])
            .astype(cp.uint16)
            .astype(cp.float64)
            * report["scale"]
        )
    expected = expected.astype(cp.float32)
    session = prepare(loaded)
    assert bool(cp.all(session.frame(0, output="native") == expected[0, 0]))
    mask = cp.ones((14, 16), cp.float32)
    assert bool(
        cp.allclose(
            session.masked_sum(mask, output="native"),
            expected.sum(axis=(2, 3)),
            rtol=2e-6,
        )
    )
    error = expected.astype(cp.float64) - selected.astype(cp.float64)
    assert report["rmse"] == pytest.approx(float(cp.sqrt(cp.mean(error**2))), rel=1e-12)
    assert report["values"] == selected.size
    assert report["range_scope"] == "complete source"
    assert "GPU measured across all loaded values" in capsys.readouterr().out
    loaded.close()


@pytest.mark.parametrize("dtype", ["float16", "scaled_uint16"])
def test_precision_export_reopens_and_resaves_without_changing_units(
    tmp_path, dtype, capsys
):
    values = cp.linspace(0, 100, 8 * 8 * 16 * 16, dtype=cp.float32).reshape(
        8, 8, 16, 16
    )
    path = tmp_path / "display_master.h5"
    io.save(path, values, dtype=dtype)
    loaded = io.load(path)
    report = loaded.metadata["precision"]
    expected = (
        values.astype(cp.float16).astype(cp.float32)
        if dtype == "float16"
        else cp.rint(values.astype(cp.float64) / report["scale"])
        .astype(cp.uint16)
        .astype(cp.float64)
        * report["scale"]
    )
    expected = expected.astype(cp.float32)
    session = prepare(loaded)
    # Maximum code 65535 is a valid intensity, not a bad detector pixel.
    assert bool(cp.all(session.frame(63, output="native") == expected[7, 7]))
    assert bool(
        cp.allclose(
            cp.asarray(session.mean_dp()), expected.mean(axis=(0, 1)), rtol=2e-6
        )
    )
    copy = tmp_path / "copy_master.h5"
    io.save(copy, loaded)
    reopened = io.load(copy)
    assert reopened.metadata["precision"] == report
    assert bool(cp.all(prepare(reopened).frame(0, output="native") == expected[0, 0]))
    assert "Saved conversion report (not remeasured" in capsys.readouterr().out
    loaded.close()
    reopened.close()


def test_separate_regions_share_global_scale(tmp_path):
    values = cp.linspace(-10, 10, 8 * 8 * 16 * 16, dtype=cp.float32).reshape(
        8, 8, 16, 16
    )
    path = tmp_path / "signed_master.h5"
    io.save(path, values, dtype="float32")
    regions = io.load(
        path,
        dtype="scaled_uint16",
        scan_region=[(0, 2, 0, 2), (6, 8, 6, 8)],
        verbose=False,
    )
    assert (
        regions[0].metadata["precision"]["scale"]
        == regions[1].metadata["precision"]["scale"]
    )
    assert regions[0].metadata["precision"]["offset"] == -10
    for region in regions:
        region.close()


def test_saved_region_retains_geometry_and_prevents_raw_code_casts(tmp_path):
    values = cp.linspace(10, 20, 8 * 8 * 16 * 16, dtype=cp.float32).reshape(
        8, 8, 16, 16
    )
    original = tmp_path / "display_master.h5"
    io.save(original, values, dtype="scaled_uint16")
    selected = io.load(
        original,
        scan_region=(2, 3, 4, 5),
        detector_region=(2, 14, 1, 13),
        verbose=False,
    )
    saved = tmp_path / "region_master.h5"
    io.save(saved, selected)
    reopened = io.load(saved, verbose=False)
    assert reopened.shape == (1, 1, 12, 12)
    assert reopened.metadata["scan_shape"] == (1, 1)
    assert reopened.metadata["detector_shape"] == (12, 12)
    assert bool(
        cp.all(
            prepare(reopened).frame(0, output="native")
            == prepare(selected).frame(0, output="native")
        )
    )
    with pytest.raises(ValueError, match="restore its units"):
        io.load(original, dtype="float32", representation="dense")
    selected.close()
    reopened.close()


def test_maped_tensor_exports_and_wide_scaled_intensities(tmp_path):
    import torch

    values = (
        cp.linspace(-3e38, 3e38, 2 * 2 * 16 * 16, dtype=cp.float64)
        .astype(cp.float32)
        .reshape(2, 2, 16, 16)
    )
    path = tmp_path / "wide_master.h5"
    io.save(path, torch.from_dlpack(values), dtype="scaled_uint16")
    with io.load(path, verbose=False) as loaded:
        result = prepare(loaded).frame(0, output="native")
        assert bool(cp.all(cp.isfinite(result)))
        assert loaded.metadata["precision"]["overflow"] == 0
