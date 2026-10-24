"""Inspect and export fitted optics without changing their numerical values."""

import json

import numpy as np
import pandas as pd

from quantem.gpu.ssb.results import SSBResult


def test_inspect_all_fitted_coefficients_and_export_without_rounding():
    """Read high-order terms and geometry, then edit an independent export."""
    coefficients = {
        "C10_nm": -17.487747060046477,
        "C12_nm": 2.465943924895392,
        "phi12_deg": -31.98438599997024,
        "C30_nm": -3094.677734375,
        "C50_nm": 2030943.25,
        "C56_nm": 555512.25,
        "phi56_deg": -4.642517566680908,
    }
    fit = SSBResult(
        backend="cuda",
        object_wave=np.ones((8, 8), dtype=np.complex64),
        aberrations=coefficients,
        scan_detector_rotation_deg=358.31083229203097,
        sample_tilt_mrad=(-4.409721571110939, 5.255381849875908),
        depth_spread_nm=12.5519129023298,
        objective="broad_fifth_agreement",
    )
    table = fit.report()
    record = fit.report(to_dict=True)
    assert record["aberrations"] == coefficients
    for name, value in coefficients.items():
        coefficient, unit = name.rsplit("_", 1)
        assert table[f"{coefficient} ({unit})"].iloc[0] == value
    assert record["sample_tilt_mrad"] == fit.sample_tilt_mrad
    assert table["depth spread (nm)"].iloc[0] == record["depth_spread_nm"] == fit.depth_spread_nm
    assert record["scan_detector_rotation_deg"] == fit.scan_detector_rotation_deg
    assert json.loads(json.dumps(record))["aberrations"] == coefficients
    html = fit._repr_html_()
    assert html.count("<table") == 1
    assert "C56 (nm)" in html and "phi56 (deg)" in html and "depth spread (nm)" in html
    assert "-17.4877" in html
    record["aberrations"]["C30_nm"] = 0.0
    table.loc[table.index[0], "C10 (nm)"] = 0.0
    assert fit.aberrations == coefficients
    assert fit.report()["C10 (nm)"].iloc[0] == coefficients["C10_nm"]


def test_compare_reports_and_export_series_trial_ids():
    """Compare fits and retain the series report's existing trial-table meaning."""
    fit = SSBResult(
        backend="cuda",
        object_wave=np.ones((8, 8), dtype=np.complex64),
        aberrations={"C10_nm": 1.23456789, "C12_nm": 0.0, "phi12_deg": 0.0},
    )
    comparison = pd.concat({"first": fit.report(), "repeat": fit.report()})
    assert comparison["C10 (nm)"].tolist() == [1.23456789, 1.23456789]
    assert "C30 (nm)" not in comparison
    series = SSBResult(
        backend="cuda",
        object_wave=np.ones((2, 8, 8), dtype=np.complex64),
        aberrations={},
        trial_records=[
            {"trial": 12, "params": {"C10_nm": 1.23456789}, "loss": 0.123456789,
             "sample_tilt_mrad": [1.0, 2.0]},
            {"trial": 20, "params": {"C10_nm": 2.34567891}, "loss": 0.012345678},
        ],
    )
    record = series.report(to_dict=True)
    assert record[12]["C10_nm"] == 1.23456789
    assert record[20]["loss"] == 0.012345678
    pd.testing.assert_frame_equal(series.report(), series.trials)
    assert "trial" in series._repr_html_()
    record[12]["sample_tilt_mrad"][0] = 99.0
    table = series.report()
    table.loc[12, "sample_tilt_mrad"][1] = 99.0
    assert series.trial_records[0]["sample_tilt_mrad"] == [1.0, 2.0]
