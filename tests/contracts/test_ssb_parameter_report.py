"""Inspect and export fitted optics without changing their numerical values."""

import json

import numpy as np
import pandas as pd
import pytest

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
        elapsed_s=12.345678901,
        timings={"joint_fit_s": 12.345678901, "refinement_s": 7.123456789},
        sample_agreement=0.987654321,
        loss=0.044388994574546814,
        num_trials=200,
        refinement="newton",
        refine_nfev=250,
        optimizer={"stop": "evaluation limit"},
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
    assert table["Elapsed (s)"].iloc[0] == record["elapsed_s"] == fit.elapsed_s
    assert table["Model agreement"].iloc[0] == record["sample_agreement"] == fit.sample_agreement
    assert table["Variance diagnostic loss"].iloc[0] == record["loss"] == fit.loss
    assert record["stop_reason"] == "evaluation limit"
    assert record["num_trials"] == 200 and record["refine_nfev"] == 250
    assert record["timings"] == fit.timings
    assert json.loads(json.dumps(record))["aberrations"] == coefficients
    html = fit._repr_html_()
    assert html.count("<table") == 1
    assert "C56 (nm)" in html and "phi56 (deg)" in html and "depth spread (nm)" in html
    assert "-17.4877" in html
    record["aberrations"]["C30_nm"] = 0.0
    record["timings"]["refinement_s"] = 0.0
    table.loc[table.index[0], "C10 (nm)"] = 0.0
    assert fit.aberrations == coefficients
    assert fit.report()["C10 (nm)"].iloc[0] == coefficients["C10_nm"]
    assert fit.timings["refinement_s"] == 7.123456789


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


def test_report_option_displays_saved_convergence_once(monkeypatch):
    """Use the normal table API while drawing an existing scalar history."""
    import IPython.display
    import matplotlib.pyplot as plt

    fit = SSBResult(
        backend="cuda", object_wave=np.ones((8, 8), dtype=np.complex64),
        aberrations={"C10_nm": 1.23456789},
        objective="phase_variance",
        trial_records=[
            {"trial": 0, "params": {"C10_nm": 1.0}, "loss": 0.3},
            {"trial": 1, "params": {"C10_nm": 1.23456789}, "loss": 0.2},
        ],
    )
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    initial_figures = plt.get_fignums()
    table = fit.report(show_convergence=True)
    pd.testing.assert_frame_equal(table, fit.report())
    assert len(displayed) == 1
    np.testing.assert_array_equal(displayed[0].axes[0].lines[0].get_ydata(), [0.3, 0.2])
    assert plt.get_fignums() == initial_figures  # no duplicate inline rendering
    assert fit.aberrations == {"C10_nm": 1.23456789}
    with pytest.raises(ValueError, match="export data"):
        fit.report(to_dict=True, show_convergence=True)
    fit.trial_records = None
    with pytest.raises(ValueError, match="no recorded search history"):
        fit.report(show_convergence=True)
    assert len(displayed) == 1
