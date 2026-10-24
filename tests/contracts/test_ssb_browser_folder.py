"""MPS SSB reads the bright-field columns of a ShowPtycho folder exactly as quantem.widget writes it.

``ShowPtycho.export`` (quantem.widget ``showptycho/export.py``) writes ``snapshots/cal.json`` as
``quantem.ssb-browser-calibration/3``, ``snapshots/manifest.json`` as ``quantem.ssb-browser-folder/3`` and the
counts under ``source/``. The fixture below is that layout written by hand, every key the widget writes included.
"""

import json

import numpy as np
import pytest

from quantem.gpu.ssb.mps.frames import load_bf_columns_mps

COLUMNS = np.arange(3 * 16, dtype=np.uint8).reshape(3, 16)


def write_widget_folder(folder, *, schema="quantem.ssb-browser-calibration/3", scan_region=(2, 6, 1, 5)):
    """A ShowPtycho folder of a 4 x 4 crop at rows 2:6, cols 1:5 of a 8 x 8 scan, three bright-field pixels."""
    (folder / "source").mkdir(parents=True)
    (folder / "snapshots").mkdir()
    (folder / "source" / "bf_columns.u8").write_bytes(COLUMNS.tobytes())
    calibration = {
        "schema": schema,
        "source_path": "redacted_local_source",
        "scan_region": list(scan_region),
        "scan_shape": [4, 4],
        "source_scan_shape": [8, 8],
        "precision": {"real_dtype": "float32", "complex_dtype": "complex64"},
        "num_bright_field_pixels": 3,
        "g_shape": [3, 4, 4],
        "g_dtype": "complex64_interleaved_re_im_native_le",
        "phase_shape": [4, 4],
        "phase_dtype": "float32_native_le",
        "detector_shape": [3, 3],
        "bright_field_radius_px": 1.25,
        "bright_field_center_px": [1.0, 1.5],
        "bright_field_rows": [0, 1, 1],
        "bright_field_cols": [1, 1, 2],
        "bright_field_k_row": [-0.1, 0.0, 0.0],
        "bright_field_k_col": [-0.05, -0.05, 0.05],
        "scan_q_row": [0.0, 0.25, -0.5, -0.25],
        "scan_q_col": [0.0, 0.25, -0.5, -0.25],
        "aperture_k": [1.0, 1.0, 1.0],
        "alpha_k2": [0.0, 0.0, 0.0],
        "cos2phi_k": [1.0, 1.0, 1.0],
        "sin2phi_k": [0.0, 0.0, 0.0],
        "wavelength_A": 0.019687,
        "voltage_kV": 300.0,
        "semiangle_mrad": 20.0,
        "scan_sampling_A": [0.5, 0.5],
        "detector_sampling_mrad": [0.7, 0.8],
        "scan_detector_rotation_deg": 197.0,
        "scan_detector_rotation_flipped": False,
        "aberrations": {"C10_nm": -10.0, "C12_nm": 0.4, "phi12_deg": 30.0},
        "display_negate_phase": False,
        "dc_value": [376.0, 0.0],
        "gpu_memory_gb": 4.5,
        "source_transport": "bf_columns",
        "source_files": ["source/bf_columns.u8"],
        "source_decode_dtype": "uint8",
        "persistent_bf_cache": False,
        "bf_column_companion": True,
        "bf_column_companion_path": "source/bf_columns.u8",
        "bf_column_encoding": "uint8",
        "webgpu_source_policy": "bf_columns_preferred_exact",
    }
    (folder / "snapshots" / "cal.json").write_text(json.dumps(calibration, indent=2))
    bf_columns = {
        "kind": "bf_columns", "path": "source/bf_columns.u8", "url": "source/bf_columns.u8", "dtype": "uint8",
        "encoding": "uint8", "num_bright_field_pixels": 3, "scan_shape": [4, 4], "plane": 16,
        "bytes_per_bright_field_pixel": 16, "bits_per_value": 8, "bytes": COLUMNS.nbytes, "max_value": 47, "link": "copy",
    }
    (folder / "snapshots" / "manifest.json").write_text(json.dumps({
        "schema": "quantem.ssb-browser-folder/3", "title": "ShowPtycho", "index": "index.html",
        "calibration": "snapshots/cal.json",
        "source": {"kind": "bf_columns", "bf_columns": bf_columns, "preferred_browser_source": "bf_columns"},
        "arrays": {}, "persistent_arrays": [],
    }, indent=2))
    (folder / "snapshots" / "snapshots.json").write_text(json.dumps({"schema": "quantem.ssb-snapshots/1", "snapshots": []}))


def test_reads_the_columns_and_calibration_of_a_widget_folder(tmp_path):
    write_widget_folder(tmp_path / "showptycho")
    frames = load_bf_columns_mps(tmp_path / "showptycho")
    assert frames.scan_shape == (4, 4)
    assert frames.detector_shape == (3, 3)
    assert frames.selection.bright_field_center_px == (1.0, 1.5)
    assert frames.selection.bright_field_radius_px == 1.25
    assert frames.detector_sampling_mrad == (0.7, 0.8)
    assert frames.metadata == {"voltage_kV": 300.0, "semiangle_mrad": 20.0, "scan_sampling_A": [0.5, 0.5]}
    assert frames.dc_value == complex(376.0, 0.0)
    assert frames.max_value == 47
    np.testing.assert_array_equal(frames.selection.rows, [0, 1, 1])
    np.testing.assert_array_equal(frames.selection.cols, [1, 1, 2])
    np.testing.assert_array_equal(frames.columns(np.array([1, 0]), np.array([2, 1])), COLUMNS[[2, 0]])


def test_transposing_exact_columns_preserves_counts_and_original_coordinates(tmp_path):
    write_widget_folder(tmp_path / "showptycho")
    frames = load_bf_columns_mps(tmp_path / "showptycho")
    view = frames.transpose_detector()
    assert view._columns is frames._columns
    assert view.selection.bright_field_center_px == (1.5, 1.0)
    assert view.detector_sampling_mrad == (.8, .7)
    np.testing.assert_array_equal(view.columns([2, 1], [1, 0]), COLUMNS[[2, 0]])
    assert frames.selection.bright_field_center_px == (1.0, 1.5)
    restored = view.transpose_detector()
    np.testing.assert_array_equal(restored.columns([1, 0], [2, 1]), COLUMNS[[2, 0]])


@pytest.mark.parametrize("schema", ["quantem.ssb-calibration/2", "quantem.ssb-browser-calibration/2", None])
def test_another_calibration_layout_is_refused_by_name(tmp_path, schema):
    write_widget_folder(tmp_path / "showptycho", schema=schema)
    with pytest.raises(ValueError, match="quantem.ssb-browser-calibration/3"):
        load_bf_columns_mps(tmp_path / "showptycho")


def test_a_scan_region_that_does_not_span_the_scan_shape_is_refused(tmp_path):
    write_widget_folder(tmp_path / "showptycho", scan_region=(2, 6, 1, 6))
    with pytest.raises(ValueError, match="scan_region"):
        load_bf_columns_mps(tmp_path / "showptycho")


def test_python_fit_calibration_does_not_block_acquisition_loading(tmp_path):
    """Reopening saved Python optics falls back to the original acquisition."""
    from quantem.gpu.ssb.calibration import CALIBRATION_SCHEMA
    from quantem.gpu.ssb.mps.frames import find_bf_columns

    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"schema": CALIBRATION_SCHEMA}))
    assert find_bf_columns(str(tmp_path / "acquisition.h5"), str(calibration)) is None


def test_python_calibration_cannot_disguise_a_broken_column_declaration(tmp_path):
    from quantem.gpu.ssb.calibration import CALIBRATION_SCHEMA

    write_widget_folder(tmp_path / "showptycho", schema=CALIBRATION_SCHEMA)
    with pytest.raises(ValueError, match="quantem.ssb-browser-calibration/3"):
        load_bf_columns_mps(tmp_path / "showptycho")
