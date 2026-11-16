"""2026-11-16: every scan says which detector pixels were filled, how, and from what mask."""

from types import SimpleNamespace

from quantem.gpu.io.metadata_display import calibration_table

RECORD = {"method": "median", "largest_window_px": 3, "pixel_count": 6, "applied": True,
          "coordinates_row_col": [[21, 73], [101, 88], [125, 52], [142, 70], [143, 150], [164, 191]],
          "source_flags": [2, 4, 8, 16, 20, 2]}


def test_the_calibration_table_shows_the_pixel_correction_and_its_mask():
    table = calibration_table({"voltage_kV": 80.0, "hot_pixel_correction": RECORD,
                               "pixel_mask_source": "the detector's pixel mask (DCU detector/config/pixel_mask)"})
    assert table.loc["Pixel correction", "Value"] == "median 3 x 3, 6 pixels"
    assert "DCU" in table.loc["Pixel correction", "Source"]


def test_direct_ptycho_names_a_filled_pixel_inside_its_bright_field_disk(capsys):
    from quantem.gpu.direct_ptycho.workflow import DirectPtycho

    ptycho = SimpleNamespace(bright_field_center_px=(104.3, 103.0), bright_field_radius_px=42.9)
    DirectPtycho._say_masked_pixels_in_disk(ptycho, {"hot_pixel_correction": RECORD})
    said = capsys.readouterr().out
    assert said.startswith("1 masked detector pixel inside the bright-field disk (101, 88): filled")
