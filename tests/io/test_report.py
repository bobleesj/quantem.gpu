"""Acquisition reports expose ordinary tables without decoding measurements."""

import h5py
import numpy as np
import pandas as pd
import pytest

from quantem.gpu import io


def test_file_report_reads_headers_and_returns_selectable_columns(tmp_path, monkeypatch):
    path = tmp_path / "acquisition.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("measurements", shape=(2, 3, 4, 4), dtype="uint16")
    read = h5py.Dataset.__getitem__

    def header_only(dataset, key):
        assert dataset.name != "/measurements", "A report must not read detector values"
        return read(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", header_only)
    table = io.report(path)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["Value", "Unit", "Metadata"]
    assert table.loc["Scan shape", "Value"] == "2 × 3"
    assert table.loc["Diffraction shape", "Value"] == "4 × 4"
    assert table.loc["Frames", "Value"] == 6
    assert table.loc["Accelerating voltage", "Metadata"] == "Not recorded"
    assert list(table[["Value", "Unit"]].columns) == ["Value", "Unit"]
    pd.testing.assert_frame_equal(table, io.inspect(path)._summary())


def test_loaded_report_preserves_metadata_and_rejects_file_selectors():
    data = io.Dataset4dstemGPU(np.zeros((2, 3, 4, 4), dtype=np.uint16), {"voltage_kV": 300})
    pd.testing.assert_frame_equal(io.report(data), data.metadata_table())
    assert io.report(data).loc["Accelerating voltage", "Value"] == 300
    for options in ({"scan_shape": (2, 3)}, {"dataset_path": "/measurements"}):
        with pytest.raises(ValueError, match="omit them for a loaded acquisition"):
            io.report(data, **options)
