"""Read native HDF5 exports independently and compare every original scalar bit."""
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest


def test_scientific_export(tmp_path):
    executable = os.environ.get("SCIENTIFIC_METADATA_EXE")
    if not executable:
        pytest.skip("Set SCIENTIFIC_METADATA_EXE to the native parity executable")
    subprocess.run([executable, str(tmp_path)], check=True)
    with h5py.File(tmp_path / "scientific.h5", "r") as file:
        counts = file["images/counts"][:]
        assert counts.dtype == np.dtype("<u4")
        np.testing.assert_array_equal(counts, np.array([0, 1, 65535, 16777217, 4294967294, 4294967295], dtype="<u4").reshape(2, 3))
        measured = file["images/measurements"][:]
        assert measured.dtype == np.dtype("<f4")
        np.testing.assert_array_equal(measured.view("<u4"), np.array([0x80000000, 0x3f000000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc01234], dtype="<u4").reshape(3, 2))
        metadata = json.loads(file["metadata"][()])
        assert metadata["notes"] == "Crystalline area; 20 µs dwell."
        assert metadata["calibration"] == 0.4153
