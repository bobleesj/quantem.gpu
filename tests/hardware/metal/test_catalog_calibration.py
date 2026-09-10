"""Independent NXem calibration fixtures through the native catalog API."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import h5py
import numpy as np


@unittest.skipUnless(os.environ.get("CATALOG_CALIBRATION_EXE"), "Build CatalogCalibrationParity")
class CatalogCalibrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.master = self.root / "acquisition_master.h5"
        self.companion = self.root / "acquisition_em_metadata.h5"
        with h5py.File(self.root / "acquisition_data_000001.h5", "w") as f:
            f.create_dataset("entry/data/data", data=np.zeros((6, 8, 8), dtype="u2"),
                             chunks=(1, 8, 8))
        with h5py.File(self.master, "w") as f:
            group = f.create_group("entry/data")
            group.attrs["scan_shape"] = np.array([2, 3], dtype="u8")
            group["data_000001"] = h5py.ExternalLink("acquisition_data_000001.h5", "/entry/data/data")

    def metadata(self, row=0.4, column=0.6, units="nm", rows=2, frames=1):
        with h5py.File(self.companion, "w") as f:
            scan = f.create_group("electron_microscope/scan_controller")
            scan["scan_type"] = "regular"
            regular = scan.create_group("regular_scan")
            regular["n_pixels_y"] = rows
            regular["n_pixels_x"] = 3
            regular["n_frames"] = frames
            for axis, value in (("y", row), ("x", column)):
                d = regular.create_dataset("pixel_size_" + axis, data=value)
                if units is not None:
                    d.attrs["units"] = units

    def read(self):
        result = subprocess.run([os.environ["CATALOG_CALIBRATION_EXE"], str(self.master),
                                 str(self.root / "cache")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)["datasets"][0]

    def test_units_axes_cache_and_removal(self):
        self.assertNotIn("sourceScanCalibration", self.read())
        for factor, unit in ((1, "nm"), (1e-9, "m"), (1e-3, "um")):
            self.metadata(row=0.4 * factor, column=0.6 * factor, units=unit)
            dataset = self.read()
            for run in (dataset, self.read()):
                calibration = run["sourceScanCalibration"]
                self.assertAlmostEqual(calibration["rowSamplingAngstrom"], 4)
                self.assertAlmostEqual(calibration["columnSamplingAngstrom"], 6)
                self.assertEqual(calibration["origin"], "source_metadata")
        self.metadata(row=0.8)
        self.assertAlmostEqual(self.read()["sourceScanCalibration"]["rowSamplingAngstrom"], 8)
        self.companion.unlink()
        self.assertNotIn("sourceScanCalibration", self.read())

    def test_invalid_calibration_stays_uncalibrated(self):
        for arguments in ({"units": None}, {"units": "mrad"}, {"rows": 3},
                          {"frames": 2}, {"row": 0}, {"row": -1},
                          {"row": float("nan")}, {"row": float("inf")}):
            with self.subTest(arguments=arguments):
                self.metadata(**arguments)
                self.assertNotIn("sourceScanCalibration", self.read())

    def test_unrelated_metadata_is_not_borrowed(self):
        self.metadata()
        self.companion.rename(self.root / "other_em_metadata.h5")
        self.assertNotIn("sourceScanCalibration", self.read())

    def test_microscope_metadata_units_and_pair_identity(self):
        self.metadata()
        with h5py.File(self.companion, "a") as f:
            values = {
                "electron_source/accelerating_voltage": (300000, "V"),
                "illumination_system/semi_convergence_angle": (0, "mrad"),
                "imaging_system/camera_length": (0.23, "m"),
                "imaging_system/reciprocal_pixel_size_y": (0.18217391304347827, "mrad"),
                "imaging_system/reciprocal_pixel_size_x": (0.00018217391304347827, "rad"),
                "scan_controller/regular_scan/dwell_time": (0.00005, "s"),
            }
            for path, (value, unit) in values.items():
                d = f.create_dataset("electron_microscope/" + path, data=value)
                d.attrs["units"] = unit
        dataset = self.read()
        self.assertEqual(dataset["metadata"]["sourceFormat"], "ARINA HDF5 + NXem metadata")
        self.assertEqual(dataset["metadata"]["microscope_metadata_source"], self.companion.name)
        self.assertAlmostEqual(dataset["kPixelSizeRow"], 0.18217391304347827, places=15)
        self.assertAlmostEqual(dataset["kPixelSizeCol"], 0.18217391304347827, places=15)
        self.assertEqual(dataset["kPixelUnit"], "mrad")
        with h5py.File(self.companion, "a") as f:
            f["electron_microscope/scan_controller/regular_scan/n_pixels_y"][()] = 3
        rejected = self.read()
        self.assertEqual(rejected["metadata"]["sourceFormat"], "ARINA HDF5")
        self.assertNotIn("kPixelSizeRow", rejected)


if __name__ == "__main__":
    unittest.main()
