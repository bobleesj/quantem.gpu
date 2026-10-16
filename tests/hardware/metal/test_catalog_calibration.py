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

    def test_non_hdf5_folder_is_distinct_from_invalid_input(self):
        mixed = self.root / "mixed"
        mixed.mkdir()
        (mixed / "scan_x2_y3.raw").write_bytes(b"raw fixture placeholder")
        with h5py.File(mixed / "simulation.hdf5", "w") as handle:
            handle["dp"] = np.zeros((6, 2, 2), dtype="f4")
        result = subprocess.run(
            [os.environ["CATALOG_CALIBRATION_EXE"], str(mixed), str(self.root / "cache")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stderr.strip(), "EMPTY_CATALOG")
        result = subprocess.run(
            [os.environ["CATALOG_CALIBRATION_EXE"], str(mixed / "missing.h5"),
             str(self.root / "cache")], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("does not exist", result.stderr)

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

    def stem_metadata(self, calibrated=1, rows=2, pixel=0.05):
        with h5py.File(self.master.with_name("acquisition.h5"), "w") as f:
            attrs = f.create_group("STEM Metadata").attrs
            for key, value in {
                "NumPixelsY": rows, "NumPixelsX": 3, "Pixel Size": pixel,
                "IsPixelSizeCalibrated": calibrated, "High Tension": 300000.0,
                "Camera Length": 0.091, "NumPixelsPrescan": 16, "Rotation": 12,
            }.items():
                attrs[key] = [value]

    def test_stem_nominal_calibrated_cache_and_removal(self):
        self.stem_metadata(calibrated=0)
        nominal = self.read()
        self.assertNotIn("sourceScanCalibration", nominal)
        self.assertEqual(nominal["metadata"]["scan_sampling_nominal_A"], "0.5")
        self.assertIn("nominal", nominal["metadata"]["microscope_metadata_warning"])
        self.stem_metadata(calibrated=1)
        calibrated = self.read()
        self.assertNotEqual(nominal["id"], calibrated["id"])
        self.assertEqual(calibrated["sourceScanCalibration"]["rowSamplingAngstrom"], 0.5)
        self.assertEqual(calibrated["metadata"][
            "electron_microscope/electron_source/accelerating_voltage"], "300000.0")
        self.assertNotIn("kPixelSizeRow", calibrated)
        self.master.with_name("acquisition.h5").unlink()
        self.assertNotIn("sourceScanCalibration", self.read())

    def test_stem_invalid_or_wrong_grid_not_promoted(self):
        for arguments in ({"rows": 3}, {"pixel": 5e-324}, {"pixel": 0}):
            with self.subTest(arguments=arguments):
                self.stem_metadata(**arguments)
                result = self.read()
                self.assertNotIn("sourceScanCalibration", result)
                self.assertIn("microscope_metadata_warning", result["metadata"])

    def test_stem_microscope_voltage_takes_precedence(self):
        self.stem_metadata()
        with h5py.File(self.master, "a") as f:
            f["entry/instrument/detector/description"] = "Dectris ARINA Si"
            f["entry/instrument/detector/detectorSpecific/photon_energy"] = 200000.0
        metadata = self.read()["metadata"]
        self.assertIn("disagrees", metadata["microscope_metadata_warning"])
        self.assertEqual(metadata["electron_microscope/electron_source/accelerating_voltage"],
                         "300000.0")


if __name__ == "__main__":
    unittest.main()
