"""Mean-dark/v1: float32 sample - mean(dark), with no clipping or rescaling.

Reference means and detector reductions use independent NumPy float64 sums.
The corrected DP rounds to float32 before subsequent products, matching the
public contract. Small fixtures exercise repeated selection and moving masks.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("EMPAD_SOURCE_PARITY_EXE"), "Build EMPADSourceParity first")
class EMPADBackgroundTests(unittest.TestCase):
    def test_confirmed_dark_corrects_every_product_without_changing_files(self):
        with tempfile.TemporaryDirectory(prefix="empad-dark-parity-") as directory:
            root = Path(directory)
            pixel = np.arange(16384).reshape(128, 128)
            sample = np.stack([(pixel % 19 - 7) * 0.25 + frame for frame in range(6)]).astype("<f4")
            # 258 frames crosses the streaming boundary. Constant binary fractions
            # make the expected calibration exactly representable in float32.
            dark = np.broadcast_to((pixel % 7) * 0.125 + 0.5, (258, 128, 128)).astype("<f4").copy()

            def write(name, values):
                folder = root / name
                folder.mkdir()
                path = folder / f"scan_x{len(values)}_y1.raw"
                records = np.zeros((len(values), 130, 128), dtype="<f4")
                records[:, :128] = values
                records.tofile(path)
                return path

            sample_path = write("sample", sample)
            dark_path = write("dark", dark)
            original = sample_path.read_bytes(), dark_path.read_bytes()
            expected = (sample - dark.mean(axis=0, dtype=np.float64).astype(np.float32)).astype(np.float32)
            output = root / "result"
            environment = dict(os.environ, EMPAD_TEST_METAL="1", EMPAD_TEST_BACKGROUND=str(dark_path),
                               EMPAD_TEST_APERTURE_SEQUENCE="1")
            for controls in ({}, {"QGPU_EMPAD_SERIAL_DETECTOR": "1", "QGPU_EMPAD_INCREMENTAL": "0",
                                  "QGPU_EMPAD_COM_CONTROL": "1"}):
                run = subprocess.run([os.environ["EMPAD_SOURCE_PARITY_EXE"], str(sample_path), str(output),
                                      "5,0,3,0"], env=environment | controls, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                selected = np.fromfile(output, dtype="<f4").reshape(4, 128, 128)
                np.testing.assert_array_equal(selected, expected[[5, 0, 3, 0]])
                self.assertTrue(np.any(selected < 0))
                np.testing.assert_allclose(np.fromfile(str(output) + ".mean", dtype="<f4").reshape(128, 128),
                                           expected.mean(axis=0, dtype=np.float64), rtol=1e-6, atol=1e-6)
                row, col = np.indices((128, 128))
                radius = (row - 64)**2 + (col - 64)**2
                masks = [radius <= 16**2, (radius >= 8**2) & (radius <= 16**2),
                         (radius >= 32**2) & (radius <= 63**2), np.ones((128, 128), dtype=bool)]
                sums = np.array([expected[:, mask].sum(axis=1, dtype=np.float64) for mask in masks])
                np.testing.assert_allclose(np.fromfile(str(output) + ".products", dtype="<f4").reshape(4, 6),
                                           sums, rtol=1e-6, atol=1e-6)
                total = expected.sum(axis=(1, 2), dtype=np.float64)
                for name, coordinates in [("com-row", row), ("com-column", col)]:
                    np.testing.assert_allclose(np.fromfile(str(output) + "." + name, dtype="<f4"),
                                               (expected * coordinates).sum(axis=(1, 2)) / total,
                                               rtol=1e-6, atol=1e-6)
                receipt = json.loads(Path(str(output) + ".capabilities.json").read_text())["residentReceipt"]
                sequence_masks = np.fromfile(str(output) + ".aperture-masks", dtype=np.uint8).reshape(96, 128, 128)
                sequence_expected = np.array([expected[:, mask != 0].sum(axis=1, dtype=np.float64)
                                              for mask in sequence_masks])
                np.testing.assert_allclose(np.fromfile(str(output) + ".apertures", dtype="<f4").reshape(96, 6),
                                           sequence_expected, rtol=1e-6, atol=1e-6)
                self.assertEqual(receipt["calibrationSchema"], "empad-mean-dark/v1")
                self.assertEqual(receipt["workingLogicalSHA256"], receipt["sourceRawLogicalSHA256"])
            self.assertEqual(original, (sample_path.read_bytes(), dark_path.read_bytes()))

    def test_nonfinite_dark_is_rejected_without_publishing_correction(self):
        with tempfile.TemporaryDirectory(prefix="empad-dark-invalid-") as directory:
            root = Path(directory)
            sample = root / "scan_x1_y1.raw"
            (root / "dark").mkdir()
            dark = root / "dark" / "scan_x1_y1.raw"
            values = np.zeros((130, 128), dtype="<f4")
            values.tofile(sample)
            values[0, 0] = np.nan
            values.tofile(dark)
            run = subprocess.run([os.environ["EMPAD_SOURCE_PARITY_EXE"], str(sample), str(root / "out"), "0"],
                                 env=dict(os.environ, EMPAD_TEST_METAL="1", EMPAD_TEST_BACKGROUND=str(dark)),
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("non-finite", run.stderr)
