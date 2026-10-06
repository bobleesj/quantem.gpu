"""Full original EMPAD float parity for the public AutoDisk Pd@Pt acquisition.

Source: https://github.com/swang59/AutoDisk_Demo (64x64 scan, 128x128 detector).
This is a real original RAW test, not a converted Cornell MATLAB export.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(
    os.environ.get("EMPAD_PUBLIC_RAW") and os.environ.get("EMPAD_SOURCE_PARITY_EXE"),
    "Set EMPAD_PUBLIC_RAW and build EMPADSourceParity",
)
class PublicEMPADParity(unittest.TestCase):
    def test_complete_original_float_samples_and_detector_products(self):
        raw = Path(os.environ["EMPAD_PUBLIC_RAW"])
        case = os.environ.get("EMPAD_PUBLIC_CASE", "autodisk")
        size, algorithm, checksum = {
            "autodisk": (64, "sha256", "c431b6ae2a506b189ae86fbaf254d97de7c9258383c6dde96c7a871d1a9f7805"),
            # Original EMPAD acquisition: https://zenodo.org/records/17246822.
            "mos2-mose2": (256, "md5", "c50f643c2cc87360bfdc746afd026cce"),
            # Original SnSe EMPAD: https://zenodo.org/records/10079791.
            "snse-7": (256, "md5", "3ddea699e1e09008f20405dc0331090a"),
            "snse-8": (256, "md5", "044c8497c431735ae1bfc09685c72029"),
            "snse-10": (256, "md5", "e76fb4991e6eb2c96995ddb45990dde5"),
        }[case]
        frames = size * size
        with raw.open("rb") as stream:
            digest = hashlib.file_digest(stream, algorithm).hexdigest()
        self.assertEqual(digest, checksum)
        self.assertEqual(raw.stat().st_size, frames * 130 * 128 * 4)
        records = np.memmap(raw, dtype="<f4", mode="r", shape=(frames, 130, 128))
        pixels = records[:, :128, :]
        with tempfile.TemporaryDirectory(prefix="empad-public-parity-") as directory:
            output = Path(directory) / "resident.bin"
            result = subprocess.run(
                [os.environ["EMPAD_SOURCE_PARITY_EXE"], str(raw), str(output),
                 "all", str(size), str(size)],
                env=dict(os.environ, EMPAD_TEST_METAL="1", EMPAD_TEST_BUDGET=str(max(1073741824, frames * 16384 * 5))),
                text=True, capture_output=True, timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            print(result.stdout, end="")
            actual = np.memmap(output, dtype="<u4", mode="r", shape=(frames, 128, 128))
            # Bit equality includes fractional/negative measurements and NaN payloads.
            for start in range(0, frames, 64):
                np.testing.assert_array_equal(actual[start:start + 64], pixels[start:start + 64].view("<u4"))
            products = np.fromfile(str(output) + ".products", dtype="<f4").reshape(4, frames)
            rows, cols = np.indices((128, 128))
            distance = (rows - 64) ** 2 + (cols - 64) ** 2
            masks = [distance <= 256, (distance >= 64) & (distance <= 256),
                     (distance >= 1024) & (distance <= 3969), np.ones((128, 128), dtype=bool)]
            for name, mask, measured in zip(("BF", "ABF", "ADF", "total"), masks, products):
                expected = np.empty(frames, dtype=np.float64)
                for start in range(0, frames, 64):
                    expected[start:start + 64] = pixels[start:start + 64, mask].sum(axis=1, dtype=np.float64)
                relative = np.abs(measured - expected) / np.maximum(np.abs(expected), 1)
                print(f"{name}: max_abs={np.max(np.abs(measured - expected)):.9g} max_rel={np.max(relative):.9g}")
                np.testing.assert_allclose(measured, expected, rtol=1e-6, atol=1e-6)
            print(f"FULL_EMPAD_PARITY samples={frames * 16384} detector=128x128 scan={size}x{size} PASS")
            total = pixels.sum(axis=(1, 2), dtype=np.float64)
            row_expected = np.empty(frames, dtype=np.float64)
            column_expected = np.empty(frames, dtype=np.float64)
            mean_expected = np.zeros((128, 128), dtype=np.float64)
            for start in range(0, frames, 64):
                chunk = pixels[start:start + 64].astype(np.float64)
                mean_expected += chunk.sum(axis=0) / frames
                row_expected[start:start + 64] = (chunk * rows).sum(axis=(1, 2)) / total[start:start + 64]
                column_expected[start:start + 64] = (chunk * cols).sum(axis=(1, 2)) / total[start:start + 64]
            for name, expected in [("mean", mean_expected.ravel()), ("com-row", row_expected),
                                   ("com-column", column_expected)]:
                measured = np.fromfile(str(output) + "." + name, dtype="<f4")
                np.testing.assert_allclose(measured, expected, rtol=1e-6, atol=1e-6)
                print(f"{name}: max_abs={np.max(np.abs(measured - expected)):.9g} PASS")


if __name__ == "__main__":
    unittest.main()
