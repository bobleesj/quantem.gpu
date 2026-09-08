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
        with raw.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.assertEqual(digest, "c431b6ae2a506b189ae86fbaf254d97de7c9258383c6dde96c7a871d1a9f7805")
        records = np.memmap(raw, dtype="<f4", mode="r", shape=(4096, 130, 128))
        pixels = records[:, :128, :]
        with tempfile.TemporaryDirectory(prefix="empad-public-parity-") as directory:
            output = Path(directory) / "resident.bin"
            result = subprocess.run(
                [os.environ["EMPAD_SOURCE_PARITY_EXE"], str(raw), str(output),
                 ",".join(map(str, range(4096))), "64", "64"],
                env=dict(os.environ, EMPAD_TEST_METAL="1", EMPAD_TEST_BUDGET="1073741824"),
                text=True, capture_output=True, timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            print(result.stdout, end="")
            actual = np.memmap(output, dtype="<u4", mode="r", shape=(4096, 128, 128))
            # Bit equality includes fractional/negative measurements and NaN payloads.
            for start in range(0, 4096, 64):
                np.testing.assert_array_equal(actual[start:start + 64], pixels[start:start + 64].view("<u4"))
            products = np.fromfile(str(output) + ".products", dtype="<f4").reshape(4, 4096)
            rows, cols = np.indices((128, 128))
            distance = (rows - 64) ** 2 + (cols - 64) ** 2
            masks = [distance <= 256, (distance >= 64) & (distance <= 256),
                     (distance >= 1024) & (distance <= 3969), np.ones((128, 128), dtype=bool)]
            for name, mask, measured in zip(("BF", "ABF", "ADF", "total"), masks, products):
                expected = np.empty(4096, dtype=np.float64)
                for start in range(0, 4096, 64):
                    expected[start:start + 64] = pixels[start:start + 64, mask].sum(axis=1, dtype=np.float64)
                relative = np.abs(measured - expected) / np.maximum(np.abs(expected), 1)
                print(f"{name}: max_abs={np.max(np.abs(measured - expected)):.9g} max_rel={np.max(relative):.9g}")
                np.testing.assert_allclose(measured, expected, rtol=1e-6, atol=1e-6)
            print("FULL_EMPAD_PARITY samples=67108864 detector=128x128 scan=64x64 PASS")


if __name__ == "__main__":
    unittest.main()
