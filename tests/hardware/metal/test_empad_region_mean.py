"""Check region means against the unchanged GPU path before and after .qem save.

Set EMPAD_REGION_MEAN_EXE to the built EMPADRegionMeanParity executable.
These small special-value fixtures supplement full real-acquisition checks.
"""

from array import array
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("EMPAD_REGION_MEAN_EXE"),
                     "Build EMPADRegionMeanParity first")
class EMPADRegionMeanTests(unittest.TestCase):
    def test_selected_means_preserve_gpu_results_across_qem_roundtrip(self):
        """Move circles and rectangles over signed, sparse and non-finite data."""
        with tempfile.TemporaryDirectory(prefix="empad-region-mean-") as directory:
            root = Path(directory)
            raw = root / "scan_x19_y17.raw"
            special = [0x80000000, 0xBF800000, 0x3E800000, 0x7FC01234,
                       0x7F800000, 0xFF800000, 0x00000001, 0x477FFF00]
            with raw.open("wb") as stream:
                for frame in range(17 * 19):
                    # Repeated low-bit words exercise entropy and constants;
                    # signed words and IEEE payloads retain their original bits.
                    words = array("I", (0x3F800000 | ((pixel + frame) % 31)
                                        for pixel in range(128 * 128)))
                    words[:len(special)] = array("I", special)
                    words[8] = 0x3F800001 if frame in (1, 62) else 0x3F800000
                    words[9] = 0xBF800000 if frame % 2 else 0x3F800000
                    stream.write(words.tobytes())
                    stream.write(array("I", [0xDEADBEEF] * 256).tobytes())
            for window, qem in (("64", False), ("512", True)):
                environment = dict(os.environ, EMPAD_MEAN_VERIFY="1",
                                   QGPU_EMPAD_WINDOW=window)
                if qem:
                    environment["EMPAD_MEAN_QEM"] = str(root / "roundtrip.qem")
                result = subprocess.run(
                    [os.environ["EMPAD_REGION_MEAN_EXE"], str(raw)],
                    env=environment, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout.count("EXACT_GPU_PARITY"), 14)
                if qem:
                    self.assertIn("QEM_REOPEN_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
