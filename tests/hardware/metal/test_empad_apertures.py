"""Audit repeated aperture changes against original float64 sums, not prior images."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("EMPAD_PUBLIC_RAW") and os.environ.get("EMPAD_SOURCE_PARITY_EXE"),
                     "Set original EMPAD source and built native parity executable")
class EMPADApertureSequence(unittest.TestCase):
    def test_real_aperture_changes_against_original(self):
        raw = Path(os.environ["EMPAD_PUBLIC_RAW"])
        frames = raw.stat().st_size // (130 * 128 * 4)
        source = np.memmap(raw, dtype="<f4", mode="r", shape=(frames, 130, 128))[:, :128].reshape(frames, 16384)
        with tempfile.TemporaryDirectory(prefix="empad-apertures-") as directory:
            output = Path(directory) / "source.bin"
            result = subprocess.run([os.environ["EMPAD_SOURCE_PARITY_EXE"], str(raw), str(output), "0"],
                env=dict(os.environ, EMPAD_TEST_METAL="1", EMPAD_TEST_BUDGET="6000000000",
                         EMPAD_TEST_APERTURE_SEQUENCE="1"), text=True, capture_output=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("EMPAD_APERTURE_SEQUENCE steps=96", result.stdout)
            images = np.fromfile(str(output) + ".apertures", dtype="<f4").reshape(96, frames)
            masks = np.fromfile(str(output) + ".aperture-masks", dtype=np.uint8).reshape(96, 16384)
            selected = np.unique(np.linspace(0, frames - 1, 128, dtype=int))
            full_steps = {0, 1, 2, 3, 32, 64, 65, 95}
            for step, mask in enumerate(masks):
                indices = np.arange(frames) if step in full_steps else selected
                expected = np.empty(len(indices), dtype=np.float64)
                for start in range(0, len(indices), 64):
                    batch = indices[start:start + 64]
                    expected[start:start + len(batch)] = source[batch][:, mask != 0].sum(axis=1, dtype=np.float64)
                measured = images[step, indices]
                np.testing.assert_allclose(measured, expected, rtol=1e-6, atol=1e-6,
                                           err_msg=f"aperture step {step}; do not relax the reference")
            print(f"APERTURE_SEQUENCE_PASS steps=96 full_scan_maps={len(full_steps)} sampled_frames={len(selected)}", flush=True)


if __name__ == "__main__":
    unittest.main()
