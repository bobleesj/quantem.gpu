"""Exact bounded reads of explicitly shaped HDF5 float diffraction stacks.

These tiny reader tests do not qualify a resident codec or scientific product.
Set EMPAD_SOURCE_PARITY_EXE to the native EMPADSourceParity executable.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import h5py
import numpy as np


@unittest.skipUnless(os.environ.get("EMPAD_SOURCE_PARITY_EXE"), "Build EMPADSourceParity first")
class NamedFloatSourceTests(unittest.TestCase):
    """Require explicit scan axes and preserve IEEE words, not decimal values."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="named-float-parity-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "simulation.hdf5"
        self.output = self.root / "selected.bin"
        self.bits = np.arange(6 * 128 * 128, dtype="<u4").reshape(6, 128, 128)
        self.bits[:, 0, :6] = [0x80000000, 0xBF800000, 0x7FC01234,
                               0x7F800000, 0xFF800000, 0x00000001]

    def make(self, *, shape=(2, 3), dtype="<f4", compression=None, abtem=False):
        with h5py.File(self.source, "w") as handle:
            data = self.bits.view("<f4") if dtype == "<f4" else np.zeros(self.bits.shape, dtype)
            dp = handle.create_dataset("dp", data=data, compression=compression)
            if shape is not None:
                if abtem:
                    handle["abtem_params/N_scan_slow"] = shape[0]
                    handle["abtem_params/N_scan_fast"] = shape[1]
                else:
                    dp.attrs["scan_shape"] = shape
            # A dose variant must not be silently substituted for /dp.
            handle["dp_1e+04"] = np.ones((6, 128, 128), dtype="<f4")

    def invoke(self, succeeds):
        result = subprocess.run(
            [os.environ["EMPAD_SOURCE_PARITY_EXE"], str(self.source),
             str(self.output), "5,0,3,0"],
            env={**os.environ, "EMPAD_TEST_METAL": "0"},
            capture_output=True, text=True, check=False,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.output.read_bytes(), self.bits[[5, 0, 3, 0]].tobytes())
            metadata = json.loads(Path(str(self.output) + ".metadata.json").read_text())
            self.assertEqual(metadata["formatIdentifier"], "hdf5-contiguous-float32/v1")
            self.assertEqual(metadata["microscope"]["sourceDataset"], "/dp")
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("explicit scan_shape", result.stderr)
            self.assertFalse(self.output.exists())

    def test_explicit_scan_axes_and_duplicate_frame_reads(self):
        self.make()
        self.invoke(True)

    def test_recorded_abtem_scan_axes(self):
        self.make(abtem=True)
        self.invoke(True)

    def test_missing_or_inconsistent_shape_is_not_guessed(self):
        for shape in (None, (3, 3), (0, 6), (-2, -3), (2.0, 3.0)):
            with self.subTest(shape=shape):
                self.make(shape=shape)
                self.invoke(False)

    def test_unsupported_precision_and_compression_are_not_cast(self):
        for dtype, compression in (("<f8", None), ("<f4", "gzip")):
            with self.subTest(dtype=dtype, compression=compression):
                self.make(dtype=dtype, compression=compression)
                self.invoke(False)

    def test_external_payload_is_not_followed(self):
        self.make()
        target = self.source.rename(self.root / "external.hdf5")
        with h5py.File(self.source, "w") as handle:
            handle["dp"] = h5py.ExternalLink(target.name, "/dp")
        self.invoke(False)


if __name__ == "__main__":
    unittest.main()
