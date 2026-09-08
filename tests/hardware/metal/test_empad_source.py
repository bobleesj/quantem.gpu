"""Compare the native EMPAD reader with independent original record bytes.

Run with EMPAD_SOURCE_PARITY_EXE pointing to the built Swift parity executable.
Fixtures are synthetic and do not substitute for public real-data qualification.
"""

import os
import math
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("EMPAD_SOURCE_PARITY_EXE"), "Build EMPADSourceParity first")
class EMPADSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="empad-source-parity-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = self.root / "scan_x3_y2.raw"
        self.shape = (2, 3)
        # Freeze special IEEE-754 words, including a NaN payload and subnormal.
        words = (0x80000000, 0xBF800000, 0x3E800000, 0x7FC01234,
                 0x7F800000, 0xFF800000, 0x00000001, 0x477FFF00)
        self.frames = []
        with self.raw.open("wb") as stream:
            for frame in range(6):
                pixels = bytearray(b"".join(
                    struct.pack("<f", ((pixel + frame) % 17 - 8) * 0.125)
                    for pixel in range(128 * 128)
                ))
                pixels[:32] = struct.pack("<8I", *words)
                pixels[-4:] = struct.pack("<f", frame + 0.125)
                self.frames.append(bytes(pixels))
                stream.write(pixels)
                stream.write(struct.pack("<I", 0xDEADBEEF) * 256)

    def read(self, source, *shape, succeeds=True, metal=False, budget=None,
             mutate=None, cancel_at=None):
        output = self.root / "selected.bin"
        indices = (5, 0, 3, 0) + ((64, 65) if len(self.frames) > 64 else ())
        environment = dict(os.environ, EMPAD_TEST_METAL="1" if metal else "0")
        if budget is not None:
            environment["EMPAD_TEST_BUDGET"] = str(budget)
        if mutate is not None:
            environment["EMPAD_TEST_MUTATE"] = mutate
        if cancel_at is not None:
            environment["EMPAD_TEST_CANCEL_AT"] = str(cancel_at)
        result = subprocess.run(
            [os.environ["EMPAD_SOURCE_PARITY_EXE"], str(source), str(output),
             ",".join(map(str, indices)), *map(str, shape)], capture_output=True, text=True,
            env=environment,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"scan={self.shape[0]}x{self.shape[1]}", result.stdout)
            self.assertEqual(output.read_bytes(), b"".join(self.frames[i] for i in indices))
            if metal:
                capabilities = json.loads(Path(str(output) + ".capabilities.json").read_text())
                receipt = capabilities["residentReceipt"]
                logical_hash = hashlib.sha256(b"".join(self.frames)).hexdigest()
                identity = hashlib.sha256(
                    b"quantem.gpu.empad-tensor/v1\0float32-le\0"
                    + struct.pack("<4Q", *self.shape, 128, 128)
                    + logical_hash.encode()
                ).hexdigest()
                self.assertEqual(receipt["sourceIdentitySHA256"], identity)
                self.assertEqual(receipt["workingLogicalSHA256"], logical_hash)
                self.assertEqual(receipt["sourceRawLogicalSHA256"], logical_hash)
                self.assertEqual(receipt["sourceDtype"], "float32")
                self.assertEqual(receipt["workingDtype"], "float32")
                self.assertEqual(receipt["representation"], "packed")
                self.assertEqual(receipt["workingLogicalTensorBytes"], len(self.frames) * 65536)
                self.assertEqual(capabilities["exactIntegerBits"], 0)
                products = {entry["product"]: entry for entry in capabilities["products"]}
                self.assertEqual(products["diffraction-pattern"]["numerics"], "exact-float32-bits")
                self.assertEqual(products["dpc"]["availability"], "unavailable")
            if cancel_at is not None:
                self.assertIn(f"EMPAD_CANCELLED checks={cancel_at}", result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            if mutate is not None:
                self.assertIn("changed during loading", result.stderr)

    def test_metal_packing_and_apertures_across_staging_windows(self):
        self.frames *= 11
        self.shape = (2, 33)
        self.raw = self.root / "scan_x33_y2.raw"
        with self.raw.open("wb") as stream:
            for frame in self.frames:
                stream.write(frame)
                stream.write(struct.pack("<I", 0xDEADBEEF) * 256)
        self.read(self.raw, metal=True)
        products = (self.root / "selected.bin.products").read_bytes()
        actual = struct.unpack(f"<{4 * len(self.frames)}f", products)
        for kind in range(4):
            for frame_index, frame in enumerate(self.frames):
                values = struct.unpack("<16384f", frame)
                if kind == 3:
                    self.assertTrue(math.isnan(actual[kind * len(self.frames) + frame_index]))
                    continue
                selected = []
                for pixel, value in enumerate(values):
                    distance = (pixel // 128 - 64) ** 2 + (pixel % 128 - 64) ** 2
                    bounds = ((0, 256), (64, 256), (1024, 3969))[kind]
                    if bounds[0] <= distance <= bounds[1]:
                        selected.append(value)
                # These dyadic fixture values sum exactly in float32. No tolerance
                # conceals a discrepancy between integer and float reductions.
                self.assertEqual(actual[kind * len(self.frames) + frame_index], math.fsum(selected))
        self.read(self.raw, metal=True, budget=1, succeeds=False)
        self.read(self.raw, metal=True, cancel_at=1)
        self.read(self.raw, metal=True, cancel_at=5)

    def test_source_changes_with_restored_mtime_are_rejected(self):
        self.read(self.raw, mutate="raw", succeeds=False)
        xml = self.root / "acquisition.xml"
        xml.write_text('<root><raw_file filename="scan_x3_y2.raw"/>'
                       '<pix_x>3</pix_x><pix_y>2</pix_y></root>')
        self.read(xml, mutate="xml", succeeds=False)

    def test_rectangular_raw_preserves_measurements_and_selection(self):
        self.read(self.raw)
        self.read(self.raw, 2, 3)
        self.read(self.raw, 3, 2, succeeds=False)

    def test_xml_acquisition_matches_raw_frame_bits(self):
        xml = self.root / "acquisition.xml"
        xml.write_text(
            '<root><raw_file filename="scope/path/scan_x3_y2.raw"/>'
            '<scan_parameters mode="search"><scan_resolution_x>1</scan_resolution_x>'
            '<scan_resolution_y>1</scan_resolution_y></scan_parameters>'
            '<scan_parameters mode="acquire"><scan_resolution_x>3</scan_resolution_x>'
            '<scan_resolution_y>2</scan_resolution_y></scan_parameters></root>'
        )
        self.read(xml)

    def test_xml_conflicts_and_unsafe_entities_are_rejected(self):
        xml = self.root / "acquisition.xml"
        prefix = '<root><raw_file filename="scan_x3_y2.raw"/>'
        invalid_fields = [
            '<pix_x>3</pix_x>',
            '<pix_x>unknown</pix_x><pix_y>2</pix_y>',
            '<pix_x>3</pix_x><pix_y>2</pix_y><pix_y>3</pix_y>',
            '<raw_file filename="another.raw"/><pix_x>3</pix_x><pix_y>2</pix_y>',
            '<pix_x>3</pix_x><pix_y>2</pix_y><type>series</type>',
            '<pix_x>3</pix_x><pix_y>2</pix_y>'
            '<scan_parameters mode="acquire"><scan_resolution_x>2</scan_resolution_x>'
            '<scan_resolution_y>3</scan_resolution_y></scan_parameters>',
        ]
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                xml.write_text(prefix + fields + '</root>')
                self.read(xml, succeeds=False)
        xml.write_text('<!DOCTYPE root [<!ENTITY x SYSTEM "file:///nonexistent">]>'
                       + prefix + '<pix_x>3</pix_x><pix_y>2</pix_y>&x;</root>')
        self.read(xml, succeeds=False)
        # A corrected legacy acquisition must still open after the rejected XML.
        xml.write_text('<root><raw_file filename="scan_x3_y2.raw"/>'
                       '<pix_x>3</pix_x><pix_y>2</pix_y></root>')
        self.read(xml)

    def test_incomplete_and_ambiguous_acquisitions_do_not_open(self):
        unnamed = self.root / "unlabelled.raw"
        self.raw.rename(unnamed)
        self.read(unnamed, succeeds=False)
        self.read(unnamed, 2, 3)
        with unnamed.open("ab") as stream:
            stream.write(b"extra")
        self.read(unnamed, 2, 3, succeeds=False)
        with unnamed.open("r+b") as stream:
            stream.truncate(100)
        self.read(unnamed, 2, 3, succeeds=False)


if __name__ == "__main__":
    unittest.main()
