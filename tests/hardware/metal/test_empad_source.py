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
from unittest.mock import patch


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
             mutate=None, cancel_at=None, hash_cache=None, cache_hit=None):
        output = self.root / "selected.bin"
        indices = (5, 0, 3, 0) + ((64, 65) if len(self.frames) > 64 else ())
        environment = dict(os.environ, EMPAD_TEST_METAL="1" if metal else "0")
        if budget is not None:
            environment["EMPAD_TEST_BUDGET"] = str(budget)
        if mutate is not None:
            environment["EMPAD_TEST_MUTATE"] = mutate
        if cancel_at is not None:
            environment["EMPAD_TEST_CANCEL_AT"] = str(cancel_at)
        if hash_cache is not None:
            environment["EMPAD_TEST_HASH_CACHE"] = str(hash_cache)
        result = subprocess.run(
            [os.environ["EMPAD_SOURCE_PARITY_EXE"], str(source), str(output),
             ",".join(map(str, indices)), *map(str, shape)], capture_output=True, text=True,
            env=environment,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
            if cache_hit is not None:
                self.assertIn(f"EMPAD_SOURCE_HASH cached={int(cache_hit)}", result.stdout)
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
                self.assertEqual(products["dpc"]["availability"], "resident-on-demand")
            if cancel_at is not None:
                self.assertIn(f"EMPAD_CANCELLED checks={cancel_at}", result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            if mutate is not None:
                self.assertIn("changed during loading", result.stderr)

    @patch.dict(os.environ, {"QGPU_EMPAD_WINDOW": "64"})
    def test_metal_packing_and_apertures_across_staging_windows(self):
        self.frames *= 11
        self.shape = (2, 33)
        self.raw = self.root / "scan_x33_y2.raw"
        with self.raw.open("wb") as stream:
            for frame in self.frames:
                stream.write(frame)
                stream.write(struct.pack("<I", 0xDEADBEEF) * 256)
        self.read(self.raw, metal=True)
        for name in ("com-row", "com-column"):
            coordinates = struct.unpack(f"<{len(self.frames)}f", (self.root / ("selected.bin." + name)).read_bytes())
            self.assertTrue(all(math.isnan(value) for value in coordinates))
        mean = struct.unpack("<16384f", (self.root / "selected.bin.mean").read_bytes())
        decoded = [struct.unpack("<16384f", frame) for frame in self.frames]
        for pixel, measured in enumerate(mean):
            values = [frame[pixel] for frame in decoded]
            expected = math.fsum(values) / len(values)
            if math.isnan(expected):
                self.assertTrue(math.isnan(measured))
            else:
                self.assertTrue(math.isclose(measured, expected, rel_tol=1e-6, abs_tol=1e-6))
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

    def test_every_packing_width_preserves_original_bits(self):
        # Force every width from 0 through 32, including cross-word fields,
        # signed zero and non-finite payloads. Packing must do no float math.
        self.frames = []
        with self.raw.open("wb") as stream:
            for frame in range(6):
                words = []
                for row in range(128):
                    width = row % 33
                    mask = (1 << width) - 1
                    base = 0x3F800000
                    words.extend(base ^ (0 if col == 0 else mask if col == 1 else
                                 ((col * 2654435761 + frame) & mask)) for col in range(128))
                pixels = struct.pack("<16384I", *words)
                self.frames.append(pixels)
                stream.write(pixels)
                stream.write(b"\xff" * 1024)
        self.read(self.raw, metal=True)

    def test_small_budget_reduces_staging_not_source_coverage(self):
        # Below Metal's fixed command/pipeline overhead must fail closed.
        self.read(self.raw, metal=True, budget=700000, succeeds=False)
        self.frames *= 11
        self.shape = (2, 33)
        self.raw = self.root / "scan_x33_y2.raw"
        with self.raw.open("wb") as stream:
            for frame in self.frames:
                stream.write(frame)
                stream.write(b"\xff" * 1024)
        # The complete acquisition fits, but its full staging pair does not.
        self.read(self.raw, metal=True, budget=6 * 1024 * 1024)

    def test_cancellation_before_final_publication(self):
        self.read(self.raw, metal=True, cancel_at=5)

    def test_hash_cache_reuses_only_unchanged_original_identity(self):
        cache = self.root / "source-hashes.json"
        original = self.raw.read_bytes()
        self.read(self.raw, metal=True, hash_cache=cache, cache_hit=False)
        self.read(self.raw, metal=True, hash_cache=cache, cache_hit=True)
        self.assertEqual(self.raw.read_bytes(), original)
        record = json.loads(cache.read_text())
        record["logicalSHA256"] = "0" * 64
        cache.write_text(json.dumps(record))
        self.read(self.raw, metal=True, hash_cache=cache, cache_hit=False)
        stamp = self.raw.stat()
        self.frames[0] = struct.pack("<f", 0.625) + self.frames[0][4:]
        with self.raw.open("r+b") as stream:
            stream.write(self.frames[0][:4])
        os.utime(self.raw, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.read(self.raw, metal=True, hash_cache=cache, cache_hit=False)
        self.read(self.raw, metal=True, hash_cache=cache, cache_hit=True)

    def test_hash_cache_cannot_overwrite_source(self):
        original = self.raw.read_bytes()
        self.read(self.raw, metal=True, hash_cache=self.raw, cache_hit=False)
        alias = self.root / "alias.json"
        alias.symlink_to(self.raw)
        self.read(self.raw, metal=True, hash_cache=alias, cache_hit=False)
        self.assertEqual(self.raw.read_bytes(), original)

    @patch.dict(os.environ, {"EMPAD_TEST_APERTURE_SEQUENCE": "1"})
    def test_aperture_sequence_recovers_after_removing_nonfinite_pixels(self):
        self.read(self.raw, metal=True)
        masks = (self.root / "selected.bin.aperture-masks").read_bytes()
        measured = struct.unpack("<576f", (self.root / "selected.bin.apertures").read_bytes())
        decoded = [struct.unpack("<16384f", frame) for frame in self.frames]
        for step in range(96):
            mask = masks[step * 16384:(step + 1) * 16384]
            for frame, values in enumerate(decoded):
                expected = sum(value for value, included in zip(values, mask) if included)
                actual = measured[step * 6 + frame]
                if math.isnan(expected):
                    self.assertTrue(math.isnan(actual))
                else:
                    self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6),
                                    (step, frame, actual, expected))

    def test_nearly_cancelled_apertures_keep_low_order_parts(self):
        annulus = [p for p in range(16384)
                   if 64 <= (p // 128 - 64) ** 2 + (p % 128 - 64) ** 2 <= 256]
        self.frames = []
        for frame in range(6):
            values = [0.0] * 16384
            # Separate large terms and small residuals across lanes/groups.
            for index in range(189):
                values[annulus[(index + frame * 37) % len(annulus)]] = (1e8, 0.125, -1e8)[index % 3]
            self.frames.append(struct.pack("<16384f", *values))
        with self.raw.open("wb") as stream:
            for frame in self.frames:
                stream.write(frame)
                stream.write(bytes(1024))
        self.read(self.raw, metal=True)
        actual = struct.unpack("<24f", (self.root / "selected.bin.products").read_bytes())
        for kind in range(4):
            expected = 0.0 if kind == 2 else 63 * 0.125
            for frame in range(6):
                self.assertTrue(math.isclose(actual[kind * 6 + frame], expected,
                                             rel_tol=1e-6, abs_tol=1e-6),
                                (kind, frame, actual[kind * 6 + frame], expected))

    def test_signed_and_zero_intensity_coordinates(self):
        values = [[0.0] * 16384, [-0.125] * 16384,
                  [float(pixel // 128) - 63.5 for pixel in range(16384)],
                  [0.0] * 16384, [0.0] * 16384, [0.125] * 16384]
        values[3][7 * 128 + 12] = 0.25
        values[4][3 * 128 + 9] = -0.25
        values[4][12 * 128 + 4] = 1.5
        self.frames = [struct.pack("<16384f", *frame) for frame in values]
        with self.raw.open("wb") as stream:
            for frame in self.frames:
                stream.write(frame)
                stream.write(bytes(1024))
        self.read(self.raw, metal=True)
        for name, coordinate in (("com-row", lambda p: p // 128), ("com-column", lambda p: p % 128)):
            measured = struct.unpack("<6f", (self.root / ("selected.bin." + name)).read_bytes())
            for frame, actual in zip(values, measured):
                total = math.fsum(frame)
                if total == 0:
                    self.assertTrue(math.isnan(actual))
                else:
                    expected = math.fsum(value * coordinate(pixel) for pixel, value in enumerate(frame)) / total
                    self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))

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

    def test_generation2_unpadded_records_and_sibling_xml(self):
        self.raw.write_bytes(b"".join(self.frames))
        xml = self.root / "acquisition.xml"
        xml.write_text('<root><scan><type>scan</type><shape>(2, 3)</shape>'
                       '<exposure_time>0.0001</exposure_time></scan>'
                       '<sensor><type>EMPAD2</type><shape>(128,128)</shape></sensor>'
                       '<rawfile><datatype>float32</datatype><filename>scan_x3_y2.raw</filename>'
                       '<framecount>10</framecount></rawfile></root>')
        # Acquisition framecount may include flyback; raster shape and exact
        # exported byte count govern the actual stored records.
        self.read(xml, metal=True)
        metadata = json.loads((self.root / 'selected.bin.metadata.json').read_text())
        self.assertEqual(metadata['formatIdentifier'], 'empad-g2-float32-xml/v1')
        self.assertEqual(metadata['recordBytes'], 65536)
        self.read(self.raw)
        xml.write_text(xml.read_text().replace('float32', 'uint32'))
        self.read(xml, succeeds=False)

    def test_emd_contiguous_float32_preserves_bits(self):
        import h5py
        import numpy as np
        path = self.root / 'acquisition.h5'
        with h5py.File(path, 'w') as f:
            f.attrs['authoring_program'] = 'emdfile'
            f.attrs['version_major'] = 1
            values = np.frombuffer(b''.join(self.frames), dtype='<f4').reshape(2, 3, 128, 128)
            f.create_dataset('datacube_root/datacube/data', data=values)
            cal = f.create_group('datacube_root/metadatabundle/calibration')
            cal['R_pixel_size'] = 0.5
            cal['R_pixel_units'] = 'A'
            cal['Q_pixel_size'] = 2.0
            cal['Q_pixel_units'] = 'mrad'
        self.read(path, metal=True)
        metadata = json.loads((self.root / 'selected.bin.metadata.json').read_text())
        self.assertEqual(metadata['formatIdentifier'], 'emd1-contiguous-float32/v1')
        self.assertEqual(metadata['scanRowAngstrom'], 0.5)
        self.assertIsNone(metadata['diffractionInverseNanometers'])
        with h5py.File(path, 'r+') as f:
            f.attrs['version_major'] = 2
        self.read(path, succeeds=False)

    def test_emd_incompatible_storage_and_external_links_are_rejected(self):
        import h5py
        import numpy as np
        path = self.root / 'acquisition.h5'
        for dtype, options in [('<f8', {}), ('>f4', {}), ('<f4', {'compression': 'gzip'})]:
            with h5py.File(path, 'w') as f:
                f.attrs['authoring_program'] = 'emdfile'
                f.attrs['version_major'] = 1
                f.create_dataset('datacube_root/datacube/data',
                                 data=np.zeros((2, 3, 128, 128), dtype=dtype), **options)
            self.read(path, succeeds=False)
        with h5py.File(self.root / 'external.h5', 'w') as f:
            f.create_dataset('data', data=np.zeros((2, 3, 128, 128), dtype='<f4'))
        with h5py.File(path, 'w') as f:
            f.attrs['authoring_program'] = 'emdfile'
            f.attrs['version_major'] = 1
            f['datacube_root/datacube/data'] = h5py.ExternalLink('external.h5', '/data')
        self.read(path, succeeds=False)

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

    def test_xml_sampling_preserves_rectangular_acquisition_units(self):
        xml = self.root / "acquisition.xml"
        prefix = '<root><raw_file filename="scan_x3_y2.raw"/>'
        # Modern fields describe the maximum dimension, not separate x/y FOVs.
        xml.write_text(prefix + '<timestamp isoformat="2026-01-02T03:04:05"/>'
                       '<iom_measurements><full_scan_field_of_view>'
                       '<x>2.16e-9</x><y>2.16e-9</y><scale_factor>0.72</scale_factor>'
                       '</full_scan_field_of_view></iom_measurements></root>')
        self.read(xml)
        metadata = json.loads((self.root / 'selected.bin.metadata.json').read_text())
        self.assertAlmostEqual(metadata['scanRowAngstrom'], 10)
        self.assertAlmostEqual(metadata['scanColumnAngstrom'], 10)
        self.assertIsNone(metadata['diffractionInverseNanometers'])
        self.assertEqual(metadata['acquisitionDate'], '2026-01-02T03:04:05')
        xml.write_text(prefix + '<iom_measurements>'
                       '<optics.get_full_scan_field_of_view>[2e-9, 6e-9]</optics.get_full_scan_field_of_view>'
                       '<calibrated_pixelsize>1.826537060227288e-10</calibrated_pixelsize>'
                       '</iom_measurements></root>')
        self.read(xml)
        metadata = json.loads((self.root / 'selected.bin.metadata.json').read_text())
        self.assertAlmostEqual(metadata['scanRowAngstrom'], 10)
        self.assertAlmostEqual(metadata['scanColumnAngstrom'], 20)
        self.assertAlmostEqual(metadata['diffractionInverseNanometers'], 0.1826537060227288)
        # Unknown or malformed calibration must not invent physical units.
        xml.write_text(prefix + '<iom_measurements><full_scan_field_of_view>'
                       '<x>2e-9</x><y>4e-9</y><scale_factor>0</scale_factor>'
                       '</full_scan_field_of_view><calibrated_pixelsize>nan</calibrated_pixelsize>'
                       '</iom_measurements></root>')
        self.read(xml)
        metadata = json.loads((self.root / 'selected.bin.metadata.json').read_text())
        self.assertIsNone(metadata['scanRowAngstrom'])
        self.assertIsNone(metadata['diffractionInverseNanometers'])

    def test_xml_without_dimensions_can_use_explicit_shape(self):
        unnamed = self.root / 'unlabelled.raw'
        self.raw.rename(unnamed)
        xml = self.root / 'acquisition.xml'
        xml.write_text('<root><raw_file filename="unlabelled.raw"/></root>')
        self.read(xml, succeeds=False)
        self.read(xml, 2, 3)

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
