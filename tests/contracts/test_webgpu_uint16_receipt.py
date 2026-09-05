"""Exact raw uint16 files remain eligible for browser resident receipts."""

import json
import struct
import zlib

import pytest

from tests.contracts.test_webgpu_compact_v3_parser import _parse, compact_bundle
from tests.contracts.io.test_compact_h5_uint16_builder import (
    test_uint16_builder_preserves_width_16_and_mask_applied_products as build_fixture,
)
from tests.contracts.test_webgpu_resident_contract import _node


@pytest.mark.parametrize("unaligned_tail_bytes", [0, 1, 2, 3])
def test_exact_uint16_source_is_admitted_without_mask_constants(
    compact_bundle, tmp_path, unaligned_tail_bytes
):
    build_fixture(tmp_path, unaligned_tail_bytes=unaligned_tail_bytes)
    source = tmp_path / "exact-uint16.h5"
    observed = _parse(compact_bundle, source)
    assert observed["ok"], observed
    assert observed["rawReconstructionAvailable"]
    result = _node(compact_bundle, """
import fs from 'node:fs';
const module = await import(process.argv[1]);
try {
  await module.loadCompactH5WebGPU(new File([fs.readFileSync(process.argv[2])], 'exact.h5'), {
    expectedReceipt: {}, device: { get limits() { throw new Error('device-used'); } },
  });
} catch (error) { console.log(JSON.stringify(String(error))); }
""", str(source))
    assert "exact raw resident receipt" not in result, result
    assert "schema" in result, result

    contents = bytearray(source.read_bytes())
    header_size = struct.unpack_from("<I", contents, 8)[0]
    manifest = json.loads(contents[24:24 + header_size])
    manifest["working_dtype"] = "uint8"
    encoded = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    struct.pack_into("<II", contents, 8, len(encoded), zlib.crc32(encoded))
    contents[24:24 + len(encoded)] = encoded
    legacy = tmp_path / "legacy-low8.h5"
    legacy.write_bytes(contents)
    rejected = _parse(compact_bundle, legacy)
    assert rejected["ok"], rejected
    assert not rejected["rawReconstructionAvailable"]
