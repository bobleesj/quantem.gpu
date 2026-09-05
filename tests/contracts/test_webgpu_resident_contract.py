"""Cross-language source admission and calibrated resident identity."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest

from quantem.gpu.io.resident_contract import (
    ResidentGenerationReceipt,
    metadata_sha256,
)
from quantem.gpu.io import DataRepresentation
from tests.contracts.test_webgpu_compact_v3_parser import _write_v3
from tests.contracts.test_webgpu_compact_v3_parser import (
    compact_bundle as compact_bundle,  # noqa: PLC0414
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def receipt_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("webgpu-receipt") / "receipt.mjs"
    subprocess.run(
        [
            "npx",
            "--no-install",
            "esbuild",
            str(ROOT / "src/quantem/gpu/io/backends/webgpu/resident-contract.ts"),
            "--bundle",
            "--platform=browser",
            "--format=esm",
            f"--outfile={output}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def _node(bundle: Path, script: str, *arguments: str) -> object:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, bundle.as_uri(), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_metadata_hash_matches_python_for_calibration_and_unicode(receipt_bundle):
    metadata = [
        {"center": [97.25982403668188, 95.66573351654954], "radius": 41.0},
        {"units": "Å⁻¹", "sample": "三星", "𐀀": -0.0, "\ue000": 0},
        {"values": [True, None, 1, 1.0, -0.25, 1e-300, 1e300]},
    ]
    observed = _node(
        receipt_bundle,
        """
const module = await import(process.argv[1]);
console.log(JSON.stringify(JSON.parse(process.argv[2]).map(module.metadataSha256)));
""",
        json.dumps(metadata),
    )
    assert observed == [metadata_sha256(value) for value in metadata]


def _expected_receipt(source: Path) -> ResidentGenerationReceipt:
    contents = source.read_bytes()
    header_bytes = struct.unpack_from("<I", contents, 8)[0]
    manifest = json.loads(contents[24 : 24 + header_bytes])
    return ResidentGenerationReceipt(
        representation=DataRepresentation.LOSSLESS_PACKED,
        source_identity_sha256=manifest["source_identity_sha256"],
        source_shape=(1, 64, 2, 3),
        working_shape=(1, 64, 2, 3),
        source_dtype="uint16",
        working_dtype="uint8",
        source_logical_tensor_bytes=768,
        working_logical_tensor_bytes=384,
        physical_resident_bytes=224,
        container_bytes=len(contents),
        storage_schema=manifest["schema"],
        scan_bin=1,
        detector_bin=1,
        crop=None,
        detector_mask_count=1,
        detector_mask_sha256=manifest["detector_mask_sha256"],
        detector_mask_schema="quantem.gpu.detector-mask-identity/opaque-v1",
        calibration_schema=None,
        calibration_sha256=None,
        provenance_schema="quantem.gpu.packed-detector-h5-manifest/v1",
        provenance_sha256=metadata_sha256(manifest),
        source_raw_logical_sha256=manifest["source_raw_logical_sha256"],
        working_logical_sha256=manifest["prepared_uint8_sha256"],
        implementation_revision="a54007b+test-fixture",
    )


def test_trusted_receipt_admits_only_the_matching_source_before_device_use(
    compact_bundle: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "qualified.h5"
    seal = _write_v3(source)
    receipt = _expected_receipt(source).to_camel_case_dict()
    observed = _node(
        compact_bundle,
        """
import fs from 'node:fs';
const module = await import(process.argv[1]);
const source = new File([fs.readFileSync(process.argv[2])], 'qualified.h5');
const expected = JSON.parse(process.argv[3]);
const results = [];
for (const receipt of [expected,
  {...expected, provenanceSHA256: '0'.repeat(64)},
  {...expected, physicalResidentBytes: 1},
  {...expected, implementationRevision: 'different-build'}]) {
  try {
    await module.loadCompactH5WebGPU(source, {
      expectedReceipt: receipt, expectedWholeFileSha256: process.argv[4],
      implementationRevision: expected.implementationRevision,
      device: { get limits() { throw new Error('device-admission-reached'); } },
    });
  } catch (error) { results.push(String(error)); }
}
console.log(JSON.stringify(results));
""",
        str(source),
        json.dumps(receipt),
        seal,
    )
    assert "device-admission-reached" in observed[0]
    assert "provenanceSHA256" in observed[1]
    assert "physicalResidentBytes" in observed[2]
    assert "implementationRevision" in observed[3]


def test_unrecoverable_excluded_pixels_cannot_acquire_a_lossless_raw_receipt(
    compact_bundle: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "masked-only.h5"
    _write_v3(source, include_pixel_sha256=False, include_raw_values=False)
    observed = _node(
        compact_bundle,
        """
import fs from 'node:fs';
const module = await import(process.argv[1]);
try {
  await module.loadCompactH5WebGPU(new File([fs.readFileSync(process.argv[2])], 'masked.h5'), {
    expectedReceipt: {}, device: { get limits() { throw new Error('device-used'); } },
  });
} catch (error) { console.log(JSON.stringify(String(error))); }
""",
        str(source),
    )
    assert "exact raw resident receipt" in observed
    assert "device-used" not in observed
