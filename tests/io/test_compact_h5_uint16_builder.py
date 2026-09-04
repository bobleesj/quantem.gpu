"""Exact uint16 producer tests for the compact QGIX v1 path."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.io._compact_h5 import (
    CompactH5Index,
    CompactH5ReferenceDecoder,
    prepare_compact_h5_metadata_copy,
)


def _packed_detector_major(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scan_count, detector_pixels = values.shape
    assert scan_count == 128
    descriptors = np.empty(detector_pixels, dtype="<u4")
    words: list[int] = []
    for pixel in range(detector_pixels):
        samples = values[:, pixel].astype(np.uint64)
        width = int(samples.max(initial=0)).bit_length()
        descriptors[pixel] = np.uint32((len(words) << 5) | width)
        accumulator = 0
        bits = 0
        for sample in samples:
            accumulator |= int(sample) << bits
            bits += width
            while bits >= 32:
                words.append(accumulator & 0xFFFFFFFF)
                accumulator >>= 32
                bits -= 32
        assert bits == 0
    return descriptors, np.asarray(words, dtype="<u4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_uint16_builder_preserves_width_16_and_mask_applied_products(
    tmp_path: Path,
) -> None:
    values = np.zeros((128, 2), dtype="<u2")
    values[:, 0] = np.arange(128, dtype=np.uint16) * 509
    values[0, 0] = np.uint16(65535)
    values[:, 1] = np.uint16(65535)
    descriptors, payload = _packed_detector_major(values)
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    descriptor_path = packed_dir / "descriptors.bin"
    payload_path = packed_dir / "payload.bin"
    descriptors.tofile(descriptor_path)
    payload.tofile(payload_path)
    source_identity = hashlib.sha256(b"source identity").hexdigest()
    raw_sha256 = hashlib.sha256(values.tobytes()).hexdigest()
    packed_manifest = packed_dir / "manifest.json"
    packed_manifest.write_text(
        json.dumps(
            {
                "raw_logical_sha256": raw_sha256,
                "shards": [
                    {
                        "ordinal": 0,
                        "scan_count": 128,
                        "descriptors": {
                            "path": descriptor_path.name,
                            "sha256": _sha256(descriptor_path),
                        },
                        "payload": {
                            "path": payload_path.name,
                            "sha256": _sha256(payload_path),
                        },
                    }
                ],
            }
        )
    )
    source_contract = tmp_path / "source-contract.json"
    source_contract.write_text(
        json.dumps(
            {
                "source_identity_sha256": source_identity,
                "source_raw_logical_sha256": raw_sha256,
                "source_shape": [8, 16, 1, 2],
                "source_dtype": "uint16",
                "detector_mask_sha256": hashlib.sha256(b"mask").hexdigest(),
                "masked_detector_pixels": [[0, 1]],
                "detector_calibration": {
                    "schema": "quantem.gpu.detector-calibration/v1",
                    "source_identity_sha256": source_identity,
                    "detector_center_px": [0.0, 0.5],
                    "bright_field_radius_px": 0.5,
                    "dpc_rotation_degrees": 0.0,
                    "dpc_component_order_exchanged": False,
                    "method": "test fixture",
                },
            }
        )
    )
    output = tmp_path / "exact-uint16.h5"
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts" / "build_compact_h5_uint16.py")
    )
    receipt = namespace["build_compact_h5_uint16"](
        packed_manifest,
        source_contract,
        output,
    )

    index = CompactH5Index.from_file(output)
    decoder = CompactH5ReferenceDecoder(index)
    decoder.validate_shard_metadata(0)
    assert receipt["working_dtype"] == "uint16"
    assert receipt["maximum_width"] == 16
    assert index.schema_version == 1
    assert index.manifest["masked_detector_payload_policy"] == (
        "retained_exactly_in_payload"
    )
    assert index.raw_reconstruction_available
    for scan in (0, 1, 63, 127):
        row, column = divmod(scan, 16)
        assert decoder.value(row, column, 0, 0) == int(values[scan, 0])
        assert decoder.value(row, column, 0, 1) == 0
        assert decoder.raw_value(row, column, 0, 1) == int(values[scan, 1])

    validator = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts" / "validate_compact_h5_v1_full.py")
    )
    full_validation = validator["validate_compact_h5_v1_full"](
        output,
        expected_whole_file_sha256=receipt["whole_file_sha256"],
    )
    assert full_validation["status"] == "pass"
    assert full_validation["shard_count"] == 1
    assert full_validation["decoded_sha256_checks"] == 1
    with pytest.raises(ValueError, match="whole-file SHA-256"):
        validator["validate_compact_h5_v1_full"](
            output,
            expected_whole_file_sha256="0" * 64,
        )

    original = output.read_bytes()
    working = values.copy()
    working[:, 1] = 0
    total = working.sum(axis=1, dtype=np.uint64)
    row_moment = np.zeros_like(total)
    column_moment = working[:, 1].astype(np.uint64)
    prepared_output = tmp_path / "exact-uint16-prepared.h5"
    prepared = prepare_compact_h5_metadata_copy(
        output,
        prepared_output,
        expected_source_sha256=hashlib.sha256(original).hexdigest(),
        working_logical_sha256=hashlib.sha256(working.tobytes()).hexdigest(),
        masked_detector_raw_values=(65535,),
        prepared_dpc_moments=(total, row_moment, column_moment),
    )
    assert output.read_bytes() == original
    assert prepared_output.stat().st_size > output.stat().st_size
    assert (
        prepared.manifest["working_logical_sha256"]
        == hashlib.sha256(working.tobytes()).hexdigest()
    )
    assert prepared.masked_detector_raw_values == (65535,)
    assert prepared.prepared_dpc_moments is not None
    prepared_values = CompactH5ReferenceDecoder(prepared).prepared_dpc_moment_values()
    assert prepared_values is not None
    assert all(
        np.array_equal(observed, expected)
        for observed, expected in zip(
            prepared_values,
            (total, row_moment, column_moment),
            strict=True,
        )
    )

    changed = bytearray(prepared_output.read_bytes())
    changed[prepared.prepared_dpc_moments.file_offset] ^= 1
    corrupt_output = tmp_path / "exact-uint16-prepared-corrupt.h5"
    corrupt_output.write_bytes(changed)
    corrupt = CompactH5ReferenceDecoder(CompactH5Index.from_file(corrupt_output))
    with pytest.raises(ValueError, match="Prepared DPC SHA-256"):
        corrupt.prepared_dpc_moment_values()

    missing_identity_output = tmp_path / "missing-working-identity.h5"
    with pytest.raises(ValueError, match="require working_logical_sha256"):
        prepare_compact_h5_metadata_copy(
            output,
            missing_identity_output,
            masked_detector_raw_values=(65535,),
            prepared_dpc_moments=(total, row_moment, column_moment),
        )
    assert not missing_identity_output.exists()

    wrong_source_output = tmp_path / "wrong-source-identity.h5"
    with pytest.raises(ValueError, match="Compact input SHA-256"):
        prepare_compact_h5_metadata_copy(
            output,
            wrong_source_output,
            expected_source_sha256="0" * 64,
        )
    assert not wrong_source_output.exists()

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        namespace["build_compact_h5_uint16"](
            packed_manifest,
            source_contract,
            output,
        )

    mismatched_contract = tmp_path / "mismatched-source-contract.json"
    contract_value = json.loads(source_contract.read_text())
    contract_value["source_raw_logical_sha256"] = "0" * 64
    mismatched_contract.write_text(json.dumps(contract_value))
    mismatched_output = tmp_path / "mismatched-source.h5"
    with pytest.raises(ValueError, match="raw logical hashes disagree"):
        namespace["build_compact_h5_uint16"](
            packed_manifest,
            mismatched_contract,
            mismatched_output,
        )
    assert not mismatched_output.exists()

    invalid_descriptors = descriptors.copy()
    invalid_descriptors[0] = (
        invalid_descriptors[0] & np.uint32(0xFFFFFFE0)
    ) | np.uint32(17)
    invalid_descriptors.tofile(descriptor_path)
    packed_value = json.loads(packed_manifest.read_text())
    packed_value["shards"][0]["descriptors"]["sha256"] = _sha256(descriptor_path)
    packed_manifest.write_text(json.dumps(packed_value))
    width_output = tmp_path / "width-17.h5"
    with pytest.raises(ValueError, match="width 17"):
        namespace["build_compact_h5_uint16"](
            packed_manifest,
            source_contract,
            width_output,
        )
    assert not width_output.exists()

    invalid_descriptors = descriptors.copy()
    invalid_descriptors[1] += np.uint32(1 << 5)
    invalid_descriptors.tofile(descriptor_path)
    packed_value["shards"][0]["descriptors"]["sha256"] = _sha256(descriptor_path)
    packed_manifest.write_text(json.dumps(packed_value))
    offset_output = tmp_path / "broken-offset.h5"
    with pytest.raises(ValueError, match="offsets do not exactly cover"):
        namespace["build_compact_h5_uint16"](
            packed_manifest,
            source_contract,
            offset_output,
        )
    assert not offset_output.exists()
