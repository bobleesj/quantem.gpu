"""Tests for the backend-neutral resident-generation receipt."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quantem.gpu.io.resident_contract import (
    ResidentGenerationReceipt,
    ResidentStorageEncoding,
    metadata_sha256,
)


def _receipt(
    *,
    detector_shape: tuple[int, int] = (192, 192),
    physical_resident_bytes: int = 2_300_000_000,
) -> ResidentGenerationReceipt:
    shape = (512, 512, *detector_shape)
    return ResidentGenerationReceipt(
        representation="compact-qgix-v1-uint16",
        source_identity_sha256="a" * 64,
        source_shape=shape,
        working_shape=shape,
        source_dtype="uint16",
        working_dtype="uint16",
        source_logical_tensor_bytes=512
        * 512
        * detector_shape[0]
        * detector_shape[1]
        * 2,
        working_logical_tensor_bytes=512
        * 512
        * detector_shape[0]
        * detector_shape[1]
        * 2,
        physical_resident_bytes=physical_resident_bytes,
        container_bytes=physical_resident_bytes + 65_536,
        storage_encoding=ResidentStorageEncoding.LOSSLESS_PACKED,
        storage_schema="quantem.gpu.packed-detector-h5/v1",
        scan_bin=1,
        detector_bin=1,
        crop=None,
        detector_mask_count=4,
        detector_mask_sha256="b" * 64,
        detector_mask_schema="quantem.gpu.detector-mask-identity/opaque-v1",
        calibration_schema="quantem.gpu.detector-calibration/v1",
        calibration_sha256="c" * 64,
        provenance_schema="quantem.gpu.packed-detector-h5-manifest/v1",
        provenance_sha256="e" * 64,
        source_raw_logical_sha256="d" * 64,
        working_logical_sha256=None,
    )


@pytest.mark.parametrize("detector_shape", [(192, 192), (256, 256)])
def test_receipt_supports_detector_geometry_without_hardcoding(detector_shape) -> None:
    receipt = _receipt(detector_shape=detector_shape)

    receipt.validate()

    snake = receipt.to_snake_case_dict()
    camel = receipt.to_camel_case_dict()
    assert snake["schema"] == ResidentGenerationReceipt.SCHEMA
    assert snake["source_shape"] == [512, 512, *detector_shape]
    assert camel["sourceShape"] == snake["source_shape"]
    assert camel["sourceLogicalTensorBytes"] == snake["source_logical_tensor_bytes"]
    assert camel["workingLogicalTensorBytes"] == snake["working_logical_tensor_bytes"]
    assert camel["physicalResidentBytes"] == snake["physical_resident_bytes"]


def test_receipt_rejects_ambiguous_or_lossy_memory_claims() -> None:
    receipt = _receipt()

    with pytest.raises(ValueError, match="Source logical byte count"):
        replace(receipt, source_logical_tensor_bytes=1).validate()
    with pytest.raises(ValueError, match="Working logical byte count"):
        replace(receipt, working_logical_tensor_bytes=1).validate()
    with pytest.raises(ValueError, match="lossless exact"):
        replace(receipt, lossless_exact=False).validate()


def test_uint8_working_receipt_preserves_uint16_source_size() -> None:
    receipt = replace(
        _receipt(physical_resident_bytes=2_394_650_896),
        representation="compact-qgix-v3-uint8",
        working_dtype="uint8",
        working_logical_tensor_bytes=9_663_676_416,
        container_bytes=2_394_887_216,
        storage_schema="quantem.gpu.packed-detector-h5/v3",
        working_logical_sha256="f" * 64,
    )

    receipt.validate()
    assert receipt.source_logical_tensor_bytes == 19_327_352_832
    assert receipt.working_logical_tensor_bytes == 9_663_676_416
    assert receipt.physical_resident_bytes == 2_394_650_896


def test_dense_receipt_requires_physical_and_logical_bytes_to_match() -> None:
    receipt = _receipt()

    with pytest.raises(ValueError, match="Dense physical residency"):
        replace(
            receipt,
            storage_encoding=ResidentStorageEncoding.DENSE,
        ).validate()


def test_receipt_rejects_noncanonical_wire_values() -> None:
    receipt = _receipt()

    with pytest.raises(ValueError, match="canonical"):
        replace(receipt, source_dtype="<u2").validate()
    with pytest.raises(TypeError, match="storage encoding"):
        replace(receipt, storage_encoding="lossless-packed").validate()
    with pytest.raises(ValueError, match="positive integer"):
        replace(receipt, physical_resident_bytes=1.0).validate()
    with pytest.raises(ValueError, match="Implementation revision"):
        replace(receipt, implementation_revision="").validate()


def test_calibration_hash_uses_stable_numeric_encoding() -> None:
    first = {
        "schema": "quantem.gpu.detector-calibration/v1",
        "detector_center_px": [95, 96.0],
    }
    second = {
        "detector_center_px": [95.0, 96],
        "schema": "quantem.gpu.detector-calibration/v1",
    }

    assert metadata_sha256(first) == metadata_sha256(second)
