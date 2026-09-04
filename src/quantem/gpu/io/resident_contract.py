"""Backend-neutral exact resident-generation receipts.

The receipt separates the bytes of the scientific source, the bytes of the
logical working tensor, and the bytes physically retained by a backend.  It is
pure metadata: device admission, cache eviction, and user-interface policy
remain consumer responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "ResidentGenerationReceipt",
    "ResidentStorageEncoding",
    "metadata_sha256",
]


class ResidentStorageEncoding(str, Enum):
    """Physical encoding used for a complete exact resident generation."""

    DENSE = "dense"
    LOSSLESS_PACKED = "lossless-packed"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _optional_sha256(value: str | None, label: str) -> None:
    if value is not None and not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest or null.")


def _shape(value: tuple[int, ...], label: str) -> tuple[int, int, int, int]:
    if len(value) != 4 or any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(
            f"{label} must contain positive (scan row, scan column, detector row, "
            "detector column) dimensions."
        )
    return value


def _logical_bytes(shape: tuple[int, ...], dtype: str, label: str) -> int:
    try:
        parsed = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} dtype {dtype!r} is not supported.") from error
    if parsed.kind != "u" or parsed.itemsize not in (1, 2, 4, 8):
        raise ValueError(f"{label} dtype must be an unsigned integer; got {dtype!r}.")
    return math.prod(shape) * parsed.itemsize


def metadata_sha256(value: object) -> str:
    """Hash JSON-compatible metadata with deterministic numeric encoding."""

    def normalize(item: object) -> object:
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return item
        if isinstance(item, (int, float)):
            numeric = float(item)
            if not math.isfinite(numeric):
                raise ValueError("Receipt metadata numbers must be finite.")
            return f"f64be:{struct.pack('>d', numeric).hex()}"
        if isinstance(item, list | tuple):
            return [normalize(entry) for entry in item]
        if isinstance(item, dict):
            return {str(key): normalize(entry) for key, entry in item.items()}
        raise TypeError(
            "Receipt metadata may contain only JSON objects, arrays, strings, "
            "booleans, nulls, and finite numbers."
        )

    payload = json.dumps(
        normalize(value), separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ResidentGenerationReceipt:
    """Exact scientific and storage contract for one resident generation.

    ``source_logical_tensor_bytes`` describes the immutable source values.
    ``working_logical_tensor_bytes`` describes the exact logical tensor exposed
    to kernels after the declared plan. ``physical_resident_bytes`` describes
    the backend allocation or packed residency and therefore need not equal
    either logical byte count. ``detector_mask_sha256`` is an opaque,
    source-bound identity unless ``detector_mask_schema`` explicitly names a
    byte encoding.
    """

    representation: str
    source_identity_sha256: str
    source_shape: tuple[int, int, int, int]
    working_shape: tuple[int, int, int, int]
    source_dtype: str
    working_dtype: str
    source_logical_tensor_bytes: int
    working_logical_tensor_bytes: int
    physical_resident_bytes: int
    storage_encoding: ResidentStorageEncoding
    storage_schema: str
    scan_bin: int
    detector_bin: int
    crop: tuple[int, int, int, int] | None
    detector_mask_count: int
    detector_mask_sha256: str | None
    detector_mask_schema: str | None
    calibration_schema: str | None
    calibration_sha256: str | None
    provenance_schema: str | None
    provenance_sha256: str | None
    source_raw_logical_sha256: str | None = None
    working_logical_sha256: str | None = None
    container_bytes: int | None = None
    implementation_revision: str | None = None
    lossless_exact: bool = True

    SCHEMA = "quantem.gpu.4dstem-resident-receipt/v1"

    def validate(self) -> None:
        """Fail closed if the receipt can misstate scientific or memory state."""

        if not isinstance(self.representation, str) or not self.representation.strip():
            raise ValueError("Resident representation must be named.")
        if not _is_sha256(self.source_identity_sha256):
            raise ValueError("Source identity must be a lowercase SHA-256 digest.")
        source_shape = _shape(self.source_shape, "Source shape")
        working_shape = _shape(self.working_shape, "Working shape")
        expected_source_bytes = _logical_bytes(
            source_shape, self.source_dtype, "Source"
        )
        expected_working_bytes = _logical_bytes(
            working_shape, self.working_dtype, "Working"
        )
        if self.source_dtype not in {"uint8", "uint16", "uint32", "uint64"}:
            raise ValueError("Source dtype must use a canonical unsigned-integer name.")
        if self.working_dtype not in {"uint8", "uint16", "uint32", "uint64"}:
            raise ValueError("Working dtype must use a canonical unsigned-integer name.")
        for label, value in (
            ("Source logical byte count", self.source_logical_tensor_bytes),
            ("Working logical byte count", self.working_logical_tensor_bytes),
            ("Physical resident byte count", self.physical_resident_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer.")
        if self.source_logical_tensor_bytes != expected_source_bytes:
            raise ValueError(
                "Source logical byte count does not match source shape and dtype."
            )
        if self.working_logical_tensor_bytes != expected_working_bytes:
            raise ValueError(
                "Working logical byte count does not match working shape and dtype."
            )
        if self.physical_resident_bytes <= 0:
            raise ValueError("Physical resident byte count must be positive.")
        if self.container_bytes is not None and (
            type(self.container_bytes) is not int or self.container_bytes <= 0
        ):
            raise ValueError("Container byte count must be positive or null.")
        if not isinstance(self.storage_encoding, ResidentStorageEncoding):
            raise TypeError("Resident storage encoding is unsupported.")
        if type(self.scan_bin) is not int or self.scan_bin <= 0:
            raise ValueError("Scan bin must be a positive integer.")
        if type(self.detector_bin) is not int or self.detector_bin <= 0:
            raise ValueError("Detector bin must be a positive integer.")
        if self.crop is None:
            crop_rows = source_shape[0]
            crop_columns = source_shape[1]
        else:
            if (
                len(self.crop) != 4
                or any(type(item) is not int for item in self.crop)
                or not 0 <= self.crop[0] < self.crop[1] <= source_shape[0]
                or not 0 <= self.crop[2] < self.crop[3] <= source_shape[1]
            ):
                raise ValueError(
                    "Crop must be null or a valid half-open "
                    "(row start, row stop, column start, column stop) region."
                )
            crop_rows = self.crop[1] - self.crop[0]
            crop_columns = self.crop[3] - self.crop[2]
        expected_shape = (
            (crop_rows + self.scan_bin - 1) // self.scan_bin,
            (crop_columns + self.scan_bin - 1) // self.scan_bin,
            (source_shape[2] + self.detector_bin - 1) // self.detector_bin,
            (source_shape[3] + self.detector_bin - 1) // self.detector_bin,
        )
        if working_shape != expected_shape:
            raise ValueError(
                "Working shape does not match the declared exact bin plan."
            )
        if type(self.detector_mask_count) is not int or self.detector_mask_count < 0:
            raise ValueError("Detector mask count must be a nonnegative integer.")
        if self.detector_mask_count and not _is_sha256(self.detector_mask_sha256):
            raise ValueError(
                "A nonempty detector mask requires a lowercase SHA-256 identity."
            )
        _optional_sha256(self.detector_mask_sha256, "Detector mask identity")
        if (self.detector_mask_schema is None) != (self.detector_mask_sha256 is None):
            raise ValueError(
                "Detector-mask schema and SHA-256 identity must both be present or null."
            )
        if (
            self.detector_mask_schema is not None
            and not self.detector_mask_schema.strip()
        ):
            raise ValueError("Detector-mask schema must be nonempty when present.")
        if (self.calibration_schema is None) != (self.calibration_sha256 is None):
            raise ValueError(
                "Calibration schema and SHA-256 identity must both be present or null."
            )
        if self.calibration_schema is not None and not self.calibration_schema.strip():
            raise ValueError("Calibration schema must be nonempty when present.")
        _optional_sha256(self.calibration_sha256, "Calibration identity")
        if (self.provenance_schema is None) != (self.provenance_sha256 is None):
            raise ValueError(
                "Provenance schema and SHA-256 identity must both be present or null."
            )
        if self.provenance_schema is not None and not self.provenance_schema.strip():
            raise ValueError("Provenance schema must be nonempty when present.")
        _optional_sha256(self.provenance_sha256, "Provenance identity")
        _optional_sha256(self.source_raw_logical_sha256, "Raw source identity")
        _optional_sha256(self.working_logical_sha256, "Working logical identity")
        if not self.storage_schema.strip():
            raise ValueError("Resident storage schema must be versioned and nonempty.")
        if self.implementation_revision is not None and (
            not isinstance(self.implementation_revision, str)
            or not self.implementation_revision.strip()
        ):
            raise ValueError("Implementation revision must be nonempty or null.")
        if not self.lossless_exact:
            raise ValueError(
                "Resident generation receipts require lossless exact storage."
            )
        if (
            self.storage_encoding is ResidentStorageEncoding.DENSE
            and self.physical_resident_bytes != self.working_logical_tensor_bytes
        ):
            raise ValueError(
                "Dense physical residency must equal the working logical byte count."
            )

    def to_snake_case_dict(self) -> dict[str, Any]:
        """Return the canonical Python/service wire spelling."""

        self.validate()
        return {
            "schema": self.SCHEMA,
            "representation": self.representation,
            "source_identity_sha256": self.source_identity_sha256,
            "source_shape": list(self.source_shape),
            "working_shape": list(self.working_shape),
            "source_dtype": self.source_dtype,
            "working_dtype": self.working_dtype,
            "source_logical_tensor_bytes": self.source_logical_tensor_bytes,
            "working_logical_tensor_bytes": self.working_logical_tensor_bytes,
            "physical_resident_bytes": self.physical_resident_bytes,
            "container_bytes": self.container_bytes,
            "storage_encoding": self.storage_encoding.value,
            "storage_schema": self.storage_schema,
            "lossless_exact": self.lossless_exact,
            "scan_bin": self.scan_bin,
            "detector_bin": self.detector_bin,
            "crop": list(self.crop) if self.crop is not None else None,
            "detector_mask_count": self.detector_mask_count,
            "detector_mask_sha256": self.detector_mask_sha256,
            "detector_mask_schema": self.detector_mask_schema,
            "calibration_schema": self.calibration_schema,
            "calibration_sha256": self.calibration_sha256,
            "provenance_schema": self.provenance_schema,
            "provenance_sha256": self.provenance_sha256,
            "source_raw_logical_sha256": self.source_raw_logical_sha256,
            "working_logical_sha256": self.working_logical_sha256,
            "implementation_revision": self.implementation_revision,
        }

    def to_camel_case_dict(self) -> dict[str, Any]:
        """Return the canonical Swift/TypeScript wire spelling."""

        value = self.to_snake_case_dict()
        aliases = {
            "source_identity_sha256": "sourceIdentitySHA256",
            "source_shape": "sourceShape",
            "working_shape": "workingShape",
            "source_dtype": "sourceDtype",
            "working_dtype": "workingDtype",
            "source_logical_tensor_bytes": "sourceLogicalTensorBytes",
            "working_logical_tensor_bytes": "workingLogicalTensorBytes",
            "physical_resident_bytes": "physicalResidentBytes",
            "container_bytes": "containerBytes",
            "storage_encoding": "storageEncoding",
            "storage_schema": "storageSchema",
            "lossless_exact": "losslessExact",
            "scan_bin": "scanBin",
            "detector_bin": "detectorBin",
            "detector_mask_count": "detectorMaskCount",
            "detector_mask_sha256": "detectorMaskSHA256",
            "detector_mask_schema": "detectorMaskSchema",
            "calibration_schema": "calibrationSchema",
            "calibration_sha256": "calibrationSHA256",
            "provenance_schema": "provenanceSchema",
            "provenance_sha256": "provenanceSHA256",
            "source_raw_logical_sha256": "sourceRawLogicalSHA256",
            "working_logical_sha256": "workingLogicalSHA256",
            "implementation_revision": "implementationRevision",
        }
        return {aliases.get(key, key): item for key, item in value.items()}
