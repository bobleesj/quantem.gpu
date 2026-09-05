"""Accepted isolated Python-hosted Metal backend for frozen QGIX v3 files.

The backend keeps each authenticated direct bit-packed shard resident in shared
Apple unified-memory buffers and dispatches Metal kernels without constructing
the dense four-dimensional volume. Its schema-specific API remains an explicit
``mps.compact_v3`` import while scientist-facing source selection stays with the
canonical QuantEM I/O loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
import time
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MPSCompactV3Cancelled",
    "MPSCompactV3DetectorMetrics",
    "MPSCompactV3Error",
    "MPSCompactV3ExactDPCMoments",
    "MPSCompactV3Index",
    "MPSCompactV3LoadMetrics",
    "MPSCompactV3LogicalHashMetrics",
    "MPSCompactV3MeanDiffraction",
    "MPSCompactV3PreparedDPCMoments",
    "MPSCompactV3PreparedDetectorProduct",
    "MPSCompactV3PreparedDetectorProducts",
    "MPSCompactV3Resident",
    "MPSCompactV3Shard",
    "load_compact_v3_mps",
    "read_compact_v3_index",
]

_CONTAINER_MAGIC = b"QGPUH5\0\1"
_INDEX_MAGIC = b"QGIX\0\0\0\3"
_PRELUDE = struct.Struct("<8sIIII")
_INDEX_HEADER = struct.Struct("<8sIIIIIIIII")
_SHARD_RECORD = struct.Struct("<QQQQQQQII32s")
_SHA256_HEX_LENGTH = 64
_CALIBRATION_DIGEST_ENCODING = "canonical-json-numbers-as-f64be-hex/v1"
_pipeline_cache: dict[int, tuple[Any, Any, Any, Any, Any, Any, float]] = {}


class MPSCompactV3Error(ValueError):
    """The QGIX v3 file or requested operation is not exact and admissible."""


class MPSCompactV3Cancelled(RuntimeError):
    """A caller generation superseded the load before resident publication."""


@dataclass(frozen=True)
class MPSCompactV3Shard:
    """One directly addressable packed shard."""

    payload_offset: int
    payload_bytes: int
    headers_offset: int
    headers_bytes: int
    payload_sha256: str
    header_words: int


@dataclass(frozen=True)
class MPSCompactV3PreparedDPCMoments:
    """Source-bound exact detector moments stored in one contiguous range."""

    file_offset: int
    file_bytes: int
    sha256: str
    working_uint8_sha256: str
    detector_mask_sha256: str
    scan_count: int
    selected_detector_pixels: int
    detector_columns: int
    total_bound: int
    row_moment_bound: int
    column_moment_bound: int
    narrow_integer: bool
    narrow_products: bool


@dataclass(frozen=True)
class MPSCompactV3PreparedDetectorProduct:
    """One exact source-bound canonical virtual-detector product."""

    name: str
    center_px: tuple[float, float]
    inner_radius_px: float
    outer_radius_px: float
    selected_detector_pixels: int
    mask_file_offset: int
    mask_file_bytes: int
    mask_sha256: str
    values_file_offset: int
    values_file_bytes: int
    values_sha256: str


@dataclass(frozen=True)
class MPSCompactV3PreparedDetectorProducts:
    """Authenticated BF, ABF, and ADF maps embedded in one compact source."""

    calibration_sha256: str
    working_uint8_sha256: str
    detector_mask_sha256: str
    products: tuple[MPSCompactV3PreparedDetectorProduct, ...]


@dataclass(frozen=True)
class MPSCompactV3Index:
    """Strict metadata for one accepted QGIX v3 file."""

    path: Path
    file_bytes: int
    shape: tuple[int, int, int, int]
    scans_per_shard: int
    scan_tile: int
    source_identity_sha256: str
    source_raw_logical_sha256: str | None
    working_logical_sha256: str
    detector_mask_sha256: str | None
    masked_detector_pixels_sha256: str | None
    masked_detector_raw_values: tuple[int, ...] | None
    raw_access_mode: str
    excluded_detector_pixels: tuple[int, ...]
    embedded_scientific_semantics: bool
    header_words_per_pixel: int
    prepared_dpc_moments: MPSCompactV3PreparedDPCMoments | None
    prepared_detector_products: MPSCompactV3PreparedDetectorProducts | None
    shards: tuple[MPSCompactV3Shard, ...]
    manifest: dict[str, Any]

    @property
    def scan_count(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def detector_pixels(self) -> int:
        return self.shape[2] * self.shape[3]

    @property
    def tile_count(self) -> int:
        return self.scans_per_shard // self.scan_tile

    @property
    def packed_resident_bytes(self) -> int:
        return sum(shard.payload_bytes + shard.headers_bytes for shard in self.shards)


@dataclass(frozen=True)
class MPSCompactV3LoadMetrics:
    """Measured Python MPS load phases."""

    metadata_ms: float
    pipeline_compile_ms: float
    authentication_ms: float
    source_read_ms: float
    gpu_validation_ms: float
    prepared_dpc_read_ms: float
    prepared_dpc_authentication_ms: float
    prepared_dpc_prime_ms: float
    prepared_detector_product_read_ms: float
    prepared_detector_product_authentication_ms: float
    resident_ready_ms: float
    authentication_policy: str
    mapped_authentication_bytes: int
    prepared_dpc_bytes: int
    prepared_detector_product_bytes: int
    packed_resident_bytes: int
    product_resident_bytes: int
    total_resident_bytes: int
    maximum_transient_bytes: int
    device_allocated_bytes_before: int
    device_allocated_bytes_after: int


@dataclass(frozen=True)
class MPSCompactV3DetectorMetrics:
    """One exact resident detector-mask update."""

    mode: str
    changed_detector_pixels: int
    wall_ms: float
    gpu_ms: float
    framework_overhead_ms: float
    fft_dispatch_count: int = 0


@dataclass(frozen=True)
class MPSCompactV3LogicalHashMetrics:
    """One bounded full logical working-value audit."""

    sha256: str
    logical_bytes: int
    staging_bytes: int
    wall_ms: float
    gpu_ms: float


@dataclass(frozen=True)
class MPSCompactV3MeanDiffraction:
    """Exact detector sum and float32 mean diffraction display map."""

    detector_sum: np.ndarray
    mean: np.ndarray
    wall_ms: float
    gpu_ms: float
    dispatch_count: int
    readback_bytes: int


@dataclass(frozen=True)
class MPSCompactV3ExactDPCMoments:
    """Exact total and detector-coordinate moments for every scan."""

    total: np.ndarray
    detector_row_moment: np.ndarray
    detector_column_moment: np.ndarray


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise MPSCompactV3Error(f"{label} is not a lowercase SHA-256 digest")
    return text


def _calibration_digest(calibration: dict[str, Any]) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise MPSCompactV3Error("detector calibration numbers must be finite")
            return f"f64be:{struct.pack('>d', numeric).hex()}"
        if isinstance(value, list):
            return [normalize(entry) for entry in value]
        if isinstance(value, dict):
            return {str(key): normalize(entry) for key, entry in value.items()}
        raise MPSCompactV3Error(
            "detector calibration contains an unauthenticated JSON value"
        )

    payload = json.dumps(
        normalize(calibration), separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_prepared_dpc_moments(
    path: Path,
    file_bytes: int,
    manifest: dict[str, Any],
    shape: tuple[int, int, int, int],
    source_identity: str,
    excluded: tuple[int, ...],
    ranges: list[tuple[int, int, str]],
) -> MPSCompactV3PreparedDPCMoments | None:
    prepared = manifest.get("prepared_dpc_moments")
    if prepared is None:
        return None
    if not isinstance(prepared, dict):
        raise MPSCompactV3Error("compact prepared DPC moments are not an object")
    scan_count = shape[0] * shape[1]
    detector_pixels = shape[2] * shape[3]
    selected = detector_pixels - len(excluded)
    excluded_set = set(excluded)
    total_bound = selected * 255
    row_moment_bound = 255 * sum(
        pixel // shape[3]
        for pixel in range(detector_pixels)
        if pixel not in excluded_set
    )
    column_moment_bound = 255 * sum(
        pixel % shape[3]
        for pixel in range(detector_pixels)
        if pixel not in excluded_set
    )
    expected_layout = [
        "total_lo",
        "total_hi",
        "row_lo",
        "row_hi",
        "column_lo",
        "column_hi",
        "padding_0",
        "padding_1",
    ]
    expected = {
        "schema": "quantem.gpu.prepared-dpc-moments/v1",
        "source_identity_sha256": source_identity,
        "working_uint8_sha256": manifest.get("prepared_uint8_sha256"),
        "detector_mask_sha256": manifest.get("detector_mask_sha256"),
        "detector_selection": "all-nonexcluded-v1",
        "scan_count": scan_count,
        "selected_detector_pixels": selected,
        "detector_columns": shape[3],
        "dtype": "little-endian-u32",
        "word_order": "little-endian-u32-pairs",
        "words_per_scan": 8,
        "layout": expected_layout,
        "total_bound": str(total_bound),
        "row_moment_bound": str(row_moment_bound),
        "column_moment_bound": str(column_moment_bound),
        "narrow_integer": total_bound <= np.iinfo(np.uint32).max,
        "narrow_products": max(row_moment_bound, column_moment_bound)
        <= np.iinfo(np.uint32).max,
    }
    mismatches = {
        key: (prepared.get(key), value)
        for key, value in expected.items()
        if prepared.get(key) != value
    }
    if mismatches:
        raise MPSCompactV3Error(
            f"{path} compact prepared DPC moments disagree with the source: "
            f"{mismatches}"
        )
    file_offset = prepared.get("file_offset")
    prepared_bytes = prepared.get("file_bytes")
    expected_bytes = scan_count * 8 * 4
    if (
        type(file_offset) is not int
        or type(prepared_bytes) is not int
        or file_offset < 0
        or file_offset % 4
        or prepared_bytes != expected_bytes
        or file_offset > file_bytes - prepared_bytes
    ):
        raise MPSCompactV3Error("compact prepared DPC byte range is invalid")
    prepared_end = file_offset + prepared_bytes
    for range_start, range_end, label in ranges:
        if file_offset < range_end and range_start < prepared_end:
            raise MPSCompactV3Error(f"compact prepared DPC range overlaps {label}")
    return MPSCompactV3PreparedDPCMoments(
        file_offset=file_offset,
        file_bytes=prepared_bytes,
        sha256=_require_sha256(prepared.get("sha256"), "prepared DPC identity"),
        working_uint8_sha256=_require_sha256(
            prepared.get("working_uint8_sha256"),
            "prepared DPC working uint8 identity",
        ),
        detector_mask_sha256=_require_sha256(
            prepared.get("detector_mask_sha256"),
            "prepared DPC detector mask identity",
        ),
        scan_count=scan_count,
        selected_detector_pixels=selected,
        detector_columns=shape[3],
        total_bound=total_bound,
        row_moment_bound=row_moment_bound,
        column_moment_bound=column_moment_bound,
        narrow_integer=total_bound <= np.iinfo(np.uint32).max,
        narrow_products=max(row_moment_bound, column_moment_bound)
        <= np.iinfo(np.uint32).max,
    )


def _parse_prepared_detector_products(
    path: Path,
    file_bytes: int,
    manifest: dict[str, Any],
    shape: tuple[int, int, int, int],
    source_identity: str,
    ranges: list[tuple[int, int, str]],
    prepared_dpc: MPSCompactV3PreparedDPCMoments | None,
) -> MPSCompactV3PreparedDetectorProducts | None:
    prepared = manifest.get("prepared_detector_products")
    if prepared is None:
        return None
    if not isinstance(prepared, dict):
        raise MPSCompactV3Error("compact prepared detector products are not an object")
    calibration = manifest.get("detector_calibration")
    if not isinstance(calibration, dict):
        raise MPSCompactV3Error("prepared detector products require calibration")
    center_value = calibration.get("detector_center_px")
    radius_value = calibration.get("bright_field_radius_px")
    if (
        calibration.get("schema") != "quantem.gpu.detector-calibration/v1"
        or calibration.get("source_identity_sha256") != source_identity
        or not isinstance(center_value, list)
        or len(center_value) != 2
        or any(type(value) not in {int, float} for value in center_value)
        or any(not math.isfinite(float(value)) for value in center_value)
        or not 0 <= float(center_value[0]) < shape[2]
        or not 0 <= float(center_value[1]) < shape[3]
        or type(radius_value) not in {int, float}
        or not math.isfinite(float(radius_value))
        or not 0 < float(radius_value) <= math.hypot(shape[2], shape[3])
    ):
        raise MPSCompactV3Error("prepared detector calibration is invalid")
    center = (float(center_value[0]), float(center_value[1]))
    radius = float(radius_value)
    calibration_sha256 = _calibration_digest(calibration)
    product_order = ("bf", "abf", "adf")
    geometries = {
        "bf": (0.0, radius),
        "abf": (0.5 * radius, radius),
        "adf": (radius, 2.0 * radius),
    }
    expected = {
        "schema": "quantem.gpu.prepared-detector-products/v1",
        "source_identity_sha256": source_identity,
        "working_uint8_sha256": manifest.get("prepared_uint8_sha256"),
        "detector_mask_sha256": manifest.get("detector_mask_sha256"),
        "detector_calibration_sha256": calibration_sha256,
        "detector_calibration_digest_encoding": _CALIBRATION_DIGEST_ENCODING,
        "scan_shape": list(shape[:2]),
        "detector_shape": list(shape[2:]),
        "product_dtype": "little-endian-u32",
        "mask_dtype": "uint8-binary-row-major",
        "mask_rule": "quantem.gpu.detector-mask-inclusive/v1",
        "product_order": list(product_order),
    }
    mismatches = {
        key: (prepared.get(key), value)
        for key, value in expected.items()
        if prepared.get(key) != value
    }
    if mismatches:
        raise MPSCompactV3Error(
            f"{path} compact prepared detector products disagree with the source: "
            f"{mismatches}"
        )
    raw_products = prepared.get("products")
    if not isinstance(raw_products, list) or len(raw_products) != len(product_order):
        raise MPSCompactV3Error("compact prepared detector product list is incomplete")
    detector_pixels = shape[2] * shape[3]
    scan_count = shape[0] * shape[1]
    occupied_ranges = list(ranges)
    if prepared_dpc is not None:
        occupied_ranges.append(
            (
                prepared_dpc.file_offset,
                prepared_dpc.file_offset + prepared_dpc.file_bytes,
                "prepared DPC moments",
            )
        )
    parsed: list[MPSCompactV3PreparedDetectorProduct] = []
    for ordinal, (raw, name) in enumerate(
        zip(raw_products, product_order, strict=True)
    ):
        if not isinstance(raw, dict):
            raise MPSCompactV3Error(
                f"compact prepared detector product {ordinal} is invalid"
            )
        inner, outer = geometries[name]
        product_expected = {
            "name": name,
            "center_px": list(center),
            "inner_radius_px": inner,
            "outer_radius_px": outer,
        }
        product_mismatches = {
            key: (raw.get(key), value)
            for key, value in product_expected.items()
            if raw.get(key) != value
        }
        if product_mismatches:
            raise MPSCompactV3Error(
                f"compact prepared {name.upper()} geometry disagrees with calibration: "
                f"{product_mismatches}"
            )
        selected = raw.get("selected_detector_pixels")
        mask_offset = raw.get("mask_file_offset")
        mask_bytes = raw.get("mask_file_bytes")
        values_offset = raw.get("values_file_offset")
        values_bytes = raw.get("values_file_bytes")
        if (
            type(selected) is not int
            or not 0 <= selected <= detector_pixels
            or type(mask_offset) is not int
            or type(mask_bytes) is not int
            or mask_offset < 0
            or mask_bytes != detector_pixels
            or mask_offset > file_bytes - mask_bytes
            or type(values_offset) is not int
            or type(values_bytes) is not int
            or values_offset < 0
            or values_bytes != scan_count * 4
            or values_offset > file_bytes - values_bytes
        ):
            raise MPSCompactV3Error(
                f"compact prepared {name.upper()} byte ranges are invalid"
            )
        mask_sha256 = _require_sha256(
            raw.get("mask_sha256"), f"prepared {name} mask identity"
        )
        values_sha256 = _require_sha256(
            raw.get("values_sha256"), f"prepared {name} values identity"
        )
        occupied_ranges.extend(
            [
                (mask_offset, mask_offset + mask_bytes, f"prepared {name} mask"),
                (
                    values_offset,
                    values_offset + values_bytes,
                    f"prepared {name} values",
                ),
            ]
        )
        parsed.append(
            MPSCompactV3PreparedDetectorProduct(
                name=name,
                center_px=center,
                inner_radius_px=inner,
                outer_radius_px=outer,
                selected_detector_pixels=selected,
                mask_file_offset=mask_offset,
                mask_file_bytes=mask_bytes,
                mask_sha256=mask_sha256,
                values_file_offset=values_offset,
                values_file_bytes=values_bytes,
                values_sha256=values_sha256,
            )
        )
    occupied_ranges.sort()
    for previous, current in pairwise(occupied_ranges):
        if current[0] < previous[1]:
            raise MPSCompactV3Error(
                f"compact ranges overlap between {previous[2]} and {current[2]}"
            )
    return MPSCompactV3PreparedDetectorProducts(
        calibration_sha256=calibration_sha256,
        working_uint8_sha256=str(manifest["prepared_uint8_sha256"]),
        detector_mask_sha256=str(manifest["detector_mask_sha256"]),
        products=tuple(parsed),
    )


def read_compact_v3_index(path: str | os.PathLike[str]) -> MPSCompactV3Index:
    """Parse and cross-check both metadata copies without reading payloads."""
    source = Path(path).expanduser().resolve(strict=True)
    file_bytes = source.stat().st_size
    started = time.perf_counter()
    del started  # The caller owns phase timing; parsing remains side-effect free.
    with source.open("rb") as stream:
        prelude = stream.read(_PRELUDE.size)
        if len(prelude) != _PRELUDE.size:
            raise MPSCompactV3Error("compact container prelude is truncated")
        magic, header_bytes, header_crc32, binary_offset, binary_bytes = (
            _PRELUDE.unpack(prelude)
        )
        if magic != _CONTAINER_MAGIC:
            raise MPSCompactV3Error("source has no QuantEM compact container prelude")
        if (
            header_bytes <= 0
            or binary_offset < _PRELUDE.size + header_bytes
            or binary_offset + binary_bytes > file_bytes
        ):
            raise MPSCompactV3Error("compact metadata range is outside the file")
        header = stream.read(header_bytes)
        if len(header) != header_bytes or zlib.crc32(header) != header_crc32:
            raise MPSCompactV3Error("compact JSON metadata failed its CRC-32 check")
        try:
            manifest = json.loads(header)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MPSCompactV3Error("compact JSON metadata is invalid") from error
        stream.seek(binary_offset)
        binary = stream.read(binary_bytes)
        if len(binary) != binary_bytes:
            raise MPSCompactV3Error("compact binary index is truncated")

    if len(binary) < _INDEX_HEADER.size + 4 + 32:
        raise MPSCompactV3Error("compact binary index is too small")
    (
        index_magic,
        shard_count,
        reserved,
        scan_rows,
        scan_columns,
        detector_rows,
        detector_columns,
        scans_per_shard,
        scan_tile,
        header_encoding,
    ) = _INDEX_HEADER.unpack_from(binary)
    shape = (scan_rows, scan_columns, detector_rows, detector_columns)
    if index_magic != _INDEX_MAGIC or reserved != 0 or header_encoding != 1:
        raise MPSCompactV3Error("unsupported QGIX v3 binary header")
    if min(*shape, shard_count, scans_per_shard) <= 0 or scan_tile != 32:
        raise MPSCompactV3Error("invalid QGIX v3 dimensions or scan tile")
    if scans_per_shard % scan_tile or scan_rows * scan_columns != (
        shard_count * scans_per_shard
    ):
        raise MPSCompactV3Error("QGIX v3 shards do not exactly cover the scan")
    detector_pixels = detector_rows * detector_columns
    tile_count = scans_per_shard // scan_tile
    header_words_per_pixel = (tile_count + 31) // 32 + (tile_count + 7) // 8
    expected_header_words = detector_pixels * header_words_per_pixel

    cursor = _INDEX_HEADER.size
    (mask_count,) = struct.unpack_from("<I", binary, cursor)
    cursor += 4
    if mask_count > detector_pixels or cursor + mask_count * 4 + 32 > len(binary):
        raise MPSCompactV3Error("invalid QGIX v3 detector mask")
    excluded = struct.unpack_from(f"<{mask_count}I", binary, cursor)
    cursor += mask_count * 4
    if len(set(excluded)) != mask_count or any(
        pixel >= detector_pixels for pixel in excluded
    ):
        raise MPSCompactV3Error("detector mask indices are not unique and in range")
    source_identity = binary[cursor : cursor + 32].hex()
    cursor += 32

    ranges: list[tuple[int, int, str]] = []
    shards: list[MPSCompactV3Shard] = []
    for shard_index in range(shard_count):
        if cursor + _SHARD_RECORD.size > len(binary):
            raise MPSCompactV3Error(f"shard record {shard_index} is truncated")
        (
            payload_offset,
            payload_bytes,
            lengths_offset,
            lengths_bytes,
            headers_offset,
            headers_bytes,
            decoded_bytes,
            descriptor_count,
            chunk_count,
            payload_sha256,
        ) = _SHARD_RECORD.unpack_from(binary, cursor)
        cursor += _SHARD_RECORD.size
        if (
            payload_bytes <= 0
            or payload_bytes != decoded_bytes
            or payload_bytes % 4
            or lengths_offset != 0
            or lengths_bytes != 0
            or chunk_count != 0
            or descriptor_count != expected_header_words
            or headers_bytes != expected_header_words * 4
            or payload_bytes // 4 >= 1 << 27
        ):
            raise MPSCompactV3Error(f"shard {shard_index} has inconsistent counts")
        for offset, count, label in (
            (payload_offset, payload_bytes, "payload"),
            (headers_offset, headers_bytes, "headers"),
        ):
            if offset > file_bytes or count > file_bytes - offset:
                raise MPSCompactV3Error(
                    f"shard {shard_index} {label} range is outside the file"
                )
            ranges.append((offset, offset + count, f"shard {shard_index} {label}"))
        shards.append(
            MPSCompactV3Shard(
                payload_offset=payload_offset,
                payload_bytes=payload_bytes,
                headers_offset=headers_offset,
                headers_bytes=headers_bytes,
                payload_sha256=payload_sha256.hex(),
                header_words=descriptor_count,
            )
        )
    if cursor != len(binary):
        raise MPSCompactV3Error("compact binary index has trailing bytes")
    ranges.sort()
    for previous, current in pairwise(ranges):
        if current[0] < previous[1]:
            raise MPSCompactV3Error(
                f"compact ranges overlap between {previous[2]} and {current[2]}"
            )

    if not isinstance(manifest, dict):
        raise MPSCompactV3Error("compact JSON metadata must contain an object")
    if (
        manifest.get("schema") != "quantem.gpu.packed-detector-h5/v3"
        or manifest.get("status") != "complete"
        or manifest.get("payload_codec") != "direct-bitpacked-u32"
        or manifest.get("working_dtype") != "uint8"
        or manifest.get("source_shape") != list(shape)
        or manifest.get("shard_count") != shard_count
        or manifest.get("scan_tile") != scan_tile
        or manifest.get("source_identity_sha256") != source_identity
    ):
        raise MPSCompactV3Error("compact JSON metadata conflicts with the binary index")
    working_sha = _require_sha256(
        manifest.get("prepared_uint8_sha256"), "prepared uint8 identity"
    )
    raw_sha: str | None = None
    mask_sha: str | None = None
    masked_pixels_sha: str | None = None
    masked_raw_values: tuple[int, ...] | None = None
    raw_access_mode = "mask_applied_only_legacy"
    embedded_semantics = manifest.get("source_dtype") is not None
    manifest_mask = tuple(
        int(value) for value in manifest.get("masked_detector_pixels", [])
    )
    if embedded_semantics:
        raw_sha = _require_sha256(
            manifest.get("source_raw_logical_sha256"), "raw logical identity"
        )
        mask_sha = _require_sha256(
            manifest.get("detector_mask_sha256"), "detector mask identity"
        )
        if (
            manifest.get("source_dtype") != "uint16"
            or manifest.get("scan_bin") != 1
            or manifest.get("detector_bin") != 1
            or manifest.get("crop", "missing") is not None
            or manifest.get("working_value_definition")
            != "all admitted source counts exactly; authenticated dead pixels set to zero"
            or manifest_mask != tuple(excluded)
        ):
            raise MPSCompactV3Error("embedded scientific semantics are incomplete")
        raw_values_value = manifest.get("masked_detector_raw_values")
        ordered_pixels_sha_value = manifest.get("masked_detector_pixels_sha256")
        if raw_values_value is not None and ordered_pixels_sha_value is not None:
            if not isinstance(raw_values_value, list):
                raise MPSCompactV3Error("masked detector raw values must be an array")
            masked_raw_values = tuple(int(value) for value in raw_values_value)
            masked_pixels_sha = _require_sha256(
                ordered_pixels_sha_value, "ordered masked detector pixels"
            )
            ordered_pixels = struct.pack(f"<{len(excluded)}I", *excluded)
            if (
                len(masked_raw_values) != len(excluded)
                or any(
                    value < 0 or value > np.iinfo(np.uint16).max
                    for value in masked_raw_values
                )
                or hashlib.sha256(ordered_pixels).hexdigest() != masked_pixels_sha
            ):
                raise MPSCompactV3Error("raw exclusion constants are malformed")
            raw_access_mode = "exact_exclusion_constants"
        elif not excluded:
            raw_access_mode = "exact_no_exclusions"
    elif (
        excluded
        or manifest_mask
        or "source_raw_logical_sha256" in manifest
        or "detector_mask_sha256" in manifest
    ):
        raise MPSCompactV3Error(
            "minimal v3 metadata contains ambiguous source semantics"
        )
    else:
        _require_sha256(manifest.get("parent_hdf5_sha256"), "parent HDF5 identity")
        raw_access_mode = "external_audit_required"

    prepared_dpc_moments = _parse_prepared_dpc_moments(
        source,
        file_bytes,
        manifest,
        shape,
        source_identity,
        tuple(excluded),
        ranges,
    )
    prepared_detector_products = _parse_prepared_detector_products(
        source,
        file_bytes,
        manifest,
        shape,
        source_identity,
        ranges,
        prepared_dpc_moments,
    )

    return MPSCompactV3Index(
        path=source,
        file_bytes=file_bytes,
        shape=shape,
        scans_per_shard=scans_per_shard,
        scan_tile=scan_tile,
        source_identity_sha256=source_identity,
        source_raw_logical_sha256=raw_sha,
        working_logical_sha256=working_sha,
        detector_mask_sha256=mask_sha,
        masked_detector_pixels_sha256=masked_pixels_sha,
        masked_detector_raw_values=masked_raw_values,
        raw_access_mode=raw_access_mode,
        excluded_detector_pixels=tuple(excluded),
        embedded_scientific_semantics=embedded_semantics,
        header_words_per_pixel=header_words_per_pixel,
        prepared_dpc_moments=prepared_dpc_moments,
        prepared_detector_products=prepared_detector_products,
        shards=tuple(shards),
        manifest=manifest,
    )


def _metal_module():
    try:
        import Metal
    except ImportError as error:
        raise RuntimeError(
            "Python MPS compact v3 requires pyobjc-framework-Metal"
        ) from error
    return Metal


def _make_pipelines(device, Metal):
    registry_id = int(device.registryID())
    cached = _pipeline_cache.get(registry_id)
    if cached is not None:
        return cached
    started = time.perf_counter()
    shader_path = Path(__file__).parent / "kernels" / "compact_v3.msl"
    source = shader_path.read_text()
    options = Metal.MTLCompileOptions.alloc().init()
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None or error is not None:
        raise RuntimeError(f"Python MPS compact v3 Metal compile failed: {error}")
    pipelines = []
    for name in (
        "compact_v3_validate_headers",
        "compact_v3_selected_diffraction",
        "compact_v3_detector_update",
        "compact_v3_full_decode_u8",
        "compact_v3_detector_sum_u64",
    ):
        function = library.newFunctionWithName_(name)
        pipeline, pipeline_error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None or pipeline_error is not None:
            raise RuntimeError(f"Python MPS pipeline {name} failed: {pipeline_error}")
        pipelines.append(pipeline)
    queue = device.newCommandQueue()
    if queue is None:
        raise RuntimeError("Python MPS could not create a Metal command queue")
    result = (*pipelines, queue, (time.perf_counter() - started) * 1_000.0)
    _pipeline_cache[registry_id] = result
    return result


def _allocate_shared(device, Metal, nbytes: int, label: str):
    buffer = device.newBufferWithLength_options_(
        int(nbytes), Metal.MTLResourceStorageModeShared
    )
    if buffer is None:
        raise MemoryError(f"Metal could not allocate {nbytes} bytes for {label}")
    return buffer


def _buffer_view(buffer, nbytes: int | None = None) -> memoryview:
    count = int(buffer.length()) if nbytes is None else int(nbytes)
    return memoryview(buffer.contents().as_buffer(count))


def _pread_exact(fd: int, buffer: memoryview, offset: int, label: str) -> None:
    position = 0
    while position < len(buffer):
        count = os.preadv(fd, [buffer[position:]], int(offset) + position)
        if count <= 0:
            raise MPSCompactV3Error(f"short read while loading {label}")
        position += count


def _release(buffer) -> None:
    if buffer is None:
        return
    try:
        buffer.release()
    except (AttributeError, ValueError):
        pass


def _complete(command, operation: str) -> float:
    command.commit()
    command.waitUntilCompleted()
    if command.error() is not None:
        raise RuntimeError(
            f"Python MPS compact v3 {operation} failed: {command.error()}"
        )
    return max(0.0, float(command.GPUEndTime()) - float(command.GPUStartTime())) * 1_000


def _parallel_authenticate(index: MPSCompactV3Index) -> None:
    descriptor = os.open(index.path, os.O_RDONLY)
    try:
        mapped = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        try:

            def authenticate(item: tuple[int, MPSCompactV3Shard]) -> str | None:
                shard_index, shard = item
                payload = memoryview(mapped)[
                    shard.payload_offset : shard.payload_offset + shard.payload_bytes
                ]
                try:
                    observed = hashlib.sha256(payload).hexdigest()
                finally:
                    payload.release()
                if observed != shard.payload_sha256:
                    return (
                        f"shard {shard_index} is {observed}, expected "
                        f"{shard.payload_sha256}"
                    )
                return None

            with ThreadPoolExecutor(
                max_workers=min(len(index.shards), os.cpu_count() or 1)
            ) as pool:
                failures = [
                    failure
                    for failure in pool.map(authenticate, enumerate(index.shards))
                    if failure is not None
                ]
        finally:
            mapped.close()
    finally:
        os.close(descriptor)
    if failures:
        raise MPSCompactV3Error(
            "parallel mapped payload authentication failed: " + "; ".join(failures)
        )


def _load_prepared_dpc(
    index: MPSCompactV3Index,
    descriptor: int,
    device,
    Metal,
) -> tuple[Any | None, list[Any], float, float, float]:
    prepared = index.prepared_dpc_moments
    if prepared is None:
        return None, [], 0.0, 0.0, 0.0
    moment_buffer = _allocate_shared(
        device, Metal, prepared.file_bytes, "prepared DPC moments"
    )
    row_buffer = None
    column_buffer = None
    try:
        moment_view = _buffer_view(moment_buffer, prepared.file_bytes)
        read_started = time.perf_counter()
        _pread_exact(
            descriptor,
            moment_view,
            prepared.file_offset,
            "prepared DPC moments",
        )
        read_ms = (time.perf_counter() - read_started) * 1_000.0
        authentication_started = time.perf_counter()
        observed = hashlib.sha256(moment_view).hexdigest()
        authentication_ms = (time.perf_counter() - authentication_started) * 1_000.0
        if observed != prepared.sha256:
            raise MPSCompactV3Error(
                f"prepared DPC SHA-256 is {observed}, expected {prepared.sha256}"
            )

        prime_started = time.perf_counter()
        row, column = _prepared_dpc_display_arrays(
            moment_view,
            shape=index.shape,
            prepared=prepared,
        )
        row_buffer = _allocate_shared(device, Metal, row.nbytes, "prepared DPC row")
        column_buffer = _allocate_shared(
            device, Metal, column.nbytes, "prepared DPC column"
        )
        _buffer_view(row_buffer, row.nbytes)[:] = row.view(np.uint8)
        _buffer_view(column_buffer, column.nbytes)[:] = column.view(np.uint8)
        prime_ms = (time.perf_counter() - prime_started) * 1_000.0
        return (
            moment_buffer,
            [row_buffer, column_buffer],
            read_ms,
            authentication_ms,
            prime_ms,
        )
    except Exception:
        _release(moment_buffer)
        _release(row_buffer)
        _release(column_buffer)
        raise


@dataclass
class _MPSPreparedDetectorResident:
    metadata: MPSCompactV3PreparedDetectorProduct
    mask: np.ndarray
    values_buffer: Any


def _load_prepared_detector_products(
    index: MPSCompactV3Index,
    descriptor: int,
    device,
    Metal,
) -> tuple[dict[str, _MPSPreparedDetectorResident], float, float, int]:
    prepared = index.prepared_detector_products
    if prepared is None:
        return {}, 0.0, 0.0, 0
    residents: dict[str, _MPSPreparedDetectorResident] = {}
    read_ms = 0.0
    authentication_ms = 0.0
    total_bytes = 0
    excluded = set(index.excluded_detector_pixels)
    try:
        for product in prepared.products:
            mask_bytes = bytearray(product.mask_file_bytes)
            values_buffer = _allocate_shared(
                device,
                Metal,
                product.values_file_bytes,
                f"prepared {product.name} values",
            )
            read_started = time.perf_counter()
            _pread_exact(
                descriptor,
                memoryview(mask_bytes),
                product.mask_file_offset,
                f"prepared {product.name} mask",
            )
            values_view = _buffer_view(values_buffer, product.values_file_bytes)
            _pread_exact(
                descriptor,
                values_view,
                product.values_file_offset,
                f"prepared {product.name} values",
            )
            read_ms += (time.perf_counter() - read_started) * 1_000.0

            authentication_started = time.perf_counter()
            mask_observed = hashlib.sha256(mask_bytes).hexdigest()
            values_observed = hashlib.sha256(values_view).hexdigest()
            authentication_ms += (
                time.perf_counter() - authentication_started
            ) * 1_000.0
            if mask_observed != product.mask_sha256:
                raise MPSCompactV3Error(
                    f"prepared {product.name.upper()} mask SHA-256 is "
                    f"{mask_observed}, expected {product.mask_sha256}"
                )
            if values_observed != product.values_sha256:
                raise MPSCompactV3Error(
                    f"prepared {product.name.upper()} values SHA-256 is "
                    f"{values_observed}, expected {product.values_sha256}"
                )
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
            if np.any(mask > 1):
                raise MPSCompactV3Error(
                    f"prepared {product.name.upper()} mask is not binary"
                )
            if any(mask[pixel] for pixel in excluded):
                raise MPSCompactV3Error(
                    f"prepared {product.name.upper()} mask selects an excluded pixel"
                )
            selected = int(mask.sum(dtype=np.uint64))
            if selected != product.selected_detector_pixels:
                raise MPSCompactV3Error(
                    f"prepared {product.name.upper()} mask selects {selected} pixels, "
                    f"expected {product.selected_detector_pixels}"
                )
            residents[product.name] = _MPSPreparedDetectorResident(
                metadata=product,
                mask=mask,
                values_buffer=values_buffer,
            )
            total_bytes += product.mask_file_bytes + product.values_file_bytes
        return residents, read_ms, authentication_ms, total_bytes
    except Exception:
        for resident in residents.values():
            _release(resident.values_buffer)
        if "values_buffer" in locals() and (
            not residents
            or all(
                resident.values_buffer is not values_buffer
                for resident in residents.values()
            )
        ):
            _release(values_buffer)
        raise


def _prepared_dpc_display_arrays(
    moment_bytes: bytes | bytearray | memoryview,
    *,
    shape: tuple[int, int, int, int],
    prepared: MPSCompactV3PreparedDPCMoments,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate exact prepared moments and return centered float32 DPC maps."""
    words = np.frombuffer(
        moment_bytes,
        dtype="<u4",
        count=prepared.scan_count * 8,
    ).reshape(prepared.scan_count, 8)
    if np.any(words[:, 6:]):
        raise MPSCompactV3Error("prepared DPC padding words must be zero")
    total = words[:, 0].astype(np.uint64) | (words[:, 1].astype(np.uint64) << 32)
    row_moment = words[:, 2].astype(np.uint64) | (words[:, 3].astype(np.uint64) << 32)
    column_moment = words[:, 4].astype(np.uint64) | (
        words[:, 5].astype(np.uint64) << 32
    )
    if (
        np.any(total > prepared.total_bound)
        or np.any(row_moment > prepared.row_moment_bound)
        or np.any(column_moment > prepared.column_moment_bound)
        or np.any(row_moment > total * (shape[2] - 1))
        or np.any(column_moment > total * (shape[3] - 1))
        or np.any((total == 0) & ((row_moment != 0) | (column_moment != 0)))
    ):
        raise MPSCompactV3Error("prepared DPC moments violate exact source bounds")
    row = np.zeros(prepared.scan_count, dtype=np.float64)
    column = np.zeros(prepared.scan_count, dtype=np.float64)
    np.divide(row_moment, total, out=row, where=total != 0)
    np.divide(column_moment, total, out=column, where=total != 0)
    row = row.astype(np.float32)
    column = column.astype(np.float32)
    row -= float(np.mean(row, dtype=np.float64))
    column -= float(np.mean(column, dtype=np.float64))
    return row, column


class MPSCompactV3Resident:
    """Python-owned exact packed source resident in Apple unified memory."""

    def __init__(
        self,
        *,
        index: MPSCompactV3Index,
        load_metrics: MPSCompactV3LoadMetrics,
        Metal,
        device,
        queue,
        validation_pipeline,
        selected_pipeline,
        detector_pipeline,
        full_decode_pipeline,
        detector_sum_pipeline,
        payload_buffers: list[Any],
        header_buffers: list[Any],
        excluded_buffer,
        maximum_widths: np.ndarray,
        detector_outputs: list[Any],
        detector_entry_buffer,
        diffraction_buffer,
        detector_sum_buffer,
        prepared_dpc_moment_buffer,
        prepared_dpc_outputs: list[Any],
        prepared_detector_products: dict[str, _MPSPreparedDetectorResident],
    ) -> None:
        self.index = index
        self.load_metrics = load_metrics
        self._Metal = Metal
        self._device = device
        self._queue = queue
        self._validation_pipeline = validation_pipeline
        self._selected_pipeline = selected_pipeline
        self._detector_pipeline = detector_pipeline
        self._full_decode_pipeline = full_decode_pipeline
        self._detector_sum_pipeline = detector_sum_pipeline
        self._payload_buffers = payload_buffers
        self._header_buffers = header_buffers
        self._excluded_buffer = excluded_buffer
        self._maximum_widths = maximum_widths
        self._detector_outputs = detector_outputs
        self._detector_entry_buffer = detector_entry_buffer
        self._diffraction_buffer = diffraction_buffer
        self._detector_sum_buffer = detector_sum_buffer
        self._detector_sum_ready = False
        self._prepared_dpc_moment_buffer = prepared_dpc_moment_buffer
        self._prepared_dpc_outputs = prepared_dpc_outputs
        self._prepared_detector_products = prepared_detector_products
        self._detector_mask = np.zeros(index.detector_pixels, dtype=np.uint8)
        self._active_detector_output = 0
        self._released = False

    @property
    def is_released(self) -> bool:
        return self._released

    def _require_resident(self) -> None:
        if self._released:
            raise RuntimeError("the Python MPS compact source has been released")

    def extract_diffraction(self, scan_row: int, scan_column: int) -> np.ndarray:
        """Return one exact mask-applied row-major diffraction pattern."""
        self._require_resident()
        scan_rows, scan_columns, _, _ = self.index.shape
        if not (0 <= scan_row < scan_rows and 0 <= scan_column < scan_columns):
            raise IndexError("selected scan coordinate is outside the source")
        global_scan = scan_row * scan_columns + scan_column
        shard_index, local_scan = divmod(global_scan, self.index.scans_per_shard)
        parameters = struct.pack(
            "<6I",
            local_scan,
            self.index.tile_count,
            self.index.detector_pixels,
            self.index.scan_tile,
            self.index.header_words_per_pixel,
            1,
        )
        command = self._queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self._selected_pipeline)
        encoder.setBuffer_offset_atIndex_(self._payload_buffers[shard_index], 0, 0)
        encoder.setBuffer_offset_atIndex_(self._header_buffers[shard_index], 0, 1)
        encoder.setBuffer_offset_atIndex_(self._excluded_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(self._diffraction_buffer, 0, 3)
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 4)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._Metal.MTLSizeMake(self.index.detector_pixels, 1, 1),
            self._Metal.MTLSizeMake(256, 1, 1),
        )
        encoder.endEncoding()
        _complete(command, "selected diffraction")
        return np.frombuffer(
            _buffer_view(self._diffraction_buffer),
            dtype="<u4",
            count=self.index.detector_pixels,
        ).copy()

    def update_virtual_detector(
        self, mask: np.ndarray, *, force_rebase: bool = False
    ) -> MPSCompactV3DetectorMetrics:
        """Apply a binary row-major detector mask with exact signed deltas."""
        self._require_resident()
        normalized = np.asarray(mask, dtype=np.uint8).reshape(-1).copy()
        if normalized.size != self.index.detector_pixels or np.any(normalized > 1):
            raise ValueError(
                "detector mask must contain one zero-or-one byte per pixel"
            )
        if self.index.excluded_detector_pixels:
            normalized[np.asarray(self.index.excluded_detector_pixels)] = 0
        active = np.flatnonzero(normalized)
        maximum_sum = sum(
            (1 << int(self._maximum_widths[pixel])) - 1
            for pixel in active
            if self._maximum_widths[pixel]
        )
        if maximum_sum > np.iinfo(np.uint32).max:
            raise OverflowError("detector mask can overflow exact uint32 output")
        prepared_product = next(
            (
                product
                for product in self._prepared_detector_products.values()
                if np.array_equal(product.mask, normalized)
            ),
            None,
        )
        if prepared_product is not None:
            changed_count = int(np.count_nonzero(normalized != self._detector_mask))
            if changed_count == 0:
                return MPSCompactV3DetectorMetrics("delta", 0, 0.0, 0.0, 0.0)
            next_output = 1 - self._active_detector_output
            started = time.perf_counter()
            command = self._queue.commandBuffer()
            encoder = command.blitCommandEncoder()
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                prepared_product.values_buffer,
                0,
                self._detector_outputs[next_output],
                0,
                self.index.scan_count * 4,
            )
            encoder.endEncoding()
            gpu_ms = _complete(command, "prepared virtual detector")
            wall_ms = (time.perf_counter() - started) * 1_000.0
            self._active_detector_output = next_output
            self._detector_mask = normalized
            return MPSCompactV3DetectorMetrics(
                mode="prepared",
                changed_detector_pixels=changed_count,
                wall_ms=wall_ms,
                gpu_ms=gpu_ms,
                framework_overhead_ms=max(0.0, wall_ms - gpu_ms),
            )
        rebase = force_rebase or not np.any(self._detector_mask)
        if rebase:
            changed = active
            coefficients = np.ones(changed.size, dtype=np.int32)
        else:
            changed = np.flatnonzero(normalized != self._detector_mask)
            coefficients = np.where(normalized[changed] != 0, 1, -1).astype(np.int32)
        mode = "rebase" if rebase else "delta"
        if changed.size == 0:
            self._detector_mask = normalized
            return MPSCompactV3DetectorMetrics(mode, 0, 0.0, 0.0, 0.0)
        entries = np.frombuffer(
            _buffer_view(self._detector_entry_buffer),
            dtype=np.int32,
            count=self.index.detector_pixels * 2,
        ).reshape(-1, 2)[: changed.size]
        entries[:, 0] = changed.astype(np.int32)
        entries[:, 1] = coefficients
        next_output = 1 - self._active_detector_output
        started = time.perf_counter()
        command = self._queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self._detector_pipeline)
        for shard_index, (payload, headers) in enumerate(
            zip(self._payload_buffers, self._header_buffers, strict=True)
        ):
            parameters = struct.pack(
                "<8I",
                self.index.scans_per_shard,
                self.index.tile_count,
                changed.size,
                shard_index * self.index.scans_per_shard,
                1 if rebase else 0,
                self.index.scan_tile,
                self.index.header_words_per_pixel,
                1,
            )
            encoder.setBuffer_offset_atIndex_(payload, 0, 0)
            encoder.setBuffer_offset_atIndex_(headers, 0, 1)
            encoder.setBuffer_offset_atIndex_(self._detector_entry_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(
                self._detector_outputs[self._active_detector_output], 0, 3
            )
            encoder.setBuffer_offset_atIndex_(self._detector_outputs[next_output], 0, 4)
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 5)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                self._Metal.MTLSizeMake((self.index.scans_per_shard + 31) // 32, 1, 1),
                self._Metal.MTLSizeMake(128, 1, 1),
            )
        encoder.endEncoding()
        gpu_ms = _complete(command, "virtual detector")
        wall_ms = (time.perf_counter() - started) * 1_000.0
        self._active_detector_output = next_output
        self._detector_mask = normalized
        return MPSCompactV3DetectorMetrics(
            mode=mode,
            changed_detector_pixels=int(changed.size),
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
            framework_overhead_ms=max(0.0, wall_ms - gpu_ms),
        )

    def mean_diffraction_pattern(self) -> MPSCompactV3MeanDiffraction:
        """Return an exact u64 detector sum and float32 mean diffraction map.

        The first call performs one complete resident pass. Later calls reuse
        the synchronized sum without decoding a dense four-dimensional array.
        """

        self._require_resident()
        wall_ms = 0.0
        gpu_ms = 0.0
        dispatch_count = 0
        if not self._detector_sum_ready:
            _buffer_view(self._detector_sum_buffer)[:] = b"\0" * int(
                self._detector_sum_buffer.length()
            )
            started = time.perf_counter()
            parameters = struct.pack(
                "<6I",
                self.index.scans_per_shard,
                self.index.tile_count,
                self.index.detector_pixels,
                self.index.scan_tile,
                self.index.header_words_per_pixel,
                1,
            )
            for payload, headers in zip(
                self._payload_buffers, self._header_buffers, strict=True
            ):
                command = self._queue.commandBuffer()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(self._detector_sum_pipeline)
                encoder.setBuffer_offset_atIndex_(payload, 0, 0)
                encoder.setBuffer_offset_atIndex_(headers, 0, 1)
                encoder.setBuffer_offset_atIndex_(self._excluded_buffer, 0, 2)
                encoder.setBuffer_offset_atIndex_(self._detector_sum_buffer, 0, 3)
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 4)
                encoder.dispatchThreads_threadsPerThreadgroup_(
                    self._Metal.MTLSizeMake(self.index.detector_pixels, 1, 1),
                    self._Metal.MTLSizeMake(256, 1, 1),
                )
                encoder.endEncoding()
                gpu_ms += _complete(command, "mean diffraction shard")
                dispatch_count += 1
            wall_ms = (time.perf_counter() - started) * 1_000.0
            self._detector_sum_ready = True
        detector_sum = np.frombuffer(
            _buffer_view(self._detector_sum_buffer),
            dtype="<u8",
            count=self.index.detector_pixels,
        ).copy()
        mean = detector_sum.astype(np.float32) / np.float32(self.index.scan_count)
        return MPSCompactV3MeanDiffraction(
            detector_sum=detector_sum,
            mean=mean,
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
            dispatch_count=dispatch_count,
            readback_bytes=int(self._detector_sum_buffer.length()),
        )

    def activate_prepared_detector_product(
        self, name: str
    ) -> MPSCompactV3DetectorMetrics:
        """Copy one authenticated BF, ABF, or ADF map into the active output."""
        self._require_resident()
        product = self._prepared_detector_products.get(name)
        if product is None:
            available = ", ".join(sorted(self._prepared_detector_products)) or "none"
            raise ValueError(
                f"prepared detector product {name!r} is unavailable; available: {available}"
            )
        return self.update_virtual_detector(product.mask)

    def virtual_detector_values(self) -> np.ndarray:
        """Copy the last completely published exact detector map to NumPy."""
        self._require_resident()
        return np.frombuffer(
            _buffer_view(self._detector_outputs[self._active_detector_output]),
            dtype="<u4",
            count=self.index.scan_count,
        ).copy()

    def prepared_dpc_values(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Copy the source-bound centered row and column DPC display maps.

        Examples
        --------
        >>> maps = source.prepared_dpc_values()
        >>> maps is None or maps[0].shape == (source.index.scan_count,)
        True
        """
        self._require_resident()
        if len(self._prepared_dpc_outputs) != 2:
            return None
        row, column = self._prepared_dpc_outputs
        return (
            np.frombuffer(
                _buffer_view(row), dtype="<f4", count=self.index.scan_count
            ).copy(),
            np.frombuffer(
                _buffer_view(column), dtype="<f4", count=self.index.scan_count
            ).copy(),
        )

    def prepared_dpc_moment_values(self) -> MPSCompactV3ExactDPCMoments | None:
        """Copy authenticated exact totals and detector-coordinate moments."""

        self._require_resident()
        prepared = self.index.prepared_dpc_moments
        if prepared is None or self._prepared_dpc_moment_buffer is None:
            return None
        words = np.frombuffer(
            _buffer_view(self._prepared_dpc_moment_buffer),
            dtype="<u4",
            count=prepared.scan_count * 8,
        ).reshape(prepared.scan_count, 8)

        def combine(low: np.ndarray, high: np.ndarray) -> np.ndarray:
            return low.astype(np.uint64) | (high.astype(np.uint64) << np.uint64(32))

        return MPSCompactV3ExactDPCMoments(
            total=combine(words[:, 0], words[:, 1]),
            detector_row_moment=combine(words[:, 2], words[:, 3]),
            detector_column_moment=combine(words[:, 4], words[:, 5]),
        )

    def prepared_dpc_display_buffer(self, component: str):
        """Borrow one persistent prepared DPC buffer for zero-copy display.

        Examples
        --------
        >>> row = source.prepared_dpc_display_buffer("row")
        """
        self._require_resident()
        if component not in {"row", "column"}:
            raise ValueError(
                f"prepared DPC component must be 'row' or 'column'; got {component!r}"
            )
        if len(self._prepared_dpc_outputs) != 2:
            return None
        return self._prepared_dpc_outputs[0 if component == "row" else 1]

    def hash_logical_working_u8(self) -> MPSCompactV3LogicalHashMetrics:
        """Stream a full exact scan-major u8 hash with one shard-sized output."""
        self._require_resident()
        if np.any(self._maximum_widths > 8):
            raise MPSCompactV3Error("logical u8 audit found a non-mask width above 8")
        output_bytes = self.index.scans_per_shard * self.index.detector_pixels
        if output_bytes % 4:
            raise MPSCompactV3Error("logical u8 audit byte count is not word aligned")
        output = _allocate_shared(
            self._device, self._Metal, output_bytes, "logical hash staging"
        )
        hasher = hashlib.sha256()
        gpu_ms = 0.0
        started = time.perf_counter()
        for payload, headers in zip(
            self._payload_buffers, self._header_buffers, strict=True
        ):
            parameters = struct.pack(
                "<7I",
                self.index.scans_per_shard,
                self.index.detector_pixels,
                self.index.tile_count,
                self.index.scan_tile,
                self.index.header_words_per_pixel,
                1,
                output_bytes // 4,
            )
            command = self._queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self._full_decode_pipeline)
            encoder.setBuffer_offset_atIndex_(payload, 0, 0)
            encoder.setBuffer_offset_atIndex_(headers, 0, 1)
            encoder.setBuffer_offset_atIndex_(self._excluded_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(output, 0, 3)
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 4)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                self._Metal.MTLSizeMake(output_bytes // 4, 1, 1),
                self._Metal.MTLSizeMake(256, 1, 1),
            )
            encoder.endEncoding()
            gpu_ms += _complete(command, "logical hash shard")
            hasher.update(_buffer_view(output, output_bytes))
        wall_ms = (time.perf_counter() - started) * 1_000.0
        _release(output)
        return MPSCompactV3LogicalHashMetrics(
            sha256=hasher.hexdigest(),
            logical_bytes=self.index.scan_count * self.index.detector_pixels,
            staging_bytes=output_bytes,
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
        )

    def release(self) -> None:
        """Release every +1 retained Metal buffer owned by this source."""
        if self._released:
            return
        for buffer in (
            self._payload_buffers
            + self._header_buffers
            + self._detector_outputs
            + self._prepared_dpc_outputs
            + [
                product.values_buffer
                for product in self._prepared_detector_products.values()
            ]
            + [
                self._excluded_buffer,
                self._detector_entry_buffer,
                self._diffraction_buffer,
                self._detector_sum_buffer,
                self._prepared_dpc_moment_buffer,
            ]
        ):
            _release(buffer)
        self._payload_buffers = []
        self._header_buffers = []
        self._detector_outputs = []
        self._excluded_buffer = None
        self._detector_entry_buffer = None
        self._diffraction_buffer = None
        self._detector_sum_buffer = None
        self._detector_sum_ready = False
        self._prepared_dpc_moment_buffer = None
        self._prepared_dpc_outputs = []
        self._prepared_detector_products = {}
        self._released = True


def load_compact_v3_mps(
    path: str | os.PathLike[str],
    *,
    expected_whole_file_sha256: str | None = None,
    authentication_policy: str = "bounded_sequential",
    should_cancel: Callable[[], bool] | None = None,
) -> MPSCompactV3Resident:
    """Authenticate, validate, and publish a QGIX v3 source on Python MPS.

    ``expected_whole_file_sha256`` optionally binds the complete container to
    an externally retained identity before any device allocation.
    """
    if authentication_policy not in {"bounded_sequential", "parallel_mapped_full_file"}:
        raise ValueError("unsupported Python MPS compact authentication policy")
    cancelled = should_cancel or (lambda: False)
    identity_started = time.perf_counter()
    if expected_whole_file_sha256 is not None:
        expected = _require_sha256(expected_whole_file_sha256, "whole-file identity")
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                if cancelled():
                    raise MPSCompactV3Cancelled("Python MPS compact load was superseded")
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise MPSCompactV3Error(
                "Lossless Pack whole-file SHA-256 does not match the expected "
                "identity. Verify the source and its retained checksum."
            )
    whole_file_integrity_ms = (
        (time.perf_counter() - identity_started) * 1_000.0
        if expected_whole_file_sha256 is not None else 0.0
    )
    Metal = _metal_module()
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("Python MPS compact v3 requires a physical Metal device")
    total_started = identity_started
    allocated_before = int(device.currentAllocatedSize())
    metadata_started = time.perf_counter()
    index = read_compact_v3_index(path)
    metadata_ms = (time.perf_counter() - metadata_started) * 1_000.0
    if cancelled():
        raise MPSCompactV3Cancelled("Python MPS compact load was superseded")
    (
        validation_pipeline,
        selected_pipeline,
        detector_pipeline,
        full_decode_pipeline,
        detector_sum_pipeline,
        queue,
        pipeline_compile_ms,
    ) = _make_pipelines(device, Metal)

    authentication_ms = whole_file_integrity_ms
    mapped_authentication_bytes = 0
    if authentication_policy == "parallel_mapped_full_file":
        authentication_started = time.perf_counter()
        _parallel_authenticate(index)
        authentication_ms += (time.perf_counter() - authentication_started) * 1_000.0
        mapped_authentication_bytes = index.file_bytes
    if cancelled():
        raise MPSCompactV3Cancelled("Python MPS compact load was superseded")

    payload_buffers: list[Any] = []
    header_buffers: list[Any] = []
    maximum_width_buffer = _allocate_shared(
        device, Metal, index.detector_pixels * 4, "maximum widths"
    )
    _buffer_view(maximum_width_buffer)[:] = b"\0" * (index.detector_pixels * 4)
    source_read_ms = 0.0
    gpu_validation_ms = 0.0
    prepared_dpc_moment_buffer = None
    prepared_dpc_outputs: list[Any] = []
    prepared_dpc_read_ms = 0.0
    prepared_dpc_authentication_ms = 0.0
    prepared_dpc_prime_ms = 0.0
    prepared_detector_products: dict[str, _MPSPreparedDetectorResident] = {}
    prepared_detector_product_read_ms = 0.0
    prepared_detector_product_authentication_ms = 0.0
    prepared_detector_product_bytes = 0
    descriptor = os.open(index.path, os.O_RDONLY)
    try:
        for shard_index, shard in enumerate(index.shards):
            if cancelled():
                raise MPSCompactV3Cancelled("Python MPS compact load was superseded")
            payload = _allocate_shared(
                device, Metal, shard.payload_bytes, f"shard {shard_index} payload"
            )
            headers = _allocate_shared(
                device, Metal, shard.headers_bytes, f"shard {shard_index} headers"
            )
            payload_buffers.append(payload)
            header_buffers.append(headers)
            read_started = time.perf_counter()
            payload_view = _buffer_view(payload, shard.payload_bytes)
            _pread_exact(
                descriptor,
                payload_view,
                shard.payload_offset,
                f"shard {shard_index} payload",
            )
            _pread_exact(
                descriptor,
                _buffer_view(headers, shard.headers_bytes),
                shard.headers_offset,
                f"shard {shard_index} headers",
            )
            source_read_ms += (time.perf_counter() - read_started) * 1_000.0
            if authentication_policy == "bounded_sequential":
                authentication_started = time.perf_counter()
                observed = hashlib.sha256(payload_view).hexdigest()
                authentication_ms += (
                    time.perf_counter() - authentication_started
                ) * 1_000.0
                if observed != shard.payload_sha256:
                    raise MPSCompactV3Error(
                        f"shard {shard_index} payload is {observed}, expected "
                        f"{shard.payload_sha256}"
                    )
            status = _allocate_shared(device, Metal, 4, "descriptor status")
            try:
                _buffer_view(status)[:] = b"\0\0\0\0"
                parameters = struct.pack(
                    "<6I",
                    index.detector_pixels,
                    shard.payload_bytes // 4,
                    index.tile_count,
                    index.header_words_per_pixel,
                    index.scan_tile,
                    1,
                )
                command = queue.commandBuffer()
                encoder = command.computeCommandEncoder()
                encoder.setComputePipelineState_(validation_pipeline)
                encoder.setBuffer_offset_atIndex_(headers, 0, 0)
                encoder.setBuffer_offset_atIndex_(status, 0, 1)
                encoder.setBytes_length_atIndex_(parameters, len(parameters), 2)
                encoder.setBuffer_offset_atIndex_(maximum_width_buffer, 0, 3)
                encoder.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(index.detector_pixels, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1),
                )
                encoder.endEncoding()
                gpu_validation_ms += _complete(
                    command, f"shard {shard_index} validation"
                )
                validation_status = struct.unpack_from("<I", _buffer_view(status))[0]
            finally:
                _release(status)
            if validation_status:
                raise MPSCompactV3Error(
                    f"shard {shard_index} header validation status is "
                    f"{validation_status}"
                )
            if cancelled():
                raise MPSCompactV3Cancelled("Python MPS compact load was superseded")
        (
            prepared_dpc_moment_buffer,
            prepared_dpc_outputs,
            prepared_dpc_read_ms,
            prepared_dpc_authentication_ms,
            prepared_dpc_prime_ms,
        ) = _load_prepared_dpc(index, descriptor, device, Metal)
        (
            prepared_detector_products,
            prepared_detector_product_read_ms,
            prepared_detector_product_authentication_ms,
            prepared_detector_product_bytes,
        ) = _load_prepared_detector_products(index, descriptor, device, Metal)
        if cancelled():
            raise MPSCompactV3Cancelled("Python MPS compact load was superseded")
    except Exception:
        for buffer in payload_buffers + header_buffers + prepared_dpc_outputs:
            _release(buffer)
        for product in prepared_detector_products.values():
            _release(product.values_buffer)
        _release(prepared_dpc_moment_buffer)
        _release(maximum_width_buffer)
        raise
    finally:
        os.close(descriptor)

    maximum_widths = np.frombuffer(
        _buffer_view(maximum_width_buffer), dtype="<u4", count=index.detector_pixels
    ).copy()
    _release(maximum_width_buffer)
    excluded_set = set(index.excluded_detector_pixels)
    if np.any(maximum_widths > 8) or any(
        maximum_widths[pixel] != 0 for pixel in excluded_set
    ):
        for buffer in payload_buffers + header_buffers + prepared_dpc_outputs:
            _release(buffer)
        for product in prepared_detector_products.values():
            _release(product.values_buffer)
        _release(prepared_dpc_moment_buffer)
        raise MPSCompactV3Error(
            "validated widths conflict with the u8 working dtype or detector mask"
        )

    excluded_values = np.zeros(index.detector_pixels, dtype=np.uint32)
    if excluded_set:
        excluded_values[np.asarray(sorted(excluded_set))] = 1
    excluded_buffer = _allocate_shared(
        device, Metal, excluded_values.nbytes, "detector exclusion mask"
    )
    _buffer_view(excluded_buffer)[:] = excluded_values.view(np.uint8)
    product_resident_bytes = index.scan_count * 4 * 2 + index.detector_pixels * (
        4 * 2 + 8 + 8
    )
    prepared_dpc_bytes = (
        index.prepared_dpc_moments.file_bytes
        if index.prepared_dpc_moments is not None
        else 0
    )
    product_resident_bytes += prepared_dpc_bytes + len(prepared_dpc_outputs) * (
        index.scan_count * 4
    )
    product_resident_bytes += prepared_detector_product_bytes
    detector_outputs = [
        _allocate_shared(device, Metal, index.scan_count * 4, f"detector output {slot}")
        for slot in range(2)
    ]
    for output in detector_outputs:
        _buffer_view(output)[:] = b"\0" * (index.scan_count * 4)
    diffraction_buffer = _allocate_shared(
        device, Metal, index.detector_pixels * 4, "selected diffraction"
    )
    detector_entry_buffer = _allocate_shared(
        device, Metal, index.detector_pixels * 8, "detector entries"
    )
    detector_sum_buffer = _allocate_shared(
        device, Metal, index.detector_pixels * 8, "mean diffraction detector sum"
    )
    _buffer_view(diffraction_buffer)[:] = b"\0" * (index.detector_pixels * 4)
    _buffer_view(detector_sum_buffer)[:] = b"\0" * (index.detector_pixels * 8)
    total_resident_bytes = index.packed_resident_bytes + product_resident_bytes
    metrics = MPSCompactV3LoadMetrics(
        metadata_ms=metadata_ms,
        pipeline_compile_ms=pipeline_compile_ms,
        authentication_ms=authentication_ms,
        source_read_ms=source_read_ms,
        gpu_validation_ms=gpu_validation_ms,
        prepared_dpc_read_ms=prepared_dpc_read_ms,
        prepared_dpc_authentication_ms=prepared_dpc_authentication_ms,
        prepared_dpc_prime_ms=prepared_dpc_prime_ms,
        prepared_detector_product_read_ms=prepared_detector_product_read_ms,
        prepared_detector_product_authentication_ms=(
            prepared_detector_product_authentication_ms
        ),
        resident_ready_ms=(time.perf_counter() - total_started) * 1_000.0,
        authentication_policy=authentication_policy,
        mapped_authentication_bytes=mapped_authentication_bytes,
        prepared_dpc_bytes=prepared_dpc_bytes,
        prepared_detector_product_bytes=prepared_detector_product_bytes,
        packed_resident_bytes=index.packed_resident_bytes,
        product_resident_bytes=product_resident_bytes,
        total_resident_bytes=total_resident_bytes,
        maximum_transient_bytes=(
            index.file_bytes
            if mapped_authentication_bytes
            else max(prepared_dpc_bytes, prepared_detector_product_bytes)
        ),
        device_allocated_bytes_before=allocated_before,
        device_allocated_bytes_after=int(device.currentAllocatedSize()),
    )
    return MPSCompactV3Resident(
        index=index,
        load_metrics=metrics,
        Metal=Metal,
        device=device,
        queue=queue,
        validation_pipeline=validation_pipeline,
        selected_pipeline=selected_pipeline,
        detector_pipeline=detector_pipeline,
        full_decode_pipeline=full_decode_pipeline,
        detector_sum_pipeline=detector_sum_pipeline,
        payload_buffers=payload_buffers,
        header_buffers=header_buffers,
        excluded_buffer=excluded_buffer,
        maximum_widths=maximum_widths,
        detector_outputs=detector_outputs,
        detector_entry_buffer=detector_entry_buffer,
        diffraction_buffer=diffraction_buffer,
        detector_sum_buffer=detector_sum_buffer,
        prepared_dpc_moment_buffer=prepared_dpc_moment_buffer,
        prepared_dpc_outputs=prepared_dpc_outputs,
        prepared_detector_products=prepared_detector_products,
    )
