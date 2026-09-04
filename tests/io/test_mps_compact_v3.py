"""Exact and fail-closed tests for the accepted isolated MPS QGIX v3 backend."""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.io.backends.mps.compact_v3 import (
    MPSCompactV3Cancelled,
    MPSCompactV3Error,
    load_compact_v3_mps,
    read_compact_v3_index,
)
from quantem.gpu.io.backends.mps.resident_dpc import (
    MPSDPCConfiguration,
    MPSDPCProcessor,
)


def _calibration_digest_for_fixture(value: object) -> str:
    def normalize(entry: object) -> object:
        if isinstance(entry, bool) or entry is None or isinstance(entry, str):
            return entry
        if isinstance(entry, (int, float)):
            return f"f64be:{struct.pack('>d', float(entry)).hex()}"
        if isinstance(entry, list):
            return [normalize(item) for item in entry]
        if isinstance(entry, dict):
            return {str(key): normalize(item) for key, item in entry.items()}
        raise TypeError(type(entry))

    return hashlib.sha256(
        json.dumps(normalize(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixture(
    path: Path,
    *,
    raw_exclusions: bool = False,
    partial_raw_exclusions: bool = False,
    prepared_dpc: bool = False,
    prepared_dpc_overrides: dict[str, object] | None = None,
    prepared_detector_products: bool = False,
    prepared_detector_overrides: dict[str, object] | None = None,
    prepared_detector_product_overrides: dict[str, dict[str, object]] | None = None,
) -> tuple[np.ndarray, int, int]:
    tile_widths = np.asarray(
        [
            [2, 3, 4, 5],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [0, 0, 0, 0],
            [8, 7, 8, 6],
            [2, 2, 2, 2],
        ],
        dtype=np.uint8,
    )
    values = np.zeros((6, 128), dtype=np.uint32)
    payload_words: list[int] = []
    headers: list[int] = []
    for pixel, widths in enumerate(tile_widths):
        headers.append(len(payload_words))
        packed_widths = 0
        for tile, width_value in enumerate(widths):
            width = int(width_value)
            packed_widths |= width << (tile * 4)
            tile_base = len(payload_words)
            payload_words.extend([0] * width)
            for local_scan in range(32):
                scan = tile * 32 + local_scan
                value = (
                    0
                    if pixel == 3 or width == 0
                    else (scan * (pixel + 3) + pixel) % (1 << width)
                )
                values[pixel, scan] = value
                if width:
                    bit = local_scan * width
                    word = tile_base + bit // 32
                    shift = bit % 32
                    payload_words[word] = (
                        payload_words[word] | (value << shift)
                    ) & 0xFFFF_FFFF
                    if shift + width > 32:
                        payload_words[word + 1] = (
                            payload_words[word + 1] | (value >> (32 - shift))
                        ) & 0xFFFF_FFFF
        headers.append(packed_widths)
    payload = np.asarray(payload_words, dtype="<u4").tobytes()
    header_data = np.asarray(headers, dtype="<u4").tobytes()
    logical = values.T.astype(np.uint8, copy=False).tobytes()
    source_identity = bytes(range(32, 64))
    manifest = {
        "schema": "quantem.gpu.packed-detector-h5/v3",
        "status": "complete",
        "payload_codec": "direct-bitpacked-u32",
        "source_identity_sha256": source_identity.hex(),
        "source_raw_logical_sha256": "d" * 64,
        "source_shape": [8, 16, 2, 3],
        "source_dtype": "uint16",
        "working_dtype": "uint8",
        "working_value_definition": (
            "all admitted source counts exactly; authenticated dead pixels set to zero"
        ),
        "prepared_uint8_sha256": hashlib.sha256(logical).hexdigest(),
        "detector_mask_sha256": "a" * 64,
        "masked_detector_pixels": [3],
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "scan_tile": 32,
        "shard_count": 1,
    }
    if raw_exclusions or partial_raw_exclusions:
        manifest["masked_detector_raw_values"] = [65535]
    if raw_exclusions:
        manifest["masked_detector_pixels_sha256"] = hashlib.sha256(
            struct.pack("<I", 3)
        ).hexdigest()
    binary_offset = 4096
    payload_offset = 8192
    headers_offset = payload_offset + len(payload)
    prepared_words = b""
    cursor = headers_offset + len(header_data)
    prepared_offset = cursor
    extra_ranges: list[tuple[int, bytes]] = []
    if prepared_dpc:
        selected_pixels = np.asarray([0, 1, 2, 4, 5])
        selected_values = values[selected_pixels].astype(np.uint64)
        total = selected_values.sum(axis=0, dtype=np.uint64)
        detector_rows = selected_pixels // 3
        detector_columns = selected_pixels % 3
        row_moment = (selected_values * detector_rows[:, None]).sum(
            axis=0, dtype=np.uint64
        )
        column_moment = (selected_values * detector_columns[:, None]).sum(
            axis=0, dtype=np.uint64
        )
        words = np.zeros((128, 8), dtype="<u4")
        for start, source_values in ((0, total), (2, row_moment), (4, column_moment)):
            words[:, start] = source_values & np.uint64(0xFFFF_FFFF)
            words[:, start + 1] = source_values >> np.uint64(32)
        prepared_words = words.tobytes()
        total_bound = 5 * 255
        row_bound = 255 * int(detector_rows.sum())
        column_bound = 255 * int(detector_columns.sum())
        contract: dict[str, object] = {
            "schema": "quantem.gpu.prepared-dpc-moments/v1",
            "source_identity_sha256": source_identity.hex(),
            "working_uint8_sha256": manifest["prepared_uint8_sha256"],
            "detector_mask_sha256": manifest["detector_mask_sha256"],
            "detector_selection": "all-nonexcluded-v1",
            "scan_count": 128,
            "selected_detector_pixels": 5,
            "detector_columns": 3,
            "dtype": "little-endian-u32",
            "word_order": "little-endian-u32-pairs",
            "words_per_scan": 8,
            "layout": [
                "total_lo",
                "total_hi",
                "row_lo",
                "row_hi",
                "column_lo",
                "column_hi",
                "padding_0",
                "padding_1",
            ],
            "file_offset": prepared_offset,
            "file_bytes": len(prepared_words),
            "sha256": hashlib.sha256(prepared_words).hexdigest(),
            "total_bound": str(total_bound),
            "row_moment_bound": str(row_bound),
            "column_moment_bound": str(column_bound),
            "narrow_integer": True,
            "narrow_products": True,
        }
        contract.update(prepared_dpc_overrides or {})
        manifest["prepared_dpc_moments"] = contract
        extra_ranges.append((prepared_offset, prepared_words))
        cursor += len(prepared_words)
    if prepared_detector_products:
        calibration = {
            "schema": "quantem.gpu.detector-calibration/v1",
            "source_identity_sha256": source_identity.hex(),
            "detector_center_px": [0.0, 0.0],
            "bright_field_radius_px": 1.1,
            "method": "synthetic-test",
        }
        manifest["detector_calibration"] = calibration
        radius = float(calibration["bright_field_radius_px"])
        center_row, center_column = calibration["detector_center_px"]
        detector_row, detector_column = np.indices((2, 3))
        distance = np.hypot(
            detector_row - float(center_row),
            detector_column - float(center_column),
        )
        geometries = {
            "bf": (0.0, radius),
            "abf": (0.5 * radius, radius),
            "adf": (radius, 2.0 * radius),
        }
        product_records: list[dict[str, object]] = []
        for name in ("bf", "abf", "adf"):
            inner, outer = geometries[name]
            mask = ((distance >= inner) & (distance <= outer)).astype(np.uint8)
            mask.reshape(-1)[3] = 0
            mask_bytes = mask.tobytes()
            selected = np.flatnonzero(mask.reshape(-1))
            product_values = values[selected].sum(axis=0, dtype=np.uint32)
            values_bytes = product_values.astype("<u4", copy=False).tobytes()
            mask_offset = cursor
            extra_ranges.append((mask_offset, mask_bytes))
            cursor += len(mask_bytes)
            cursor = (cursor + 3) & ~3
            values_offset = cursor
            extra_ranges.append((values_offset, values_bytes))
            cursor += len(values_bytes)
            product_record: dict[str, object] = {
                "name": name,
                "center_px": list(calibration["detector_center_px"]),
                "inner_radius_px": inner,
                "outer_radius_px": outer,
                "selected_detector_pixels": int(mask.sum(dtype=np.uint64)),
                "mask_file_offset": mask_offset,
                "mask_file_bytes": len(mask_bytes),
                "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                "values_file_offset": values_offset,
                "values_file_bytes": len(values_bytes),
                "values_sha256": hashlib.sha256(values_bytes).hexdigest(),
            }
            product_record.update(
                (prepared_detector_product_overrides or {}).get(name, {})
            )
            product_records.append(product_record)
        product_contract: dict[str, object] = {
            "schema": "quantem.gpu.prepared-detector-products/v1",
            "source_identity_sha256": source_identity.hex(),
            "working_uint8_sha256": manifest["prepared_uint8_sha256"],
            "detector_mask_sha256": manifest["detector_mask_sha256"],
            "detector_calibration_sha256": _calibration_digest_for_fixture(calibration),
            "detector_calibration_digest_encoding": (
                "canonical-json-numbers-as-f64be-hex/v1"
            ),
            "scan_shape": [8, 16],
            "detector_shape": [2, 3],
            "product_dtype": "little-endian-u32",
            "mask_dtype": "uint8-binary-row-major",
            "mask_rule": "quantem.gpu.detector-mask-inclusive/v1",
            "product_order": ["bf", "abf", "adf"],
            "products": product_records,
        }
        product_contract.update(prepared_detector_overrides or {})
        manifest["prepared_detector_products"] = product_contract
    metadata = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    binary = bytearray(
        struct.pack(
            "<8sIIIIIIIII",
            b"QGIX\0\0\0\3",
            1,
            0,
            8,
            16,
            2,
            3,
            128,
            32,
            1,
        )
    )
    binary.extend(struct.pack("<I", 1))
    binary.extend(struct.pack("<I", 3))
    binary.extend(source_identity)
    binary.extend(
        struct.pack(
            "<QQQQQQQII32s",
            payload_offset,
            len(payload),
            0,
            0,
            headers_offset,
            len(header_data),
            len(payload),
            len(headers),
            0,
            hashlib.sha256(payload).digest(),
        )
    )
    prelude = struct.pack(
        "<8sIIII",
        b"QGPUH5\0\1",
        len(metadata),
        zlib.crc32(metadata),
        binary_offset,
        len(binary),
    )
    assert len(prelude) + len(metadata) <= binary_offset
    file_bytes = bytearray(cursor)
    file_bytes[: len(prelude)] = prelude
    file_bytes[24 : 24 + len(metadata)] = metadata
    file_bytes[binary_offset : binary_offset + len(binary)] = binary
    file_bytes[payload_offset : payload_offset + len(payload)] = payload
    file_bytes[headers_offset : headers_offset + len(header_data)] = header_data
    for offset, data in extra_ranges:
        file_bytes[offset : offset + len(data)] = data
    path.write_bytes(file_bytes)
    return values, payload_offset, headers_offset


def test_compact_v3_api_remains_an_explicit_submodule() -> None:
    mps = importlib.import_module("quantem.gpu.io.backends.mps")
    compact_v3 = importlib.import_module("quantem.gpu.io.backends.mps.compact_v3")

    assert "load_compact_v3_mps" in compact_v3.__all__
    assert "load_compact_v3_mps" not in mps.__all__
    assert not hasattr(mps, "load_compact_v3_mps")


def test_python_mps_v3_parser_cross_checks_metadata(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-v3.h5"
    _, _, _ = _fixture(path)
    index = read_compact_v3_index(path)
    assert index.shape == (8, 16, 2, 3)
    assert index.scan_tile == 32
    assert index.embedded_scientific_semantics is True
    assert index.excluded_detector_pixels == (3,)
    assert index.raw_access_mode == "mask_applied_only_legacy"
    assert index.masked_detector_pixels_sha256 is None
    assert index.masked_detector_raw_values is None


def test_python_mps_v3_parser_accepts_bound_raw_exclusion_constants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic-v3-raw.h5"
    _, _, _ = _fixture(path, raw_exclusions=True)
    index = read_compact_v3_index(path)
    assert index.raw_access_mode == "exact_exclusion_constants"
    assert index.masked_detector_raw_values == (65535,)
    assert (
        index.masked_detector_pixels_sha256
        == hashlib.sha256(struct.pack("<I", 3)).hexdigest()
    )


def test_python_mps_v3_parser_accepts_source_bound_prepared_dpc(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic-v3-prepared-dpc.h5"
    _, _, _ = _fixture(path, prepared_dpc=True)

    prepared = read_compact_v3_index(path).prepared_dpc_moments

    assert prepared is not None
    assert prepared.scan_count == 128
    assert prepared.file_bytes == 128 * 8 * 4
    assert prepared.selected_detector_pixels == 5
    assert prepared.detector_columns == 3


def test_python_mps_v3_parser_accepts_prepared_detector_products(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic-v3-prepared-detectors.h5"
    _fixture(path, prepared_detector_products=True)

    prepared = read_compact_v3_index(path).prepared_detector_products

    assert prepared is not None
    assert [product.name for product in prepared.products] == ["bf", "abf", "adf"]
    assert all(product.values_file_bytes == 128 * 4 for product in prepared.products)
    assert all(product.mask_file_bytes == 6 for product in prepared.products)


@pytest.mark.parametrize(
    ("root_overrides", "product_overrides", "message"),
    [
        ({"working_uint8_sha256": "0" * 64}, {}, "working_uint8_sha256"),
        ({"detector_calibration_sha256": "0" * 64}, {}, "calibration"),
        ({"product_order": ["adf", "abf", "bf"]}, {}, "product_order"),
        ({}, {"bf": {"mask_file_offset": 8192}}, "overlap"),
        ({}, {"abf": {"outer_radius_px": 9.0}}, "geometry"),
    ],
)
def test_python_mps_v3_parser_rejects_mismatched_prepared_detector_products(
    tmp_path: Path,
    root_overrides: dict[str, object],
    product_overrides: dict[str, dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "synthetic-v3-mismatched-detectors.h5"
    _fixture(
        path,
        prepared_detector_products=True,
        prepared_detector_overrides=root_overrides,
        prepared_detector_product_overrides=product_overrides,
    )

    with pytest.raises(MPSCompactV3Error, match=message):
        read_compact_v3_index(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"working_uint8_sha256": "0" * 64}, "working_uint8_sha256"),
        ({"detector_mask_sha256": "0" * 64}, "detector_mask_sha256"),
        ({"file_bytes": 4}, "byte range"),
        ({"file_offset": 8192}, "overlaps shard 0 payload"),
        ({"layout": ["total_lo"]}, "layout"),
    ],
)
def test_python_mps_v3_parser_rejects_mismatched_prepared_dpc(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "synthetic-v3-mismatched-dpc.h5"
    _fixture(
        path,
        prepared_dpc=True,
        prepared_dpc_overrides=overrides,
    )

    with pytest.raises(MPSCompactV3Error, match=message):
        read_compact_v3_index(path)


def test_python_mps_v3_partial_raw_exclusion_fields_remain_nonportable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic-v3-partial-raw.h5"
    _, _, _ = _fixture(path, partial_raw_exclusions=True)
    index = read_compact_v3_index(path)
    assert index.raw_access_mode == "mask_applied_only_legacy"
    assert index.masked_detector_raw_values is None
    assert index.masked_detector_pixels_sha256 is None


@pytest.mark.parametrize(
    "authentication_policy", ["bounded_sequential", "parallel_mapped_full_file"]
)
def test_python_mps_v3_loads_and_interacts_exactly(
    tmp_path: Path, authentication_policy: str
) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3.h5"
    values, _, _ = _fixture(path)
    source = load_compact_v3_mps(path, authentication_policy=authentication_policy)
    try:
        assert np.array_equal(source.extract_diffraction(4, 13), values[:, 77])
        expected_sum = values.sum(axis=1, dtype=np.uint64)
        mean = source.mean_diffraction_pattern()
        np.testing.assert_array_equal(mean.detector_sum, expected_sum)
        np.testing.assert_array_equal(
            mean.mean,
            expected_sum.astype(np.float32) / np.float32(128),
        )
        assert mean.dispatch_count == 1
        assert mean.readback_bytes == 6 * 8
        cached_mean = source.mean_diffraction_pattern()
        np.testing.assert_array_equal(cached_mean.detector_sum, expected_sum)
        assert cached_mean.dispatch_count == 0
        assert cached_mean.wall_ms == 0
        assert cached_mean.gpu_ms == 0
        assert source.load_metrics.total_resident_bytes > (
            source.load_metrics.packed_resident_bytes
        )
        mask = np.asarray([1, 0, 1, 1, 1, 0], dtype=np.uint8)
        rebase = source.update_virtual_detector(mask)
        assert rebase.mode == "rebase"
        expected = values[[0, 2, 4]].sum(axis=0, dtype=np.uint32)
        assert np.array_equal(source.virtual_detector_values(), expected)
        translated = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.uint8)
        delta = source.update_virtual_detector(translated)
        assert delta.mode == "delta"
        assert delta.changed_detector_pixels == 2
        translated_expected = values[[1, 2, 4]].sum(axis=0, dtype=np.uint32)
        assert np.array_equal(source.virtual_detector_values(), translated_expected)
        fresh = source.update_virtual_detector(translated, force_rebase=True)
        assert fresh.mode == "rebase"
        assert np.array_equal(source.virtual_detector_values(), translated_expected)
        logical = source.hash_logical_working_u8()
        assert logical.sha256 == source.index.working_logical_sha256
    finally:
        source.release()


def test_python_mps_v3_primes_prepared_dpc_from_exact_moments(
    tmp_path: Path,
) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3-prepared-dpc.h5"
    values, _, _ = _fixture(path, prepared_dpc=True)
    source = load_compact_v3_mps(path)
    try:
        maps = source.prepared_dpc_values()
        assert maps is not None
        selected = np.asarray([0, 1, 2, 4, 5])
        selected_values = values[selected].astype(np.uint64)
        total = selected_values.sum(axis=0, dtype=np.uint64)
        row_moment = (selected_values * (selected // 3)[:, None]).sum(
            axis=0, dtype=np.uint64
        )
        column_moment = (selected_values * (selected % 3)[:, None]).sum(
            axis=0, dtype=np.uint64
        )
        expected_row = np.divide(
            row_moment,
            total,
            out=np.zeros(total.shape, dtype=np.float64),
            where=total != 0,
        ).astype(np.float32)
        expected_column = np.divide(
            column_moment,
            total,
            out=np.zeros(total.shape, dtype=np.float64),
            where=total != 0,
        ).astype(np.float32)
        expected_row -= float(np.mean(expected_row, dtype=np.float64))
        expected_column -= float(np.mean(expected_column, dtype=np.float64))
        np.testing.assert_array_equal(maps[0], expected_row)
        np.testing.assert_array_equal(maps[1], expected_column)
        moments = source.prepared_dpc_moment_values()
        assert moments is not None
        np.testing.assert_array_equal(moments.total, total)
        np.testing.assert_array_equal(moments.detector_row_moment, row_moment)
        np.testing.assert_array_equal(moments.detector_column_moment, column_moment)
        assert source.load_metrics.prepared_dpc_bytes == 128 * 8 * 4
        assert source.load_metrics.prepared_dpc_read_ms > 0
        assert source.load_metrics.prepared_dpc_authentication_ms > 0
        assert source.load_metrics.prepared_dpc_prime_ms > 0
    finally:
        source.release()


def test_python_mps_v3_prepared_dpc_connects_to_resident_idpc(
    tmp_path: Path,
) -> None:
    pytest.importorskip("Metal")
    from quantem.gpu.dpc.workflow import integrate

    path = tmp_path / "synthetic-v3-resident-idpc.h5"
    _fixture(path, prepared_dpc=True)
    source = load_compact_v3_mps(path)
    result = None
    try:
        row_buffer = source.prepared_dpc_display_buffer("row")
        column_buffer = source.prepared_dpc_display_buffer("column")
        maps = source.prepared_dpc_values()
        assert row_buffer is not None and column_buffer is not None
        assert maps is not None
        expected = integrate(maps[0].reshape(8, 16), maps[1].reshape(8, 16))
        result = MPSDPCProcessor().process_buffers(
            row_buffer,
            column_buffer,
            MPSDPCConfiguration(
                scan_rows=8,
                scan_columns=16,
                rotation_degrees=0,
            ),
        )
        phase = np.frombuffer(
            result.phase_buffer.contents().as_buffer(expected.nbytes),
            dtype=np.float32,
        ).reshape(expected.shape)
        np.testing.assert_allclose(phase, expected, rtol=0, atol=2e-5)
        assert result.metrics.upload_bytes == 0
        assert result.metrics.readback_bytes == 0
    finally:
        if result is not None:
            result.release()
        source.release()


def test_python_mps_v3_activates_authenticated_prepared_detector_products(
    tmp_path: Path,
) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3-prepared-detectors.h5"
    values, _, _ = _fixture(path, prepared_detector_products=True)
    source = load_compact_v3_mps(path)
    try:
        prepared = source.index.prepared_detector_products
        assert prepared is not None
        for product in prepared.products:
            metrics = source.activate_prepared_detector_product(product.name)
            mask = np.frombuffer(
                path.read_bytes(),
                dtype=np.uint8,
                count=product.mask_file_bytes,
                offset=product.mask_file_offset,
            )
            expected = values[np.flatnonzero(mask)].sum(axis=0, dtype=np.uint32)
            assert metrics.mode == "prepared"
            np.testing.assert_array_equal(source.virtual_detector_values(), expected)
        assert source.load_metrics.prepared_detector_product_bytes == 3 * (6 + 512)
        assert source.load_metrics.prepared_detector_product_read_ms > 0
        assert source.load_metrics.prepared_detector_product_authentication_ms > 0
    finally:
        source.release()


def test_python_mps_v3_rejects_changed_prepared_detector_values(
    tmp_path: Path,
) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3-prepared-detectors.h5"
    _fixture(path, prepared_detector_products=True)
    changed = bytearray(path.read_bytes())
    changed[-1] ^= 1
    path.write_bytes(changed)

    with pytest.raises(MPSCompactV3Error, match="prepared ADF values SHA-256"):
        load_compact_v3_mps(path)


def test_python_mps_v3_rejects_prepared_detector_selected_count_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3-prepared-detectors.h5"
    _fixture(
        path,
        prepared_detector_products=True,
        prepared_detector_product_overrides={"bf": {"selected_detector_pixels": 0}},
    )

    with pytest.raises(MPSCompactV3Error, match="BF mask selects"):
        load_compact_v3_mps(path)


def test_python_mps_v3_rejects_changed_prepared_dpc_bytes(tmp_path: Path) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3-prepared-dpc.h5"
    _fixture(path, prepared_dpc=True)
    changed = bytearray(path.read_bytes())
    changed[-1] ^= 1
    path.write_bytes(changed)

    with pytest.raises(MPSCompactV3Error, match="prepared DPC SHA-256"):
        load_compact_v3_mps(path)


def test_python_mps_v3_parallel_authentication_rejects_changed_payload(
    tmp_path: Path,
) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3.h5"
    _, payload_offset, _ = _fixture(path)
    changed = bytearray(path.read_bytes())
    changed[payload_offset] ^= 1
    path.write_bytes(changed)
    with pytest.raises(
        MPSCompactV3Error, match="parallel mapped payload authentication"
    ):
        load_compact_v3_mps(path, authentication_policy="parallel_mapped_full_file")


def test_python_mps_v3_gpu_validation_rejects_changed_header(tmp_path: Path) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3.h5"
    _, _, headers_offset = _fixture(path)
    changed = bytearray(path.read_bytes())
    changed[headers_offset + 4] ^= 1
    path.write_bytes(changed)
    with pytest.raises(MPSCompactV3Error, match="header validation status"):
        load_compact_v3_mps(path)


@pytest.mark.parametrize("cancel_at", [1, 4])
def test_python_mps_v3_cancelled_generation_never_publishes(
    tmp_path: Path, cancel_at: int
) -> None:
    pytest.importorskip("Metal")
    path = tmp_path / "synthetic-v3.h5"
    _, _, _ = _fixture(path)
    calls = 0

    def should_cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls == cancel_at

    with pytest.raises(MPSCompactV3Cancelled, match="superseded"):
        load_compact_v3_mps(path, should_cancel=should_cancel)
