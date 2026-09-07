"""Scientific workflow tests for compact 4D-STEM HDF5 random access."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.io._compact_h5 import (
    CompactH5Index,
    CompactH5ReferenceDecoder,
    prepare_compact_h5_metadata_copy,
)


def _raw_lz4_literals(source: bytes) -> bytes:
    literal_count = len(source)
    token = min(literal_count, 15) << 4
    result = bytearray([token])
    if literal_count >= 15:
        remaining = literal_count - 15
        while remaining >= 255:
            result.append(255)
            remaining -= 255
        result.append(remaining)
    result.extend(source)
    return bytes(result)


def _pack_detector_major(values: np.ndarray) -> tuple[bytes, bytes]:
    scan_count, detector_pixels = values.shape
    assert scan_count <= 128
    widths = bytearray()
    words: list[int] = []
    for pixel in range(detector_pixels):
        column = values[:, pixel]
        width = int(np.bitwise_or.reduce(column)).bit_length()
        widths.append(width)
        tile_words = [0] * (4 * width)
        if width == 0:
            continue
        for scan, item in enumerate(column):
            bit = scan * width
            word, shift = divmod(bit, 32)
            tile_words[word] |= int(item) << shift
            if shift + width > 32:
                tile_words[word + 1] |= int(item) >> (32 - shift)
        words.extend(word & 0xFFFFFFFF for word in tile_words)
    return bytes(widths), struct.pack(f"<{len(words)}I", *words)


def _write_fixture(
    path: Path,
    values: np.ndarray,
    *,
    include_calibration: bool = True,
    shape: tuple[int, int, int, int] = (2, 3, 2, 2),
    masked_pixels: tuple[int, ...] = (2,),
) -> None:
    scan_count = shape[0] * shape[1]
    assert values.shape == (scan_count, shape[2] * shape[3])
    widths, decoded = _pack_detector_major(values)
    chunks = [decoded[start : start + 128] for start in range(0, len(decoded), 128)]
    compressed_chunks = [_raw_lz4_literals(chunk) for chunk in chunks]
    compressed = b"".join(compressed_chunks)
    lengths = bytes(len(chunk) - 1 for chunk in compressed_chunks)
    payload_offset = 65536
    lengths_offset = payload_offset + len(compressed)
    widths_offset = lengths_offset + len(lengths)
    source_identity = hashlib.sha256(values.tobytes()).digest()
    manifest = {
        "schema": "quantem.gpu.packed-detector-h5/v1",
        "status": "complete",
        "source_identity_sha256": source_identity.hex(),
        "source_raw_logical_sha256": source_identity.hex(),
        "source_shape": list(shape),
        "source_dtype": "uint16",
        "working_dtype": "uint16",
        "working_value_definition": "exact uint16 source counts",
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "shard_count": 1,
        "scans_per_shard": scan_count,
        "payload_chunk_bytes": 128,
        "payload_chunk_codec": "independent raw LZ4 blocks",
        "payload_chunk_length_codec": "uint8 encoded_bytes_minus_one",
        "descriptor_codec": "uint8 five-bit widths",
        "masked_detector_pixels": [
            list(divmod(pixel, shape[3])) for pixel in masked_pixels
        ],
        "shards": [
            {
                "ordinal": 0,
                "scan_count": scan_count,
                "payload_file_offset": payload_offset,
                "payload_file_bytes": len(compressed),
                "payload_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "lengths_file_offset": lengths_offset,
                "lengths_file_bytes": len(lengths),
                "descriptor_widths_file_offset": widths_offset,
                "descriptor_widths_file_bytes": len(widths),
                "descriptor_widths_sha256": hashlib.sha256(widths).hexdigest(),
                "payload_decoded_bytes": len(decoded),
                "descriptor_count": len(widths),
                "payload_chunk_count": len(chunks),
                "payload_decoded_sha256": hashlib.sha256(decoded).hexdigest(),
            }
        ],
    }
    if include_calibration:
        manifest["detector_calibration"] = {
            "schema": "quantem.gpu.detector-calibration/v1",
            "source_identity_sha256": source_identity.hex(),
            "detector_center_px": [0.75, 1.25],
            "bright_field_radius_px": 1.5,
            "dpc_rotation_degrees": 176.25,
            "dpc_component_order_exchanged": False,
            "method": "test-fixture",
        }
    header = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    binary = bytearray(
        struct.pack("<8sIIIIIII", b"QGIX\0\0\0\1", 1, 128, *shape, scan_count)
    )
    binary.extend(struct.pack("<I", len(masked_pixels)))
    binary.extend(struct.pack(f"<{len(masked_pixels)}I", *masked_pixels))
    binary.extend(source_identity)
    binary.extend(
        struct.pack(
            "<QQQQQQQII32s",
            payload_offset,
            len(compressed),
            lengths_offset,
            len(lengths),
            widths_offset,
            len(widths),
            len(decoded),
            len(widths),
            len(chunks),
            hashlib.sha256(decoded).digest(),
        )
    )
    binary_offset = (24 + len(header) + 7) & ~7
    user_block = bytearray(65536)
    user_block[:24] = struct.pack(
        "<8sIIII",
        b"QGPUH5\0\1",
        len(header),
        zlib.crc32(header),
        binary_offset,
        len(binary),
    )
    user_block[24 : 24 + len(header)] = header
    user_block[binary_offset : binary_offset + len(binary)] = binary
    path.write_bytes(user_block + compressed + lengths + widths)


def _write_v3_fixture(
    path: Path,
    values: np.ndarray,
    *,
    detector_shape: tuple[int, int],
    masked_pixels: tuple[int, ...] = (),
    include_masked_pixel_sha256: bool = True,
    masked_pixel_sha256: str | None = None,
    include_masked_raw_values: bool = True,
    masked_raw_values: tuple[object, ...] | None = None,
    scan_shape: tuple[int, int] | None = None,
    header_encoding: int = 1,
) -> dict[str, int]:
    """Write an independent one-shard QGIX v3 fixture for parser tests."""
    scan_count, detector_pixels = values.shape
    assert detector_pixels == detector_shape[0] * detector_shape[1]
    assert scan_count % 32 == 0
    working = values.copy()
    if masked_pixels:
        working[:, list(masked_pixels)] = 0
    assert header_encoding in (1, 2)
    assert int(working.max(initial=0)) <= (255 if header_encoding == 1 else 65535)
    tile_count = scan_count // 32
    widths = np.zeros((detector_pixels, tile_count), dtype=np.uint8)
    pixel_payloads: list[list[int]] = []
    for pixel in range(detector_pixels):
        pixel_words: list[int] = []
        for tile in range(tile_count):
            column = working[tile * 32 : (tile + 1) * 32, pixel]
            width = int(np.bitwise_or.reduce(column)).bit_length()
            if header_encoding == 2 and width == 15:
                width = 16
            widths[pixel, tile] = width
            tile_words = [0] * width
            for scan, item in enumerate(column):
                if width == 0:
                    break
                bit = scan * width
                word, shift = divmod(bit, 32)
                tile_words[word] |= int(item) << shift
                if shift + width > 32:
                    tile_words[word + 1] |= int(item) >> (32 - shift)
            pixel_words.extend(word & 0xFFFFFFFF for word in tile_words)
        pixel_payloads.append(pixel_words)
    payload_words = [word for pixel in pixel_payloads for word in pixel]
    payload = struct.pack(f"<{len(payload_words)}I", *payload_words)

    checkpoint_words = (tile_count + 31) // 32
    width_words = (tile_count + 7) // 8
    header_words_per_pixel = checkpoint_words + width_words
    headers = np.zeros((detector_pixels, header_words_per_pixel), dtype="<u4")
    pixel_sizes = widths.sum(axis=1, dtype=np.uint64)
    if detector_pixels > 1:
        np.cumsum(pixel_sizes[:-1], dtype=np.uint64, out=headers[1:, 0])
    cumulative = np.zeros((detector_pixels, tile_count + 1), dtype=np.uint32)
    np.cumsum(widths, axis=1, dtype=np.uint32, out=cumulative[:, 1:])
    for checkpoint in range(1, checkpoint_words):
        headers[:, checkpoint] = cumulative[:, checkpoint * 32]
    for tile in range(tile_count):
        codes = np.minimum(widths[:, tile], 15).astype(np.uint32)
        headers[:, checkpoint_words + tile // 8] |= codes << ((tile % 8) * 4)
    header_payload = headers.tobytes()

    shape = (*(scan_shape or (1, scan_count)), *detector_shape)
    source_identity = hashlib.sha256(values.tobytes()).digest()
    raw_sha256 = hashlib.sha256(values.tobytes()).hexdigest()
    prepared_sha256 = hashlib.sha256(working.astype(np.uint8).tobytes()).hexdigest()
    mask_bytes = struct.pack(f"<{len(masked_pixels)}I", *masked_pixels)
    payload_offset = 65536
    headers_offset = payload_offset + len(payload)
    manifest = {
        "schema": "quantem.gpu.packed-detector-h5/v3",
        "status": "complete",
        "payload_codec": "direct-bitpacked-u32",
        "source_identity_sha256": source_identity.hex(),
        "source_raw_logical_sha256": raw_sha256,
        "source_shape": list(shape),
        "source_dtype": "uint16",
        "working_dtype": "uint8",
        "working_value_definition": (
            "all admitted source counts exactly; authenticated dead pixels set to zero"
        ),
        "prepared_uint8_sha256": prepared_sha256,
        "detector_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "masked_detector_pixels": list(masked_pixels),
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "scan_tile": 32,
        "shard_count": 1,
    }
    if include_masked_pixel_sha256:
        manifest["masked_detector_pixels_sha256"] = (
            masked_pixel_sha256
            if masked_pixel_sha256 is not None
            else hashlib.sha256(mask_bytes).hexdigest()
        )
    if include_masked_raw_values:
        if masked_raw_values is None:
            raw_values = []
            for pixel in masked_pixels:
                assert np.all(values[:, pixel] == values[0, pixel])
                raw_values.append(int(values[0, pixel]))
        else:
            raw_values = list(masked_raw_values)
        manifest["masked_detector_raw_values"] = raw_values
    if header_encoding == 2:
        manifest["working_dtype"] = "uint16"
        manifest.pop("prepared_uint8_sha256")
        manifest["working_logical_sha256"] = hashlib.sha256(
            working.astype("<u2").tobytes()
        ).hexdigest()
    header = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    binary = bytearray(
        struct.pack(
            "<8sIIIIIIIII",
            b"QGIX\0\0\0\3",
            1,
            0,
            *shape,
            scan_count,
            32,
            header_encoding,
        )
    )
    binary.extend(struct.pack("<I", len(masked_pixels)))
    binary.extend(mask_bytes)
    binary.extend(source_identity)
    binary.extend(
        struct.pack(
            "<QQQQQQQII32s",
            payload_offset,
            len(payload),
            0,
            0,
            headers_offset,
            len(header_payload),
            len(payload),
            headers.size,
            0,
            hashlib.sha256(payload).digest(),
        )
    )
    binary_offset = (24 + len(header) + 7) & ~7
    user_block = bytearray(65536)
    user_block[:24] = struct.pack(
        "<8sIIII",
        b"QGPUH5\0\1",
        len(header),
        zlib.crc32(header),
        binary_offset,
        len(binary),
    )
    user_block[24 : 24 + len(header)] = header
    user_block[binary_offset : binary_offset + len(binary)] = binary
    path.write_bytes(user_block + payload + header_payload)
    return {
        "binary_offset": binary_offset,
        "headers_offset": headers_offset,
        "header_words_per_pixel": header_words_per_pixel,
        "checkpoint_words": checkpoint_words,
        "width_words": width_words,
        "payload_offset": payload_offset,
    }


def test_reference_decoder_preserves_row_column_order_and_mask(tmp_path: Path) -> None:
    values = np.array(
        [
            [0, 1, 0, 3],
            [1, 2, 0, 4],
            [0, 3, 0, 7],
            [1, 0, 0, 8],
            [0, 1, 0, 511],
            [1, 2, 0, 9],
        ],
        dtype=np.uint16,
    )
    path = tmp_path / "compact-fixture.h5"
    _write_fixture(path, values)

    index = CompactH5Index.from_file(path)
    decoder = CompactH5ReferenceDecoder(index)
    decoder.validate_shard_metadata(0)

    assert index.shape == (2, 3, 2, 2)
    assert index.detector_calibration == {
        "schema": "quantem.gpu.detector-calibration/v1",
        "source_identity_sha256": index.source_identity_sha256,
        "detector_center_px": [0.75, 1.25],
        "bright_field_radius_px": 1.5,
        "dpc_rotation_degrees": 176.25,
        "dpc_component_order_exchanged": False,
        "method": "test-fixture",
    }
    assert index.logical_source_bytes == values.size * 2
    assert index.resident_bytes > values.nbytes
    for scan_row in range(2):
        for scan_column in range(3):
            scan = scan_row * 3 + scan_column
            for detector_row in range(2):
                for detector_column in range(2):
                    pixel = detector_row * 2 + detector_column
                    assert decoder.value(
                        scan_row,
                        scan_column,
                        detector_row,
                        detector_column,
                    ) == int(values[scan, pixel])
    assert decoder.value(1, 2, 1, 0) == 0
    assert decoder.value(1, 1, 1, 1) == 511


def test_detector_calibration_remains_optional_for_existing_v1_files(
    tmp_path: Path,
) -> None:
    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    path = tmp_path / "compact-without-calibration.h5"
    _write_fixture(path, values, include_calibration=False)

    assert CompactH5Index.from_file(path).detector_calibration is None


def test_detector_calibration_must_match_source_identity(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    path = tmp_path / "compact-wrong-calibration.h5"
    _write_fixture(path, values)
    changed = bytearray(path.read_bytes())
    header_bytes = struct.unpack_from("<I", changed, 8)[0]
    manifest = json.loads(changed[24 : 24 + header_bytes])
    manifest["detector_calibration"]["source_identity_sha256"] = "0" * 64
    header = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    assert len(header) == header_bytes
    struct.pack_into("<I", changed, 12, zlib.crc32(header))
    changed[24 : 24 + header_bytes] = header
    path.write_bytes(changed)

    with pytest.raises(ValueError, match="belongs to a different source"):
        CompactH5Index.from_file(path)


def test_prepare_metadata_copy_preserves_source_and_authenticates_envelope(
    tmp_path: Path,
) -> None:
    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    source = tmp_path / "compact-source.h5"
    destination = tmp_path / "compact-prepared.h5"
    _write_fixture(source, values, include_calibration=False)
    original = source.read_bytes()

    prepared = prepare_compact_h5_metadata_copy(
        source,
        destination,
        detector_calibration={
            "detector_center_px": [0.75, 1.25],
            "bright_field_radius_px": 1.5,
            "dpc_rotation_degrees": 176.25,
            "dpc_component_order_exchanged": False,
            "method": "test-preparation",
        },
    )

    assert source.read_bytes() == original
    assert destination.stat().st_size == source.stat().st_size
    assert prepared.detector_calibration == {
        "schema": "quantem.gpu.detector-calibration/v1",
        "source_identity_sha256": prepared.source_identity_sha256,
        "detector_center_px": [0.75, 1.25],
        "bright_field_radius_px": 1.5,
        "dpc_rotation_degrees": 176.25,
        "dpc_component_order_exchanged": False,
        "method": "test-preparation",
    }
    record = prepared.manifest["shards"][0]
    shard = prepared.shards[0]
    envelope_start = min(
        shard.payload_offset,
        shard.lengths_offset,
        shard.widths_offset,
    )
    envelope_end = max(
        shard.payload_offset + shard.payload_bytes,
        shard.lengths_offset + shard.lengths_bytes,
        shard.widths_offset + shard.widths_bytes,
    )
    assert (
        record["encoded_envelope_sha256"]
        == hashlib.sha256(
            destination.read_bytes()[envelope_start:envelope_end]
        ).hexdigest()
    )
    decoder = CompactH5ReferenceDecoder(prepared)
    assert decoder.value(1, 1, 1, 1) == int(values[4, 3])

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_compact_h5_metadata_copy(source, destination)


def test_parser_rejects_changed_contract_metadata(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.uint16).reshape(6, 4)
    path = tmp_path / "compact-fixture.h5"
    _write_fixture(path, values)
    changed = bytearray(path.read_bytes())
    changed[32] ^= 1
    path.write_bytes(changed)

    with pytest.raises(ValueError, match="CRC-32"):
        CompactH5Index.from_file(path)


def test_v3_uint16_preserves_counts_across_checkpoints(tmp_path: Path) -> None:
    """Retain real wide counts and masked constants across every tile boundary."""
    values = np.zeros((1280, 6), dtype=np.uint16)
    for tile in range(40):
        width = (0, 8, 9, 14, 15, 16)[tile % 6]
        values[tile * 32 : (tile + 1) * 32, :5] = (1 << width) - 1
    values[:6, 0] = [254, 255, 256, 32767, 32768, 65535]
    values[:, 5] = 65535
    path = tmp_path / "direct-v3-uint16.h5"
    _write_v3_fixture(
        path, values, detector_shape=(2, 3), masked_pixels=(5,), header_encoding=2
    )
    index = CompactH5Index.from_file(path)
    decoder = CompactH5ReferenceDecoder(index)
    decoder.validate_shard_metadata(0)
    decoder.validate_shard_payload(0)
    assert index.header_encoding == 2
    for scan in range(1280):
        np.testing.assert_array_equal(
            decoder.raw_diffraction(0, scan), values[scan].reshape(2, 3)
        )
    for selected in ([0, 1], [1, 2, 3], list(range(6)), []):
        mask = np.zeros(6, dtype=np.uint8)
        mask[selected] = 1
        working = values.copy()
        working[:, 5] = 0
        expected = working[:, selected].sum(axis=1, dtype=np.uint64)
        np.testing.assert_array_equal(
            decoder.detector_sum(mask.reshape(2, 3)).ravel(), expected
        )


def test_v3_reference_decoder_preserves_exact_values_mask_and_products(
    tmp_path: Path,
) -> None:
    scans = np.arange(64, dtype=np.uint16)
    values = np.stack(
        [
            scans % 2,
            (scans * 13 + 17) % 256,
            np.full_like(scans, 65535),
            (scans * 3) % 127,
            np.zeros_like(scans),
            (scans * 5 + 11) % 64,
        ],
        axis=1,
    )
    path = tmp_path / "direct-v3.h5"
    _write_v3_fixture(path, values, detector_shape=(2, 3), masked_pixels=(2,))

    index = CompactH5Index.from_file(path)
    decoder = CompactH5ReferenceDecoder(index)
    decoder.validate_shard_metadata(0)
    decoder.validate_shard_payload(0)

    assert index.schema_version == 3
    assert index.scan_tile == 32
    assert index.header_encoding == 1
    assert index.shape == (1, 64, 2, 3)
    assert (
        index.masked_detector_pixels_sha256
        == hashlib.sha256(struct.pack("<I", 2)).hexdigest()
    )
    assert index.masked_detector_raw_values == (65535,)
    assert index.raw_reconstruction_available
    assert (
        index.resident_bytes
        == index.shards[0].payload_bytes + index.shards[0].widths_bytes
    )
    working = values.copy()
    working[:, 2] = 0
    for scan_column in (0, 4, 10, 31, 32, 47, 63):
        for detector_pixel in range(6):
            detector_row, detector_column = divmod(detector_pixel, 3)
            assert decoder.value(0, scan_column, detector_row, detector_column) == int(
                working[scan_column, detector_pixel]
            )
        assert np.array_equal(
            decoder.selected_diffraction(0, scan_column),
            working[scan_column].reshape(2, 3),
        )
        assert decoder.raw_value(0, scan_column, 0, 2) == 65535
        assert np.array_equal(
            decoder.raw_diffraction(0, scan_column),
            values[scan_column].reshape(2, 3),
        )
    detector_mask = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.uint8)
    expected_sum = working[:, [0, 1, 3, 5]].sum(axis=1, dtype=np.uint32)
    assert np.array_equal(decoder.detector_sum(detector_mask).ravel(), expected_sum)


@pytest.mark.parametrize(
    ("include_masked_pixel_sha256", "include_masked_raw_values"),
    [(False, False), (True, False), (False, True)],
)
def test_v3_legacy_missing_portability_fields_is_explicitly_nonportable(
    tmp_path: Path,
    include_masked_pixel_sha256: bool,
    include_masked_raw_values: bool,
) -> None:
    scans = np.arange(64, dtype=np.uint16)
    values = np.stack([scans % 17, np.full_like(scans, 65535)], axis=1)
    path = tmp_path / "legacy-nonportable-v3.h5"
    _write_v3_fixture(
        path,
        values,
        detector_shape=(1, 2),
        masked_pixels=(1,),
        include_masked_pixel_sha256=include_masked_pixel_sha256,
        include_masked_raw_values=include_masked_raw_values,
    )

    index = CompactH5Index.from_file(path)
    decoder = CompactH5ReferenceDecoder(index)
    assert (index.masked_detector_pixels_sha256 is not None) is (
        include_masked_pixel_sha256
    )
    assert (index.masked_detector_raw_values is not None) is include_masked_raw_values
    assert not index.raw_reconstruction_available
    assert decoder.value(0, 19, 0, 1) == 0
    with pytest.raises(ValueError, match="admissible only for explicit mask-applied"):
        index.require_raw_reconstruction()
    with pytest.raises(ValueError, match="masked_detector_pixels_sha256"):
        decoder.raw_value(0, 19, 0, 1)


@pytest.mark.parametrize(
    "raw_values",
    [(), (65536,), (True,)],
)
def test_v3_raw_constants_must_align_and_be_exact_uint16(
    tmp_path: Path, raw_values: tuple[object, ...]
) -> None:
    scans = np.arange(64, dtype=np.uint16)
    values = np.stack([scans % 17, np.full_like(scans, 65535)], axis=1)
    path = tmp_path / "invalid-raw-constants-v3.h5"
    _write_v3_fixture(
        path,
        values,
        detector_shape=(1, 2),
        masked_pixels=(1,),
        masked_raw_values=raw_values,
    )

    with pytest.raises(ValueError, match="aligned one-to-one"):
        CompactH5Index.from_file(path)


def test_v3_mask_and_raw_constants_use_ordered_row_major_pixel_indices(
    tmp_path: Path,
) -> None:
    scans = np.arange(64, dtype=np.uint16)
    values = np.stack(
        [scans % 17, np.full_like(scans, 60000), np.full_like(scans, 61000)],
        axis=1,
    )
    path = tmp_path / "unordered-mask-v3.h5"
    _write_v3_fixture(
        path,
        values,
        detector_shape=(1, 3),
        masked_pixels=(2, 1),
        masked_raw_values=(61000, 60000),
    )

    with pytest.raises(ValueError, match="ordered row-major"):
        CompactH5Index.from_file(path)


def test_v3_masked_pixel_digest_binds_ordered_binary_index(tmp_path: Path) -> None:
    scans = np.arange(64, dtype=np.uint16)
    values = np.stack([scans % 17, np.full_like(scans, 65535)], axis=1)
    path = tmp_path / "wrong-masked-pixel-digest-v3.h5"
    _write_v3_fixture(
        path,
        values,
        detector_shape=(1, 2),
        masked_pixels=(1,),
        masked_pixel_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="does not match the ordered binary-index"):
        CompactH5Index.from_file(path)


def test_v3_prepared_copy_adds_portable_raw_contract_without_mutating_source(
    tmp_path: Path,
) -> None:
    scans = np.arange(64, dtype=np.uint16)
    values = np.stack([scans % 17, np.full_like(scans, 65535)], axis=1)
    source = tmp_path / "legacy-nonportable-v3.h5"
    destination = tmp_path / "portable-v3.h5"
    _write_v3_fixture(
        source,
        values,
        detector_shape=(1, 2),
        masked_pixels=(1,),
        include_masked_pixel_sha256=False,
        include_masked_raw_values=False,
    )
    original = source.read_bytes()
    source_sha256 = hashlib.sha256(original).hexdigest()

    prepared = prepare_compact_h5_metadata_copy(
        source,
        destination,
        masked_detector_raw_values=(65535,),
        expected_source_sha256=source_sha256,
    )

    assert source.read_bytes() == original
    assert destination.stat().st_size == source.stat().st_size
    assert prepared.raw_reconstruction_available
    assert prepared.masked_detector_raw_values == (65535,)
    assert (
        prepared.masked_detector_pixels_sha256
        == hashlib.sha256(struct.pack("<I", 1)).hexdigest()
    )
    reopened = CompactH5Index.from_file(destination)
    reopened.require_raw_reconstruction()
    decoder = CompactH5ReferenceDecoder(reopened)
    assert decoder.value(0, 19, 0, 1) == 0
    assert decoder.raw_value(0, 19, 0, 1) == 65535

    wrong_hash_destination = tmp_path / "wrong-hash-v3.h5"
    with pytest.raises(ValueError, match="input SHA-256"):
        prepare_compact_h5_metadata_copy(
            source,
            wrong_hash_destination,
            masked_detector_raw_values=(65535,),
            expected_source_sha256="0" * 64,
        )
    assert not wrong_hash_destination.exists()


def test_cuda_rejects_uint16_checkpoint_headers_before_allocation(
    tmp_path, monkeypatch
):
    """A shared parser must not silently enable an unsupported GPU decoder."""
    from quantem.gpu.io.backends.cuda import packed

    values = np.tile(np.array([[0, 65535], [256, 32768]], dtype=np.uint16), (16, 1))
    path = tmp_path / "exact-u16.h5"
    _write_v3_fixture(
        path, values, detector_shape=(1, 2), masked_pixels=(), header_encoding=2
    )
    # Any attempted CuPy access fails: rejection belongs before device work.
    monkeypatch.setattr(packed, "cp", object())
    with pytest.raises(ValueError, match="encoding 2.*native Swift/Metal"):
        packed.load_compact_h5_cuda(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("base", "pixel bases"),
        ("checkpoint", "checkpoint"),
        ("width", "permits at most 8"),
        ("tail", "tail width nibbles"),
        ("masked", "excluded detector pixel"),
    ],
)
def test_v3_parser_rejects_malformed_compact_headers(
    tmp_path: Path, mutation: str, message: str
) -> None:
    scan_count = 1056
    scans = np.arange(scan_count, dtype=np.uint16)
    values = np.stack(
        [scans % 2, (scans * 3) % 127, np.full_like(scans, 65535)], axis=1
    )
    path = tmp_path / f"malformed-{mutation}.h5"
    locations = _write_v3_fixture(
        path, values, detector_shape=(1, 3), masked_pixels=(2,)
    )
    changed = bytearray(path.read_bytes())
    headers_offset = locations["headers_offset"]
    words_per_pixel = locations["header_words_per_pixel"]
    checkpoint_words = locations["checkpoint_words"]
    if mutation == "base":
        struct.pack_into("<I", changed, headers_offset, 1)
    elif mutation == "checkpoint":
        struct.pack_into("<I", changed, headers_offset + 4, 31)
    elif mutation == "width":
        word_offset = headers_offset + checkpoint_words * 4
        packed = struct.unpack_from("<I", changed, word_offset)[0]
        struct.pack_into("<I", changed, word_offset, (packed & ~0xF) | 9)
    elif mutation == "tail":
        last_width_word = checkpoint_words + locations["width_words"] - 1
        word_offset = headers_offset + last_width_word * 4
        packed = struct.unpack_from("<I", changed, word_offset)[0]
        struct.pack_into("<I", changed, word_offset, packed | (1 << 4))
    else:
        masked_header = headers_offset + 2 * words_per_pixel * 4
        word_offset = masked_header + checkpoint_words * 4
        struct.pack_into("<I", changed, word_offset, 1)
    path.write_bytes(changed)

    decoder = CompactH5ReferenceDecoder(CompactH5Index.from_file(path))
    with pytest.raises(ValueError, match=message):
        decoder.validate_shard_metadata(0)


def test_v3_payload_digest_and_incomplete_tile_fail_closed(tmp_path: Path) -> None:
    values = np.arange(64 * 2, dtype=np.uint16).reshape(64, 2) % 251
    path = tmp_path / "direct-v3.h5"
    locations = _write_v3_fixture(path, values, detector_shape=(1, 2))
    changed = bytearray(path.read_bytes())
    changed[locations["payload_offset"]] ^= 1
    path.write_bytes(changed)
    decoder = CompactH5ReferenceDecoder(CompactH5Index.from_file(path))
    with pytest.raises(ValueError, match="failed SHA-256"):
        decoder.validate_shard_payload(0)

    tail = tmp_path / "direct-v3-tail.h5"
    _write_v3_fixture(tail, values[:32], detector_shape=(1, 2))
    changed = bytearray(tail.read_bytes())
    header_bytes = struct.unpack_from("<I", changed, 8)[0]
    manifest = json.loads(changed[24 : 24 + header_bytes])
    manifest["source_shape"][1] = 31
    replacement = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    assert len(replacement) == header_bytes
    struct.pack_into("<I", changed, 12, zlib.crc32(replacement))
    changed[24 : 24 + header_bytes] = replacement
    binary_offset = struct.unpack_from("<I", changed, 16)[0]
    struct.pack_into("<I", changed, binary_offset + 20, 31)
    struct.pack_into("<I", changed, binary_offset + 32, 31)
    tail.write_bytes(changed)
    with pytest.raises(ValueError, match="complete 32-scan tiles"):
        CompactH5Index.from_file(tail)
