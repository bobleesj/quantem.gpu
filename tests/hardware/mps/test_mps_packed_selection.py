"""Ordered selected-pattern and exact detector-state parity on physical Metal."""

import hashlib
import json
import struct
import zlib

import numpy as np
import pytest

from quantem.gpu.io.backends.mps import packed as mps
from tests.hardware.mps.test_mps_compact_v3 import _fixture


def _two_shard_fixture(path):
    """Independently pack two distinct shards, with a frozen integer oracle."""
    first, _, _ = _fixture(path, raw_exclusions=True)
    source = path.read_bytes()
    header_bytes = struct.unpack_from("<I", source, 8)[0]
    metadata = json.loads(source[24 : 24 + header_bytes])
    expected = np.concatenate([first.T, first[:, ::-1].T])
    metadata.update(
        source_shape=[16, 16, 2, 3],
        shard_count=2,
        prepared_uint8_sha256=hashlib.sha256(
            expected.astype(np.uint8).tobytes()
        ).hexdigest(),
    )
    payloads = []
    headers = []
    for values in (first, first[:, ::-1]):
        words = []
        descriptors = []
        for pixel in range(6):
            base = len(words)
            widths = 0
            for tile in range(4):
                samples = values[pixel, tile * 32 : (tile + 1) * 32]
                width = int(samples.max()).bit_length()
                widths |= width << (tile * 4)
                bits = sum(
                    int(value) << (position * width)
                    for position, value in enumerate(samples)
                )
                words.extend(
                    (bits >> (32 * index)) & 0xFFFFFFFF for index in range(width)
                )
            descriptors.extend([base, widths])
        payloads.append(np.asarray(words, dtype="<u4").tobytes())
        headers.append(np.asarray(descriptors, dtype="<u4").tobytes())
    binary = bytearray(
        struct.pack("<8s9I", b"QGIX\0\0\0\3", 2, 0, 16, 16, 2, 3, 128, 32, 1)
    )
    binary.extend(struct.pack("<2I", 1, 3))
    binary.extend(bytes.fromhex(metadata["source_identity_sha256"]))
    cursor = 8192
    chunks = []
    for payload, header in zip(payloads, headers, strict=True):
        binary.extend(
            struct.pack(
                "<7Q2I32s",
                cursor,
                len(payload),
                0,
                0,
                cursor + len(payload),
                len(header),
                len(payload),
                len(header) // 4,
                0,
                hashlib.sha256(payload).digest(),
            )
        )
        chunks.extend([payload, header])
        cursor += len(payload) + len(header)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    output = bytearray(8192)
    output[:24] = struct.pack(
        "<8s4I", b"QGPUH5\0\1", len(encoded), zlib.crc32(encoded), 4096, len(binary)
    )
    output[24 : 24 + len(encoded)] = encoded
    output[4096 : 4096 + len(binary)] = binary
    path.write_bytes(bytes(output) + b"".join(chunks))
    return expected.reshape(16, 16, 2, 3)


def test_selected_patterns_preserve_order_duplicates_and_bounded_staging(
    tmp_path, monkeypatch
):
    """Selecting across shard boundaries returns exact independent patterns."""
    pytest.importorskip("Metal")
    path = tmp_path / "two-shards.qh5"
    expected = _two_shard_fixture(path)
    source = mps.load_compact_v3_mps(path)
    positions = np.array([[15, 15], [0, 0], [8, 0], [7, 15], [15, 15], [2, 13]])
    allocate = mps._allocate_shared
    allocations = []

    def record_allocation(device, metal, count, label):
        allocations.append((label, count))
        return allocate(device, metal, count, label)

    monkeypatch.setattr(mps, "_allocate_shared", record_allocation)
    monkeypatch.setattr(mps, "_SELECTED_STAGING_BYTES", 48)
    try:
        selected = source.extract_diffractions(positions)
        np.testing.assert_array_equal(
            selected, expected[positions[:, 0], positions[:, 1]]
        )
        assert selected.dtype == np.dtype("uint32")
        assert allocations == [("selected patterns", 48), ("selected positions", 16)]
        selected[:] = 0
        np.testing.assert_array_equal(
            source.extract_diffraction(15, 15), expected[15, 15].reshape(-1)
        )
        allocations.clear()
        assert source.extract_diffractions([]).shape == (0, 2, 3)
        assert allocations == []
        for invalid in ([[16, 0]], [[-1, 0]], [[0, 16]]):
            with pytest.raises(IndexError):
                source.extract_diffractions(invalid)
        for invalid in ([0, 1], [[0.5, 1]], [[True, False]]):
            with pytest.raises(ValueError):
                source.extract_diffractions(invalid)
        assert allocations == []
    finally:
        source.release()


def test_detector_binary_values_and_empty_rebase_preserve_exact_state(tmp_path):
    """Invalid weighted masks do not change a prior exact detector product."""
    pytest.importorskip("Metal")
    path = tmp_path / "mask-state.qh5"
    values, _, _ = _fixture(path)
    source = mps.load_compact_v3_mps(path)
    full = np.ones((2, 3), dtype=np.uint8)
    expected = values.sum(axis=0, dtype=np.uint32)
    try:
        source.update_virtual_detector(full)
        for invalid in (256, -256, 1.5, np.nan):
            mask = full.astype(np.float64)
            mask[0, 0] = invalid
            with pytest.raises(ValueError, match="zero-or-one"):
                source.update_virtual_detector(mask)
            np.testing.assert_array_equal(source.virtual_detector_values(), expected)
        source.update_virtual_detector(np.zeros_like(full), force_rebase=True)
        np.testing.assert_array_equal(
            source.virtual_detector_values(), np.zeros(128, dtype=np.uint32)
        )
        source.update_virtual_detector(full)
        np.testing.assert_array_equal(source.virtual_detector_values(), expected)
    finally:
        source.release()
