"""Physical exact count-rANS conformance, not a full-dataset timing claim."""

import numpy as np
import pytest

from quantem.gpu.io.backends.mps import _ans as mps
from tests.parity.ans_counts_fixture import make_fixture


def _read(output):
    try:
        return output.to_numpy()
    finally:
        output.release()


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ans_native_counts_selected_patterns_and_detector_masks(dtype, monkeypatch):
    """Rare counts, ordered repeated DPs and mask sums equal the independent source."""
    pytest.importorskip("Metal")
    counts, parameters = make_fixture(dtype=dtype)
    source = mps.MPSANSResidentCounts(**parameters, dtype=dtype)
    positions = np.asarray([[2, 4], [0, 1], [1, 3], [0, 1], [0, 0]])
    monkeypatch.setattr(
        mps, "_REDUCTION_STAGING_BYTES", 2 * 6 * np.dtype(dtype).itemsize
    )
    try:
        blocks = [_read(source.decode_block_device(block)) for block in range(4)]
        decoded = np.concatenate(blocks).reshape(counts.shape)
        np.testing.assert_array_equal(decoded, counts)
        assert decoded.dtype == counts.dtype
        selected = _read(source.gather_diffraction_device(positions))
        np.testing.assert_array_equal(
            selected, counts[positions[:, 0], positions[:, 1]]
        )
        assert selected.dtype == counts.dtype
        np.testing.assert_array_equal(
            _read(source.extract_diffraction_device(0, 1)), counts[0, 1]
        )
        for mask in (
            np.ones((2, 3), bool),
            np.eye(2, 3, dtype=np.uint8),
            np.zeros((2, 3), bool),
        ):
            actual = _read(source.detector_sum_device(mask))
            expected = (counts * mask).sum(axis=(2, 3), dtype=np.uint64)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.dtype("uint64")
        assert source.logical_nbytes == counts.nbytes
        assert (
            source.resident_bytes
            == sum(
                max(1, value.nbytes)
                for value in parameters.values()
                if isinstance(value, np.ndarray)
            )
            + 4
        )
        for invalid in (np.nan, 0.5, 256):
            mask = np.ones((2, 3), dtype=np.float64)
            mask[0, 0] = invalid
            with pytest.raises(ValueError, match="zero or one"):
                source.detector_sum_device(mask)
        independent = source.extract_diffraction_device(0, 1)
        source.release()
        np.testing.assert_array_equal(_read(independent), counts[0, 1])
        assert source.resident_bytes == 0
        with pytest.raises(RuntimeError, match="released"):
            source.extract_diffraction_device(0, 0)
    finally:
        source.release()


def test_ans_invalid_stream_and_native_dtype_do_not_publish_counts():
    """Malformed states and values beyond uint8 fail before resident publication."""
    pytest.importorskip("Metal")
    _, parameters = make_fixture()
    corrupted = dict(parameters, payload=parameters["payload"].copy())
    corrupted["payload"][:4] = 0
    with pytest.raises(ValueError, match="Malformed rANS"):
        mps.MPSANSResidentCounts(**corrupted)
    with pytest.raises(ValueError, match="exceed the declared native dtype"):
        mps.MPSANSResidentCounts(**parameters, dtype="uint8")


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_ans_to_packed_preserves_values_without_source_lifetime(dtype, monkeypatch):
    """Conversion keeps native counts and exact reductions after closing ANS."""
    pytest.importorskip("Metal")
    counts, parameters = make_fixture(dtype=dtype)
    source = mps.MPSANSResidentCounts(**parameters, dtype=dtype)
    packed = None
    monkeypatch.setattr(
        mps, "_REDUCTION_STAGING_BYTES", 2 * 6 * np.dtype(dtype).itemsize
    )
    try:
        packed = source.to_packed()
        assert packed.dtype == counts.dtype
        assert packed.logical_nbytes == counts.nbytes
        assert packed.conversion_owned_buffer_peak_bytes == (
            source.resident_bytes + packed.resident_bytes + 24 * 8
        )
        source.release()
        decoded = np.concatenate(
            [_read(packed.decode_block_device(block)) for block in range(4)]
        )
        np.testing.assert_array_equal(decoded.reshape(counts.shape), counts)
        positions = np.asarray([[2, 4], [0, 1], [2, 4], [1, 3]])
        np.testing.assert_array_equal(
            _read(packed.gather_diffraction_device(positions)),
            counts[positions[:, 0], positions[:, 1]],
        )
        for mask in (
            np.ones((2, 3), bool),
            np.eye(2, 3, dtype=np.uint8),
            np.zeros((2, 3), bool),
        ):
            np.testing.assert_array_equal(
                _read(packed.detector_sum_device(mask)),
                (counts * mask).sum(axis=(2, 3), dtype=np.uint64),
            )
    finally:
        source.release()
        if packed is not None:
            packed.release()


def test_ans_to_packed_exact_cross_word_layout():
    """Eleven-bit counts cross word boundaries without changing uint16 values."""
    pytest.importorskip("Metal")
    counts = np.asarray(
        [
            [0, 65535],
            [1, 0],
            [2047, 256],
            [13, 32768],
            [17, 1],
            [1023, 0],
            [999, 65535],
            [55, 1],
            [129, 42],
        ],
        dtype=np.uint16,
    )
    payload = counts.T.copy().reshape(-1).view(np.uint8)
    source = mps.MPSANSResidentCounts(
        shape=(1, 9, 1, 2),
        block_frames=9,
        scale=1,
        payload=payload,
        offsets=np.asarray([0, 18, 36], np.uint64),
        model_ids=np.zeros(2, np.uint32),
        context_offsets=np.zeros(2, np.uint32),
        symbols=np.empty(0, np.uint16),
        cumulative=np.empty(0, np.uint16),
        frequencies=np.empty(0, np.uint16),
        literal=np.ones(1, np.uint8),
    )
    packed = None
    try:
        packed = source.to_packed()
        np.testing.assert_array_equal(
            _read(packed.decode_block_device(0)).reshape(9, 2), counts
        )
        expected_words = []
        for column, width in zip(counts.T, (11, 16), strict=True):
            bits = sum(
                int(value) << (index * width) for index, value in enumerate(column)
            )
            expected_words.extend(
                (bits >> (32 * word)) & 0xFFFFFFFF
                for word in range((len(column) * width + 31) // 32)
            )
        np.testing.assert_array_equal(
            np.frombuffer(mps._buffer_view(packed._buffers[0]), np.uint32),
            np.asarray(expected_words, np.uint32),
        )
    finally:
        source.release()
        if packed is not None:
            packed.release()
