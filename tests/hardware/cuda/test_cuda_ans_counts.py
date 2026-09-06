"""Bounded exact ANS-count workflows on a physically admitted CUDA device."""

import os

import numpy as np
import pytest

from tests.parity.ans_counts_fixture import make_fixture


@pytest.fixture
def counts_source():
    if os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1":
        pytest.skip("Set QUANTEM_CUDA_ANS_TEST=1 only in an owned CUDA test window.")
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("No physical CUDA device is available.")
    from quantem.gpu.io.backends.cuda._ans import CudaANSResidentCounts

    counts, encoded = make_fixture()
    source = CudaANSResidentCounts(**encoded)
    yield counts, encoded, source
    source.release()


def test_exact_block_decode_preserves_tail_and_rare_counts(counts_source):
    counts, _, source = counts_source
    blocks = [source.decode_block_device(block).get() for block in range(4)]
    np.testing.assert_array_equal(np.concatenate(blocks).reshape(counts.shape), counts)
    assert blocks[-1].shape == (3, 2, 3)


def test_random_patterns_keep_order_duplicates_and_output_ownership(counts_source):
    counts, _, source = counts_source
    positions = np.asarray([[2, 4], [0, 0], [1, 4], [2, 4]], np.int64)
    patterns = source.gather_diffraction_device(positions)
    source.extract_diffraction_device(0, 1)
    np.testing.assert_array_equal(
        patterns.get(), counts[positions[:, 0], positions[:, 1]]
    )
    assert patterns.dtype == np.uint16


def test_moving_binary_masks_keep_exact_counts_without_dense_source(counts_source):
    counts, _, source = counts_source
    for mask in [
        np.ones((2, 3), np.uint8),
        np.asarray([[0, 1, 0], [1, 0, 1]], np.uint8),
    ]:
        expected = (counts.astype(np.uint64) * mask).sum(axis=(2, 3), dtype=np.uint64)
        np.testing.assert_array_equal(source.detector_sum_device(mask).get(), expected)
    assert source.nbytes == source.resident_bytes


def test_native_uint8_requires_exact_range_proof(counts_source):
    _, encoded, _ = counts_source
    from quantem.gpu.io.backends.cuda._ans import CudaANSResidentCounts

    with pytest.raises(ValueError, match="exceed"):
        CudaANSResidentCounts(**encoded, dtype="uint8")
    counts, small = make_fixture(dtype="uint8")
    source = CudaANSResidentCounts(**small, dtype="uint8")
    try:
        actual = source.extract_diffraction_device(2, 4)
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual.get(), counts[2, 4])
    finally:
        source.release()


def test_malformed_stream_never_becomes_resident(counts_source):
    _, encoded, _ = counts_source
    from quantem.gpu.io.backends.cuda._ans import CudaANSResidentCounts

    encoded["payload"][:4] = 0
    with pytest.raises(ValueError, match="Malformed"):
        CudaANSResidentCounts(**encoded)


def test_direct_packed_conversion_keeps_exact_products_after_original_release(
    counts_source,
):
    counts, _, original = counts_source
    original_bytes = original.resident_bytes
    packed = original.to_packed()
    try:
        assert (
            packed.conversion_owned_buffer_peak_bytes
            >= original_bytes + packed.resident_bytes
        )
        original.release()
        positions = np.asarray([[2, 4], [0, 0], [2, 4]], np.int64)
        np.testing.assert_array_equal(
            packed.gather_diffraction_device(positions).get(),
            counts[positions[:, 0], positions[:, 1]],
        )
        np.testing.assert_array_equal(
            packed.decode_block_device(3).get(), counts.reshape(15, 2, 3)[12:]
        )
        mask = np.ones((2, 3), np.uint8)
        np.testing.assert_array_equal(
            packed.detector_sum_device(mask).get(),
            counts.sum(axis=(2, 3), dtype=np.uint64),
        )
    finally:
        packed.release()


@pytest.mark.parametrize("width", [0, 1, 7, 16])
def test_direct_packed_conversion_keeps_cross_word_values(counts_source, width):
    from quantem.gpu.io.backends.cuda._ans import CudaANSResidentCounts

    values = np.full((1, 15, 1, 1), (1 << width) - 1, dtype=np.uint16)
    values[0, 0] = 0
    source = CudaANSResidentCounts(
        shape=values.shape,
        block_frames=15,
        scale=4,
        payload=np.frombuffer(values.tobytes(), np.uint8),
        offsets=np.asarray([0, values.nbytes], np.uint64),
        model_ids=np.asarray([0], np.uint32),
        context_offsets=np.asarray([0, 0], np.uint32),
        symbols=np.asarray([], np.uint16),
        cumulative=np.asarray([], np.uint16),
        frequencies=np.asarray([], np.uint16),
        literal=np.asarray([1], np.uint8),
    )
    packed = None
    try:
        packed = source.to_packed()
        np.testing.assert_array_equal(
            packed.decode_block_device(0).get(), values.reshape(15, 1, 1)
        )
        assert packed._arrays[2].get()[0] == width
        assert packed._arrays[0].size == (15 * width + 31) // 32
    finally:
        source.release()
        if packed is not None:
            packed.release()
