"""Bounded exact source-table admission, independent of accelerator availability."""

import numpy as np
import pytest

from quantem.gpu.io.backends.cuda._ans import _flat_scan_indices, _validate_arrays
from tests.parity.ans_counts_fixture import decode_reference, make_fixture


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_known_counts_keep_blocks_tail_and_native_literals(dtype):
    counts, encoded = make_fixture(dtype=dtype)
    shape, arrays = _validate_arrays(**encoded)
    assert shape == counts.shape
    np.testing.assert_array_equal(decode_reference(encoded), counts)
    assert arrays[0].nbytes == len(encoded["payload"])


def test_incomplete_probability_tables_are_not_admitted():
    _, encoded = make_fixture()
    encoded["frequencies"][0] -= 1
    with pytest.raises(ValueError, match="summing to"):
        _validate_arrays(**encoded)


def test_a_truncated_literal_tail_is_not_admitted():
    _, encoded = make_fixture()
    # Stream 21 is the literal column of the three-position final block.
    encoded["offsets"][22] -= 1
    with pytest.raises(ValueError, match="byte length"):
        _validate_arrays(**encoded)


def test_large_scan_indices_keep_exact_integer_order_without_gpu_allocation():
    positions = np.asarray([[2**53 + 1, 1], [0, 0], [2**53 + 1, 1]], np.int64)
    flat = _flat_scan_indices(positions, (2**53 + 2, 2))
    assert flat.dtype == np.uint64
    np.testing.assert_array_equal(
        flat, np.asarray([2**54 + 3, 0, 2**54 + 3], np.uint64)
    )
