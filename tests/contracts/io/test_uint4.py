from __future__ import annotations

import numpy as np
import pytest


def test_pack_uint4_numpy_round_trips_even_and_odd_shapes() -> None:
    from quantem.gpu.io.uint4 import pack_uint4_numpy, unpack_uint4_numpy

    even = np.arange(24, dtype=np.uint8).reshape(2, 3, 4) % 16
    odd = np.arange(15, dtype=np.uint8).reshape(3, 5) % 16

    packed_even = pack_uint4_numpy(even)
    packed_odd = pack_uint4_numpy(odd)

    assert packed_even.dtype == "uint4"
    assert packed_even.nbytes == even.size // 2
    assert packed_odd.nbytes == (odd.size + 1) // 2
    np.testing.assert_array_equal(unpack_uint4_numpy(packed_even), even)
    np.testing.assert_array_equal(unpack_uint4_numpy(packed_odd), odd)


def test_pack_uint4_numpy_rejects_values_above_15() -> None:
    from quantem.gpu.io.uint4 import pack_uint4_numpy

    data = np.asarray([0, 15, 16], dtype=np.uint8)

    with pytest.raises(ValueError, match="fit in 0..15"):
        pack_uint4_numpy(data)


def test_packed_uint4_reshape_keeps_buffer_and_checks_size() -> None:
    from quantem.gpu.io.uint4 import pack_uint4_numpy

    packed = pack_uint4_numpy(np.arange(16, dtype=np.uint8))
    reshaped = packed.reshape(2, 2, 4)
    inferred = packed.reshape(-1, 4)

    assert reshaped.buffer is packed.buffer
    assert reshaped.shape == (2, 2, 4)
    assert inferred.shape == (4, 4)
    with pytest.raises(ValueError, match="cannot reshape"):
        packed.reshape(3, 6)
