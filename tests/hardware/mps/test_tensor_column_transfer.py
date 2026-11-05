"""Exact detector-column transfers from Torch-owned Metal storage."""

import numpy as np
import pytest


@pytest.mark.parametrize("view", ["contiguous", "offset", "transposed"])
def test_tensor_columns_transfer_to_mlx_without_host_mapping(view):
    """A fit can consume Torch detector columns, including private storage."""
    torch = pytest.importorskip("torch")
    mx = pytest.importorskip("mlx.core")
    if not torch.backends.mps.is_available():
        pytest.skip("requires physical MPS")
    from quantem.gpu.direct_ptycho.mps.frames import staged_tensor

    values = np.arange(8192, dtype=np.float32).reshape(128, 64) / 16 - 100
    tensor = torch.as_tensor(values, device="mps")
    if view == "offset":
        tensor, values = tensor[1:], values[1:]
    elif view == "transposed":
        tensor, values = tensor.T, values.T
    actual = staged_tensor(mx, tensor, mx.float32)
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual), values)
