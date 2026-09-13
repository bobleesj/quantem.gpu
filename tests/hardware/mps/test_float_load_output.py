"""Explicit Torch output preserves the loader's MPS backend selection."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_MPS_PRECISION_TEST") != "1",
    reason="Set QUANTEM_MPS_PRECISION_TEST=1 on an Apple GPU host.",
)


@pytest.mark.parametrize("host_tensor", [False, True])
def test_float_load_output_returns_mps_tensor(host_tensor):
    """Host-staged float IO payloads return to the requested GPU without loss."""
    import importlib
    import torch

    loader = importlib.import_module("quantem.gpu.io.load")
    reference = torch.arange(8192, device="mps", dtype=torch.float32) / 8
    # Simulate host IO staging; all value construction and comparison is on MPS.
    staged = reference.cpu()
    payload = staged if host_tensor else staged.numpy()
    result = loader._convert_load_output(
        loader.LoadResult(payload, {"backend": "mps"}), "torch"
    )
    assert result.data.device.type == "mps"
    assert result.data.dtype == torch.float32
    assert bool(torch.all(result.data == reference))
