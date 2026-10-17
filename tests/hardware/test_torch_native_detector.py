"""Native Torch detector outputs preserve the host API's bounded reductions."""

import numpy as np
import pytest
import torch

from quantem.gpu import detector


@pytest.fixture(params=["cuda", "mps"])
def device(request):
    """Require the named accelerator rather than silently falling back to CPU."""
    name = request.param
    available = torch.cuda.is_available() if name == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{name} hardware is required")
    return torch.device(name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("selected", [0, 3, 35])
@pytest.mark.parametrize("tensor_mask", [False, True])
def test_native_detector_matches_host_without_copy(device, dtype, selected, tensor_mask, monkeypatch):
    """Sparse, dense and empty detectors stay native with unchanged values."""
    values = torch.arange(3 * 4 * 5 * 7, dtype=dtype, device=device).reshape(3, 4, 5, 7)
    if dtype == torch.float32:
        values = values / 17
    mask = np.zeros((5, 7), dtype=bool)
    mask.flat[:selected] = True
    session = detector.prepare(values)
    try:
        expected = session.masked_sum(mask)
        reference = values.cpu().numpy()[..., mask].sum(axis=-1, dtype=np.float64).astype(np.float32)
        np.testing.assert_allclose(expected, reference, rtol=2e-6, atol=0)
        supplied = torch.as_tensor(mask, device=device) if tensor_mask else mask

        def no_host_copy(*_args, **_kwargs):
            raise AssertionError("native detector output copied a tensor to CPU")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", no_host_copy)
            result = session.masked_sum(supplied, output="native")
        assert result.device == values.device
        assert result.dtype == torch.float32
        np.testing.assert_array_equal(result.cpu().numpy(), expected)
        out = torch.empty_like(result)
        assert session.masked_sum(supplied, output="native", out=out) is out
        torch.testing.assert_close(out, result, rtol=0, atol=0)
        saved = result.clone()
        session.masked_sum(~supplied, output="native")
        torch.testing.assert_close(result, saved, rtol=0, atol=0)
    finally:
        session.close()


@pytest.mark.parametrize("invalid", ["shape", "dtype", "strides", "host"])
def test_native_detector_rejects_invalid_output(device, invalid):
    """An incompatible reusable buffer fails with a corrective error."""
    session = detector.prepare(torch.ones((3, 4, 5, 7), device=device))
    buffers = {
        "shape": torch.empty((4, 3), device=device),
        "dtype": torch.empty((3, 4), device=device, dtype=torch.int32),
        "strides": torch.empty((4, 3), device=device).T,
        "host": torch.empty((3, 4)),
    }
    try:
        with pytest.raises(ValueError, match="contiguous float32 Torch tensor"):
            session.masked_sum(np.ones((5, 7), bool), output="native", out=buffers[invalid])
    finally:
        session.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.uint16])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("tensor_mask", [False, True])
def test_weighted_detector_matches_independent_reference(
    device, dtype, dense, tensor_mask, monkeypatch,
):
    """Fractional and negative custom weights survive integer/sparse branches."""
    host = np.arange(3 * 4 * 5 * 7, dtype=np.float32).reshape(3, 4, 5, 7)
    if dtype == torch.float32:
        host /= 17
    values = torch.as_tensor(host, device=device).to(dtype)
    mask = np.full((5, 7), 0.5 if dense else 0, dtype=np.float32)
    mask.flat[:3] = [0.25, 0.75, -0.5]
    reference = (host.astype(np.float64) * mask).sum(axis=(-2, -1)).astype(np.float32)
    supplied = torch.as_tensor(mask, device=device) if tensor_mask else mask
    session = detector.prepare(values)
    try:
        # Force more than one reduction chunk; the last chunk is partial.
        monkeypatch.setattr(session._backend, "_frames_per_chunk", lambda: 5)
        actual = session.masked_sum(supplied, output="native")
        np.testing.assert_allclose(actual.cpu().numpy(), reference, rtol=2e-6, atol=0)
        np.testing.assert_array_equal(session.masked_sum(mask), actual.cpu().numpy())
        out = torch.empty_like(actual)
        assert session.masked_sum(supplied, output="native", out=out) is out
        torch.testing.assert_close(out, actual, rtol=0, atol=0)
    finally:
        session.close()
