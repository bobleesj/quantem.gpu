"""MAPED summaries keep exact integer totals before the final float32 rounding."""

import pytest
import torch

from quantem.gpu import io


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_encoded_summary_means_preserve_large_integer_totals(tmp_path):
    shape = (4, 5, 192, 192)
    counts_t = ((torch.arange(4 * 5 * 192 * 192).reshape(shape) * 73) % 5536 + 60000).to(torch.uint16)
    path = tmp_path / "high_counts_master.h5"
    io.save(path, counts_t, backend="cpu", verbose=False)
    with io.load(path, device="mps", verbose=False) as source:
        for method, axes in (("mean_dp_device", (0, 1)), ("detector_mean_device", (2, 3))):
            summary = getattr(source.data, method)()
            try:
                actual_t = summary.to_torch()
                assert actual_t.device.type == "mps"
                assert torch.equal(actual_t.cpu(), counts_t.double().mean(dim=axes).float())
            finally:
                summary.release()

