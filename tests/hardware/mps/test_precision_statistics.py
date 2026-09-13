"""GPU error reports retain small contributions and large pixel counts."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_MPS_PRECISION_TEST") != "1",
    reason="Set QUANTEM_MPS_PRECISION_TEST=1 on an Apple GPU host.",
)


def test_precision_statistics_keep_low_bits_and_wide_counts():
    """A precision report preserves small errors beside large partial sums."""
    import torch
    from quantem.gpu.io.backends.mps.precision import _accumulate_measurement

    errors = torch.zeros((8192, 4), device="mps", dtype=torch.float32)
    errors[::2, 0] = 1.0
    errors[1::2, 0] = 2**-25
    errors[:, 1] = 0.25
    counts = torch.zeros((8192, 4), device="mps", dtype=torch.int64)
    counts[:, 0] = 2**20
    counts[:, 1] = 1
    counts = counts.to(torch.uint32)
    report = dict(squared_error=0.0, max_abs_error=0.0, changed=0,
                  positive_to_zero=0, overflow=0, values=0)
    _accumulate_measurement(errors, counts, 0, report, 2**34)
    assert report["squared_error"] == 4096 + 4096 * 2**-25
    assert report["max_abs_error"] == 0.25
    assert report["changed"] == 2**33
    assert report["positive_to_zero"] == 8192
    assert report["overflow"] == 0
    assert report["values"] == 2**34

    errors[0, 0] = float("inf")
    _accumulate_measurement(errors, counts, 0, report, 2**34)
    assert report["squared_error"] == float("inf")
