"""Direct Torch reads preserve encoded counts and independent output ownership."""

import os

import pytest
import torch

from quantem.gpu import io

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_MPS_PRECISION_TEST") != "1",
    reason="Set QUANTEM_MPS_PRECISION_TEST=1 on a physical MPS accelerator.",
)


@pytest.mark.parametrize("modulus", [256, 65536])
def test_encoded_regions_remain_exact_after_repeated_reads_and_close(tmp_path, modulus):
    """Regions preserve high counts, partial rows, late frames, and caller ownership."""
    shape = (33, 33, 17, 24)
    indices_t = torch.arange(33 * 33 * 17 * 24, device="mps").reshape(shape)
    expected_t = ((indices_t * 73 + 19) % modulus).to(torch.uint16)
    path = tmp_path / "counts_master.h5"
    io.save(path, expected_t, backend="mps", dtype="uint16", verbose=False, wait=True)
    with io.load(path, backend="mps", representation="encoded", dtype="native", verbose=False) as source:
        retained = []
        regions = [(0, 2, 0, 33), (7, 10, 0, 33), (30, 33, 28, 33), (32, 33, 32, 33)]
        for _ in range(3):
            for region in regions:
                row0, row1, column0, column1 = region
                values_t = source.read(scan_region=region)
                retained.append((values_t, expected_t[row0:row1, column0:column1]))
        assert source.data.is_released is False
    assert source.data.is_released is True
    for values_t, wanted_t in retained:
        assert values_t.device.type == "mps"
        assert values_t.dtype == wanted_t.dtype
        assert torch.equal(values_t.to(torch.int32), wanted_t.to(torch.int32))
