"""A scaled-precision result frees the scan rows a reader has passed."""

import numpy as np
import pytest
import torch

cp = pytest.importorskip("cupy")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from quantem.gpu.io.dataset import Dataset4dstem

SHAPE = (16, 8, 12, 12)


class Rows:
    """Float32 patterns handed over two scan rows at a time, as a merge hands its regions."""

    dtype = np.dtype("float32")
    shape = SHAPE
    device = torch.device("cuda:0")
    save_metadata: dict = {}

    def __init__(self):
        generator = torch.Generator().manual_seed(3)
        self.values = (torch.rand(SHAPE, generator=generator) * 40).to(self.device)

    def blocks(self):
        for row in range(0, SHAPE[0], 2):
            yield self.values[row:row + 2].reshape(-1, *SHAPE[2:])


def test_rows_already_read_are_freed_and_later_rows_still_read():
    source = Rows()
    stored = Dataset4dstem.from_blocks(source, dtype="scaled_uint16", device="cuda:0")
    try:
        held = stored.data.nbytes
        first = stored.read(scan_region=(0, 4, 0, SHAPE[1]))
        torch.testing.assert_close(first, source.values[:4], atol=0.01, rtol=0)
        stored.release_rows_before(8)
        assert 0 < stored.data.nbytes <= held // 2 + 1024            # the parts of rows 0 to 7 are gone
        later = stored.read(scan_region=(8, 16, 0, SHAPE[1]))
        torch.testing.assert_close(later, source.values[8:], atol=0.01, rtol=0)
        with pytest.raises(RuntimeError, match="released"):
            stored.read(scan_region=(2, 4, 0, SHAPE[1]))
        stored.release_rows_before(8)                                # again: nothing more to free
    finally:
        stored.close()
