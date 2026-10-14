"""``io.load.bin`` sums detector (or scan) blocks of CuPy arrays and Torch tensors on their GPU, exactly for integer counts.

A reconstruction reads a binned crop of the counts, ``bin(dataset.read(scan_region=...), factor=4)``: every electron is
kept (the block sum), integers stay integers in a wide accumulator, and the library and device of the input are kept.
Each case is checked against the same block sum written out with NumPy in int64.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")


def _require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")


def _block_sum(values: np.ndarray, factor: int, axes: tuple[int, ...]) -> np.ndarray:
    """out[..., i, ...] = sum of values[..., f i + a, ...] over a < f on each binned axis, trailing pixels dropped."""
    for axis in axes:
        kept = values.shape[axis] // factor * factor
        values = np.take(values, np.arange(kept), axis=axis)
        values = values.reshape(*values.shape[:axis], kept // factor, factor, *values.shape[axis + 1:]).sum(axis=axis + 1)
    return values


@pytest.mark.parametrize("library", ["torch", "cupy"])
@pytest.mark.parametrize("axes,binned", [("detector", (2, 3)), ("scan", (0, 1)), ("all", (0, 1, 2, 3))])
def test_bin_is_the_exact_block_sum_of_integer_counts(library, axes, binned):
    _require_cuda()
    from quantem.gpu.io.load import bin

    counts = np.random.default_rng(0).integers(0, 65536, size=(6, 7, 13, 14), dtype=np.uint16)
    data = torch.as_tensor(counts, device="cuda") if library == "torch" else cp.asarray(counts)
    out = bin(data, factor=4, axes=axes)
    assert type(out) is type(data)
    assert out.dtype == (torch.int64 if library == "torch" else cp.uint64)
    host = out.cpu().numpy() if library == "torch" else cp.asnumpy(out)
    np.testing.assert_array_equal(host, _block_sum(counts.astype(np.int64), 4, binned))


def test_partial_edges_keep_the_trailing_pixels_and_mean_divides_by_the_block():
    _require_cuda()
    from quantem.gpu.io.load import bin

    counts = np.arange(5 * 7, dtype=np.uint16).reshape(1, 5, 7)
    data = cp.asarray(counts)
    padded = np.zeros((1, 6, 8), dtype=np.int64)
    padded[:, :5, :7] = counts
    np.testing.assert_array_equal(cp.asnumpy(bin(data, factor=2, edge="partial")), _block_sum(padded, 2, (1, 2)))
    np.testing.assert_array_equal(cp.asnumpy(bin(data, factor=2, reduction="mean")),
                                  (_block_sum(counts.astype(np.float64), 2, (1, 2)) / 4).astype(np.float32))
    assert bin(data, factor=1) is data
    with pytest.raises(ValueError, match="partial"):
        bin(data, factor=2, edge="partial", reduction="mean")
    with pytest.raises(ValueError, match="4D"):
        bin(data, factor=2, axes="scan")


def test_signed_and_wide_counts_sum_without_wrapping():
    """Negative values stay negative and uint32 counts past 2^32 per block stay whole."""
    _require_cuda()
    from quantem.gpu.io.load import bin

    signed = cp.asarray(np.array([[-1, -1], [0, 0]], dtype=np.int16))
    assert int(bin(signed, factor=2)[0, 0]) == -2
    wide = cp.asarray(np.array([[4_000_000_000, 4_000_000_000], [1, 1]], dtype=np.uint32))
    assert int(bin(wide, factor=2)[0, 0]) == 8_000_000_002
    half = torch.full((2, 2), 30000, dtype=torch.float16, device="cuda")
    assert float(bin(half, factor=2, dtype=torch.float32)[0, 0]) == 120000.0
