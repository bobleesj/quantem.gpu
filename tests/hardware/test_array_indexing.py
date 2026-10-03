"""Scientist-facing basic indexing of encoded acquisitions."""
import os

import numpy as np
import pytest
import torch

from quantem.gpu import io


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_encoded_indexing(tmp_path, dtype):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    values = (np.arange(4 * 5 * 8 * 10).reshape(4, 5, 8, 10) % 101).astype(dtype)
    if dtype == np.float32:
        values = values / 8 - 3
    path = tmp_path / "input.npy"
    np.save(path, values)
    selections = [
        (1, 2), -1, (1, 2, 3, 4), (), Ellipsis,
        (Ellipsis, -1), (slice(1, 3), slice(2, 5)),
        (1, 2, slice(2, 7), slice(3, 9)),
        (slice(None, None, 2), slice(None), slice(None, None, -2), -1),
        (slice(None, None, -1), slice(4, 0, -2), slice(None), slice(None)),
        (slice(3, 1), slice(None)), (slice(20, 30), 1),
        (np.int64(-1), np.int32(2)),
    ]
    with io.load(path, backend=backend, verbose=False) as data:
        assert isinstance(data, io.Dataset4dstemGPU)
        pattern = data[1, 2]
        assert type(pattern) is torch.Tensor
        assert pattern.float().device.type == backend
        assert torch.mean(pattern.float()).device.type == backend
        assert torch.as_tensor(pattern).data_ptr() == pattern.data_ptr()
        assert data.shape == values.shape
        assert data.ndim == values.ndim
        assert data.size == values.size
        assert len(data) == len(values)
        assert data.dtype == values.dtype
        for row, actual in enumerate(data):
            assert str(actual.device).split(":")[0] == backend
            np.testing.assert_array_equal(actual.cpu().numpy(), values[row])
        for key in selections:
            actual = data[key]
            assert str(actual.device).split(":")[0] == backend
            np.testing.assert_array_equal(actual.cpu().numpy(), values[key])
        for key in [4, -5, (0, 0, 0, 0, 0), (Ellipsis, Ellipsis)]:
            with pytest.raises(IndexError):
                data[key]
        for key in [True, [0, 1], None, 1.5]:
            with pytest.raises(TypeError):
                data[key]
        with pytest.raises(ValueError):
            data[::0]
        saved = tmp_path / "saved.qem"
        io.save(saved, data)
    series = io.load([saved, saved], backend=backend, stack=False, verbose=False)
    try:
        np.testing.assert_array_equal(series[1][1, 2].cpu().numpy(), values[1, 2])
        assert isinstance(series[1].metadata, dict)
    finally:
        for data in series:
            data.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_detector_pixels_across_ans_intervals(tmp_path, dtype):
    """Detector crops preserve scans across 512-frame streams and a short tail."""
    if os.environ.get("QEM_TEST_BACKEND") != "cuda":
        pytest.skip("Selective ANS stream decoder is CUDA-specific.")
    values = np.random.default_rng(5).integers(
        0, 200, size=(35, 33, 8, 10), dtype=dtype
    )
    path = tmp_path / "intervals.npy"
    np.save(path, values)
    with io.load(path, backend="cuda", verbose=False) as data:
        for key in [
            (slice(None), slice(None), 3, 6),
            (slice(None), slice(None), slice(2, 7), slice(4, 9)),
            (slice(14, 35), slice(5, 30), slice(2, 7), slice(4, 9)),
            (slice(None, None, -2), slice(None), -1, 0),
        ]:
            np.testing.assert_array_equal(data[key].cpu().numpy(), values[key])


@pytest.mark.parametrize("apply_mask", [None, False])
def test_indexing_preserves_uncorrected_flagged_counts(tmp_path, apply_mask):
    """Raw reads preserve flagged values before and after an exact QEM save."""
    import h5py
    import hdf5plugin

    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    values = np.arange(4 * 8 * 10, dtype=np.uint16).reshape(2, 2, 8, 10)
    values[:, :, 3, 4] = 50000
    mask = np.zeros((8, 10), np.uint8)
    mask[3, 4] = 1
    original = tmp_path / "masked.h5"
    saved = tmp_path / "masked.qem"
    with h5py.File(original, "w") as handle:
        handle.create_dataset(
            "entry/data/data", data=values.reshape(4, 8, 10),
            chunks=(1, 8, 10), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = 4
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask

    with io.load(
        original, backend=backend, apply_mask=apply_mask,
        hot_pixel_correction="none", verbose=False,
    ) as loaded:
        assert loaded.lossless
        np.testing.assert_array_equal(loaded[0, 0].cpu().numpy(), values[0, 0])
        np.testing.assert_array_equal(
            loaded[:, :, 3, 4].cpu().numpy(), values[:, :, 3, 4]
        )
        io.save(saved, loaded)
    with io.load(saved, backend=backend, verbose=False) as reopened:
        np.testing.assert_array_equal(
            reopened[:, :, 2:5, 3:6].cpu().numpy(), values[:, :, 2:5, 3:6]
        )


@pytest.mark.parametrize("scan_shape", [(9, 64), (1, 576)])
def test_float_reads_split_at_decoder_byte_limit(tmp_path, scan_shape):
    """A float selection can exceed one decode window, including a wide row."""
    from quantem.gpu.io._float_ans import MAX_DECODE_BYTES

    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    shape = (*scan_shape, 128, 128)
    values = (np.arange(np.prod(shape), dtype=np.float32) % 251).reshape(shape) / 8
    path = tmp_path / "float.npy"
    np.save(path, values)
    assert values.nbytes > MAX_DECODE_BYTES
    with io.load(path, backend=backend, verbose=False) as loaded:
        for key in (
            (slice(None), slice(None), 5, 7),
            (Ellipsis, slice(32, 40), slice(45, 52)),
            (slice(None, None, -1), slice(None, None, -3), 0, 0),
            (slice(0, 0), slice(None), 5, 7),
            Ellipsis,
        ):
            np.testing.assert_array_equal(loaded[key].cpu().numpy(), values[key])
        assert loaded.data.peak_decode_bytes <= MAX_DECODE_BYTES
