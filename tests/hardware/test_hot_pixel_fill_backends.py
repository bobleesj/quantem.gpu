"""The Metal and CUDA flagged-pixel fills store the CPU reference's counts bit for bit.

Masks cover an isolated pixel (8 neighbors), edges and corners (odd and even
counts, so .5 medians round half to even) and a 3 x 3 cluster whose centre is
filled from its 5 x 5 ring.
"""

import os

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.resident.hot_pixels import fill_flagged, median_plan


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return backend


def _mask(shape):
    mask = np.zeros(shape, np.uint32)
    mask[0, 0] = mask[0, 5] = mask[0, 6] = mask[shape[0] - 1, 9] = 1
    mask[12, 20] = 1
    mask[20:23, 30:33] = 2
    return mask


def _write_master(folder, counts, mask):
    master = folder / "fill_master.h5"
    data = folder / "fill_data_000001.h5"
    with h5py.File(data, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts, chunks=(1, *counts.shape[1:]),
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(str(data), "/entry/data/data")
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = np.uint32(len(counts))
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask
    return master


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_gpu_corrector_matches_the_cpu_reference(dtype):
    backend = _backend()
    shape = (40, 48)
    mask = _mask(shape)
    # small counts so even neighbor sets often share a .5 median
    frames = np.random.default_rng(int(np.dtype(dtype).itemsize)).integers(0, 6, (64, *shape)).astype(dtype)
    frames[:, mask != 0] = np.iinfo(dtype).max
    expected = fill_flagged(frames.copy(), median_plan(mask), "median")
    if backend == "mps":
        from quantem.gpu.device.metal_runtime import buffer_view
        from quantem.gpu.resident.mps.hot_pixels import MPSHotPixelCorrector
        from quantem.gpu.resident.mps.precision import upload

        corrector = MPSHotPixelCorrector(mask, "median")
        values = upload(frames)
        corrector.apply(values)
        stored = np.frombuffer(buffer_view(values._mtl, frames.nbytes), frames.dtype).reshape(frames.shape).copy()
        corrector.close()
    else:
        import cupy as cp

        from quantem.gpu.resident.cuda.hot_pixels import CUDAHotPixelCorrector

        values = cp.asarray(frames)
        CUDAHotPixelCorrector(mask, "median").apply(values)
        stored = values.get()
    np.testing.assert_array_equal(stored, expected)
    assert (stored[:, 21, 31] == expected[:, 21, 31]).all()


def test_encoded_load_matches_the_cpu_reference_load(tmp_path):
    backend = _backend()
    shape = (40, 48)
    mask = _mask(shape)
    counts = np.random.default_rng(9).integers(0, 8, (8 * 9, *shape), dtype=np.uint16)
    counts[:, mask != 0] = 65535
    master = _write_master(tmp_path, counts, mask)
    reference = io.load(master, backend="cpu", representation="dense", scan_shape=(8, 9), verbose=False)
    with io.load(master, backend=backend, scan_shape=(8, 9), verbose=False) as loaded:
        record = loaded.metadata["hot_pixel_correction"]
        assert (record["method"], record["applied"], record["largest_window_px"]) == ("median", True, 5)
        np.testing.assert_array_equal(loaded.read().cpu().numpy(), np.asarray(reference.data))
    np.testing.assert_array_equal(np.asarray(reference.data).reshape(counts.shape),
                                  fill_flagged(counts.copy(), median_plan(mask), "median"))
