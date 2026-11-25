"""io.load(..., detector_transpose=True) on the physical backend named by QEM_TEST_BACKEND (cuda or mps).

The same swap on CUDA and Apple GPUs: the resident holds the transpose of every frame, bit for bit, for NumPy arrays,
ARINA masters (flagged pixels filled first) and .qem copies, and a float32 copy's stored checksum covers the values as
stored. References are NumPy transposes of these small fixtures, read back once for the comparison.
"""

import hashlib
import os

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import io

SCAN, DETECTOR = (23, 29), (12, 10)      # 667 positions: two ingestion blocks, the second starting mid scan row


@pytest.fixture
def backend():
    name = os.environ.get("QEM_TEST_BACKEND")
    if name not in ("cuda", "mps"):
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return name


def values_of(dtype, shape=(*SCAN, *DETECTOR)):
    """Every bit pattern of the dtype, with constant and zero streams."""
    dtype = np.dtype(dtype)
    unsigned = np.dtype(f"u{dtype.itemsize}")
    bits = np.random.default_rng(1125).integers(0, np.iinfo(unsigned).max, shape, dtype=unsigned, endpoint=True)
    bits[..., 0, :3] = 0
    bits[..., 1, :3] = 1
    return bits.view(dtype)


def stored(loaded) -> np.ndarray:
    """The resident's values as decoded on its GPU, read back once."""
    values = loaded.read(output="native")
    try:
        return values.get()
    finally:
        if hasattr(values, "release"):
            values.release()


def transposed(values) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(values, (0, 1, 3, 2)))


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "int16", "float32", "float64"])
def test_numpy_source(tmp_path, backend, dtype):
    values = values_of(dtype)
    np.save(tmp_path / "scan.npy", values)
    with io.load(tmp_path / "scan.npy", device=backend, detector_transpose=True, verbose=False) as loaded:
        assert loaded.shape == (*SCAN, DETECTOR[1], DETECTOR[0])
        assert loaded.metadata["detector_transpose"] is True
        assert stored(loaded).tobytes() == transposed(values).tobytes()


@pytest.mark.parametrize("swap", [False, True])
def test_float_checksum_covers_the_stored_values(tmp_path, backend, swap):
    values = values_of("float32", shape=(37, 29, *DETECTOR))            # 512 + 512 + 49 frames
    np.save(tmp_path / "scan.npy", values)
    expected = transposed(values) if swap else values
    with io.load(tmp_path / "scan.npy", device=backend, detector_transpose=swap, verbose=False) as loaded:
        assert loaded.data.header["logical_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()


FLAGGED = (3, 7)


def master(folder, dtype):
    counts = np.random.default_rng(7).poisson(4.0, size=(SCAN[0] * SCAN[1], 16, 12)).astype(dtype)
    mask = np.zeros((16, 12), np.uint32)
    counts[:, FLAGGED[0], FLAGGED[1]] = np.iinfo(dtype).max
    mask[FLAGGED] = 1
    with h5py.File(folder / "scan_data_000001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=counts, chunks=(1, 16, 12), **hdf5plugin.Bitshuffle(cname="lz4"))
    with h5py.File(folder / "scan_master.h5", "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data_000001.h5", "/entry/data/data")
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = np.uint32(len(counts))
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask
        handle.attrs["detector_sampling_mrad"] = np.array([0.5, 0.25])
    return folder / "scan_master.h5"


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_arina_master(tmp_path, backend, dtype):
    path = master(tmp_path, dtype)
    options = dict(device=backend, scan_shape=SCAN, hot_pixel_correction="median", verbose=False)
    with io.load(path, **options) as plain, io.load(path, detector_transpose=True, **options) as swapped:
        assert swapped.shape == (*SCAN, 12, 16)
        assert stored(swapped).tobytes() == transposed(stored(plain)).tobytes()
        assert swapped.metadata["detector_sampling_mrad"] == [0.25, 0.5]
        assert swapped.metadata["hot_pixel_correction"]["coordinates_row_col"] == [[FLAGGED[1], FLAGGED[0]]]


@pytest.mark.parametrize("dtype", ["uint16", "int16", "float32"])
def test_qem_copy_reopened_transposed_and_saved(tmp_path, backend, dtype):
    values = values_of(dtype)
    np.save(tmp_path / "scan.npy", values)
    with io.load(tmp_path / "scan.npy", device=backend, verbose=False) as loaded:
        io.save(tmp_path / "plain.qem", loaded, verbose=False)
    with io.load(tmp_path / "plain.qem", device=backend, detector_transpose=True, verbose=False) as swapped:
        assert stored(swapped).tobytes() == transposed(values).tobytes()
        io.save(tmp_path / "swapped.qem", swapped, verbose=False)
    with io.load(tmp_path / "swapped.qem", device=backend, verbose=False) as reopened:
        assert reopened.metadata["detector_transpose"] is True
        assert stored(reopened).tobytes() == transposed(values).tobytes()
    with pytest.raises(ValueError, match="already swapped"):
        io.load(tmp_path / "swapped.qem", device=backend, detector_transpose=True, verbose=False)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32", "float64"])
@pytest.mark.parametrize("frame", [(33, 70), (70, 33), (65, 31), (3, 40)])
def test_device_transpose_across_partial_tiles(backend, dtype, frame):
    """32 x 32 tiles: frames that are not square and not a multiple of 32, several tiles each way."""
    from quantem.gpu.io.arrays import transposed_frames

    values = values_of(dtype, shape=(1, 3, *frame)).reshape(3, *frame)
    if backend == "mps":
        from quantem.gpu.resident.mps import precision as metal_precision

        raw = metal_precision.upload(values)
    else:
        import cupy as cp

        raw = cp.asarray(values)
    result = transposed_frames(raw)
    try:
        assert result.get().tobytes() == np.ascontiguousarray(values.transpose(0, 2, 1)).tobytes()
    finally:
        for array in (raw, result):
            if hasattr(array, "release"):
                array.release()
