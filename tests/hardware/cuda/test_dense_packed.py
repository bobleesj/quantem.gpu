"""Exact native HDF5 tilts remain independently available in packed storage."""

from quantem.gpu.io.models import Dataset4dstemGPU
import os

import h5py
import numpy as np
import pytest

from quantem.gpu import io

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_native_dense_conversion_preserves_every_count(dtype):
    cp = pytest.importorskip("cupy")
    counts = cp.random.RandomState(4).randint(
        0, np.iinfo(dtype).max + 1, (3, 91, 4, 7), dtype=dtype
    )
    counts[:, :, 0, 0] = 0
    source = Dataset4dstemGPU(cp.asarray(counts), {"working_shape": counts.shape})
    with source.to_representation("packed") as packed:
        blocks = [packed.data.decode_block_device(i) for i in range(3)]
        assert bool(cp.all(cp.concatenate(blocks).reshape(counts.shape) == counts))
        positions = np.asarray([[2, 90], [0, 0], [2, 90]])
        assert bool(cp.all(
            packed.data.gather_diffraction_device(positions)
            == counts[cp.asarray(positions[:, 0]), cp.asarray(positions[:, 1])]
        ))
        assert bool(cp.all(source.data == counts))


def test_load_all_tilts_then_remove_files(tmp_path):
    cp = pytest.importorskip("cupy")
    counts = cp.arange(3 * 91 * 4 * 7, dtype="uint16").reshape(3, 91, 4, 7)
    paths = []
    for index in range(3):
        path = tmp_path / f"tilt{index}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("entry/data/data", data=(counts + index).get())
        paths.append(path)
    loaded = io.load(paths, stack=False,
                     dataset_path="entry/data/data", scan_shape=(3, 91), verbose=False)
    try:
        assert bool(cp.all(cp.from_dlpack(loaded[0][:]) == counts))
        for path in paths:
            path.unlink()
        for index, tilt in enumerate(loaded):
            for row, column in ((2, 90), (0, 0)):
                assert bool(cp.all(
                    cp.from_dlpack(tilt[row, column]) == counts[row, column] + index
                ))
    finally:
        for tilt in loaded:
            tilt.close()


@pytest.mark.parametrize("representation", ["dense", "packed"])
def test_full_acquisition_overrides_fail_before_opening(tmp_path, representation):
    """File-backed acquisitions stay ANS encoded; obsolete modes fail early."""
    with pytest.raises(NotImplementedError, match="must remain ANS encoded"):
        io.load(
            tmp_path / "not-opened.h5", backend="cuda",
            representation=representation, verbose=False,
        )
