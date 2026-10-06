"""Exact native HDF5 tilts remain independently available after loading."""

import os

import h5py
import pytest

from quantem.gpu import io

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)


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


@pytest.mark.parametrize("representation", ["dense"])
def test_full_acquisition_overrides_fail_before_opening(tmp_path, representation):
    """File-backed acquisitions stay ANS encoded; obsolete modes fail early."""
    with pytest.raises(NotImplementedError, match="must remain ANS encoded"):
        io.load(
            tmp_path / "not-opened.h5", backend="cuda",
            representation=representation, verbose=False,
        )
