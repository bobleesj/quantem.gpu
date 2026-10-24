"""A waiting acquisition worker prepares code without replacing native counts."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def test_prepared_loader_preserves_uint16_acquisitions(tmp_path):
    import h5py
    import hdf5plugin

    from quantem.gpu import io
    from quantem.gpu.io.encoded import _prepare_cuda_loader

    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    values = np.arange(4 * 8 * 16 * 16, dtype=np.uint16).reshape(4, 8, 16, 16)
    values[0, 0, 0, 0] = 65535
    source = tmp_path / "scan_master.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "entry/data/data", data=values.reshape(32, 16, 16),
            chunks=(1, 16, 16), **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    for _ in range(2):
        _prepare_cuda_loader()
        with io.load(source, backend="cuda", scan_shape=(4, 8), verbose=False) as data:
            actual = cp.asnumpy(cp.from_dlpack(data.read()))
            np.testing.assert_array_equal(actual, values)
