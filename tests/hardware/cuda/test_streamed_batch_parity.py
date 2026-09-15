"""Larger loader batches must retain every count and detector product."""

import os
import importlib
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io import _streamed
from quantem.gpu.io._hdf5_chunk_index import _chunk_locations_python


def test_full_acquisition_batch_parity(monkeypatch):
    """Compare all corrected counts and translated masks across batch sizes.

    Set QGPU_TEST_H5 to a real native-count HDF5 acquisition. Both sources
    remain compressed; comparison decodes only 1024 scan positions at once.
    """
    path = os.environ.get("QGPU_TEST_H5")
    if not path or not Path(path).is_file():
        pytest.skip("Set QGPU_TEST_H5 to a real HDF5 acquisition.")
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("Requires CUDA.")
    loader = importlib.import_module("quantem.gpu.io.load")
    monkeypatch.setenv(loader._FRAME_SOURCE_CACHE_ENV, "")
    loader._MASTER_FRAME_SOURCE_CACHE.clear()
    with monkeypatch.context() as patch:
        patch.setattr(loader, "_chunk_locations", _chunk_locations_python)
        patch.setattr(_streamed, "_CUDA_MAX_STAGING_SCANS", 2048)
        patch.setattr(loader, "_HDF5_READ_TASK_BYTES", 2**63 - 1)
        patch.setattr(loader, "_parse_headers_bulk", lambda *args, **kwargs: loader._parse_headers(*args))
        old = io.load(path, backend="cuda", representation="encoded", verbose=False)
    try:
        loader._MASTER_FRAME_SOURCE_CACHE.clear()
        new = io.load(path, backend="cuda", representation="encoded", verbose=False)
        try:
            assert new.shape == old.shape
            assert new.dtype == old.dtype
            assert new.metadata["hot_pixel_correction"] == old.metadata["hot_pixel_correction"]
            scans = int(np.prod(new.shape[:2]))
            for first in range(0, scans, 1024):
                stop = min(first + 1024, scans)
                before = old.data.decode_scan_range_device(first, stop)
                after = new.data.decode_scan_range_device(first, stop)
                assert bool(cp.array_equal(before, after)), (first, stop)
                del before, after
            a, b = detector.prepare([old]), detector.prepare([new])
            try:
                rows, cols = np.indices(new.shape[-2:])
                for offset, inner, outer in [(0, 0, 50), (0, 50, 100), (17, 45, 90)]:
                    radius = (rows - rows.shape[0] / 2 - offset)**2 + (cols - cols.shape[1] / 2)**2
                    mask = (radius >= inner**2) & (radius < outer**2)
                    np.testing.assert_array_equal(a.masked_sum(mask), b.masked_sum(mask))
            finally:
                a.close()
                b.close()
        finally:
            new.close()
    finally:
        old.close()
