"""A bounded view of a host acquisition reduces in the NumPy reference backend, exactly, without reading the scan.

``io.load(path, backend="cpu")`` keeps the file's array on the host, and a
viewer opens it through the same bounded reader as a GPU acquisition. Its mean
pattern, virtual image and frame reductions come from ``ArrayBackend``: counts
in uint64, signed data in int64 (a uint64 total of dark-subtracted int16 wraps
to about 1.8e19) and floats in float64, each rounded once. This runs without a
GPU.
"""

import numpy as np
import pytest

from quantem.gpu import io
from tests.hardware.test_bounded_detector_views import View, _assert_session_reductions, dense_values


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("region", [(0, 32, 0, 32), (4, 20, 9, 30)], ids=["whole", "region"])
def test_host_acquisition_reduces_exactly(tmp_path, dtype, region):
    values = dense_values(dtype)
    np.save(tmp_path / "scan.npy", values)
    with io.load(tmp_path / "scan.npy", backend="cpu", verbose=False) as loaded:
        assert loaded.dtype == dtype
        _assert_session_reductions(View(loaded, region), values)
