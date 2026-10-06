"""``masked_sum_exact(output="native")`` gives exact integer counts or refuses, never a float image.

Native output is the source's own sum dtype on its device. CUDA count sources
return uint32/uint64 counts. Sources that hold float intensities (scaled
precision, float ANS, bounded float reads) have no exact sum, so both outputs
raise ``TypeError``; the native output once returned their float32 image.
"""

import os

import numpy as np
import pytest

from quantem.gpu import detector, io
from tests.hardware.test_bounded_detector_views import View
from tests.hardware.test_encoded_detector_moments import (
    DETECTOR_SHAPE,
    _counts,
    _load,
    _valid,
)


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return backend


def _disk(shape):
    rows, cols = np.indices(shape)
    return np.hypot(rows - shape[0] / 2, cols - shape[1] / 2) <= shape[0] / 4


def test_count_sources_keep_exact_counts_on_the_device(tmp_path):
    backend = _backend()
    counts = _counts(51)
    mask = _disk(DETECTOR_SHAPE)
    with _load(tmp_path, counts, backend) as loaded:
        session = detector.prepare(loaded)
        exact = session.masked_sum_exact(mask)
        np.testing.assert_array_equal(exact, (counts.astype(np.uint64) * _valid() * mask).sum(axis=(2, 3)))
        if backend == "mps":
            with pytest.raises(NotImplementedError, match="no native exact"):
                session.masked_sum_exact(mask, output="native")
            return
        native = session.masked_sum_exact(mask, output="native")
        # 672 uint16 pixels sum below 2^32, so the series keeps uint32 counts.
        assert native.dtype == np.uint32
        np.testing.assert_array_equal(native.get(), exact)


def test_float_sources_refuse_exact_sums_on_both_outputs(tmp_path):
    backend = _backend()
    values = (np.arange(4 * 4 * 16 * 16, dtype=np.float32).reshape(4, 4, 16, 16) % 37) / 4
    mask = _disk((16, 16))
    io.save(tmp_path / "float.qem", values, backend="cpu")
    np.save(tmp_path / "float.npy", values)
    with (
        io.load(tmp_path / "float.qem", backend=backend) as float_ans,
        io.load(tmp_path / "float.npy", dtype="scaled_uint16", backend=backend, verbose=False) as precision,
    ):
        for loaded in (float_ans, precision):
            session = detector.prepare(loaded)
            np.testing.assert_allclose(session.masked_sum(mask), values[:, :, mask].sum(-1), rtol=1e-4)
            for output in ("numpy", "native"):
                with pytest.raises(TypeError, match="float"):
                    session.masked_sum_exact(mask, output=output)


def test_bounded_float_reads_refuse_exact_sums_on_both_outputs(tmp_path):
    backend = _backend()
    with _load(tmp_path, _counts(53), backend) as loaded:
        session = detector.prepare(View(loaded, (4, 20, 9, 30)))
        for output in ("numpy", "native"):
            with pytest.raises(TypeError, match="float"):
                session.masked_sum_exact(_disk(DETECTOR_SHAPE), output=output)
