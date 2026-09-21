"""Exercise the default compressed acquisition workflow on physical devices."""

import os

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io import _array_resident, _qem_reference


@pytest.fixture
def backend():
    selected = os.environ.get("QEM_TEST_BACKEND")
    if selected not in {"cuda", "mps"}:
        pytest.skip("Run check_ans_io.py on a physical CUDA or MPS device.")
    return selected


def test_original_hdf5_default_ans_roundtrip(tmp_path, backend):
    """Open a chunked detector master and keep native counts through export."""
    values = np.random.default_rng(42).integers(0, 1000, (32, 32, 8, 12), np.uint16)
    values[-1, -1, -1, -1] = 65535
    original, saved = tmp_path / "scan_master.h5", tmp_path / "copy.qem"
    with h5py.File(original, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=values.reshape(-1, 8, 12),
            chunks=(1, 8, 12),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = 1024
        handle["entry/instrument/detector/count_time"] = 0.00005
    with io.load(original, backend=backend, apply_mask=False, verbose=False) as loaded:
        assert loaded.representation.value == "encoded"
        assert loaded.metadata["backend"] == backend
        session = detector.prepare(loaded)
        np.testing.assert_array_equal(session.frame(1023), values[-1, -1])
        np.testing.assert_allclose(session.mean_dp(), values.mean((0, 1)), rtol=1e-6)
        np.testing.assert_array_equal(
            session.masked_sum(np.ones((8, 12), bool)), values.sum((2, 3))
        )
        io.save(saved, loaded)
    with io.load(saved, backend=backend, verbose=False) as reopened:
        assert reopened.representation.value == "encoded"
        np.testing.assert_array_equal(
            detector.prepare(reopened).frame(1023), values[-1, -1]
        )
        assert (
            reopened.metadata["scientific_metadata"]["source_metadata"][
                "entry/instrument/detector/count_time"
            ]
            == 0.00005
        )


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "float32"])
def test_default_array_ingestion_is_bounded_ans_only(
    tmp_path, backend, monkeypatch, dtype
):
    """Browse arrays without a full-cube upload or a reference-encoder fallback."""
    values = (np.arange(35 * 8 * 12).reshape(5, 7, 8, 12) % 97).astype(dtype)
    if dtype == "float32":
        values /= 8
    original = tmp_path / "scan.npy"
    np.save(original, values)
    limit = 3 * 8 * 12 * 4
    monkeypatch.setattr(_array_resident, "MAX_INGEST_BYTES", limit)

    def forbid_reference(*args, **kwargs):
        raise AssertionError(
            "Default ingestion must never use the CPU reference encoder."
        )

    monkeypatch.setattr(_qem_reference, "_encode_stream", forbid_reference)
    transfers = []

    def bounded_upload(upload):
        def check(block, *args, **kwargs):
            if isinstance(block, np.ndarray) and block.ndim >= 3:
                transfers.append(block.nbytes)
                assert (
                    block.nbytes <= limit
                ), "A full acquisition was uploaded before being sliced."
            return upload(block, *args, **kwargs)

        return check

    if backend == "mps":
        from quantem.gpu.io.backends.mps import precision
        from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts

        monkeypatch.setattr(precision, "upload", bounded_upload(precision.upload))
        resident_type = MPSStreamedCounts
    else:
        import cupy as cp
        from quantem.gpu._compact.streamed import StreamedCounts

        monkeypatch.setattr(cp, "asarray", bounded_upload(cp.asarray))
        resident_type = StreamedCounts
    append = resident_type.append
    uploads = []

    def bounded_append(self, block, *args, **kwargs):
        size = int(np.prod(block.shape)) * np.dtype(block.dtype).itemsize
        uploads.append(size)
        assert (
            size <= limit
        ), "A full acquisition reached the encoder instead of bounded input."
        return append(self, block, *args, **kwargs)

    monkeypatch.setattr(resident_type, "append", bounded_append)
    with io.load(original, backend=backend, verbose=False) as loaded:
        assert loaded.representation.value == "encoded"
        assert loaded.metadata["backend"] == backend
        assert loaded.metadata["load_timings"]["peak_ingest_bytes"] <= limit
        np.testing.assert_array_equal(
            detector.prepare(loaded).frame(34), values[-1, -1]
        )
    assert len(uploads) > 1
    assert len(transfers) > 1
    count = len(uploads)
    for representation in ("dense", "packed"):
        with pytest.raises(NotImplementedError, match="encoded|ANS"):
            io.load(
                original, backend=backend, representation=representation, verbose=False
            )
    assert (
        len(uploads) == count
    ), "Unsupported residency must fail before uploading measurements."
