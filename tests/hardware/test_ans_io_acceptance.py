"""Exercise the default compressed acquisition workflow on physical devices."""

import os

import h5py
import hdf5plugin
import numpy as np
import pytest

from quantem.gpu import detector, io
from quantem.gpu.io import arrays
from quantem.gpu.formats.qem import reference


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


def test_release_rows_before_frees_rows_already_read(tmp_path, backend):
    """A consumer reading the scan once from the top frees the rows behind it; the rest stay exact."""
    values = np.random.default_rng(7).integers(0, 1000, (256, 256, 4, 4), np.uint16)
    original = tmp_path / "scan_master.h5"
    with h5py.File(original, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=values.reshape(-1, 4, 4),
            chunks=(1, 4, 4),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    with io.load(original, backend=backend, apply_mask=False, verbose=False) as loaded:
        loaded.release_rows_before(128)
        # whole stored chunks are freed, never a row at or below the requested one
        assert 0 < loaded.data.released_scans <= 128 * 256
        np.testing.assert_array_equal(
            loaded.read(scan_region=(128, 256, 0, 256)).cpu().numpy(), values[128:]
        )
        with pytest.raises(ValueError, match="released"):
            loaded.read(scan_region=(0, 1, 0, 256))


@pytest.mark.parametrize("scaled", [False, True])
def test_multiple_acquisitions_load_without_requesting_a_stack(tmp_path, backend, scaled):
    """Open two saved scans and retain their separate counts or calibrations."""
    paths = [tmp_path / "first.h5", tmp_path / "second.h5"]
    originals = [
        (np.arange(32 * 32 * 8 * 12).reshape(32, 32, 8, 12) % 97 + offset)
        .astype(np.float32 if scaled else np.uint16)
        for offset in (0, 200)
    ]
    for path, values in zip(paths, originals):
        if scaled:
            io.save(path, values, dtype="scaled_uint16", backend=backend, verbose=False)
        else:
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "entry/data/data", data=values.reshape(-1, 8, 12),
                    chunks=(1, 8, 12), **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
                )
                handle["entry/instrument/detector/detectorSpecific/ntrigger"] = 1024
    acquisitions = io.load(paths, backend=backend, verbose=False)
    try:
        assert isinstance(acquisitions, list) and len(acquisitions) == 2
        for acquisition, original in zip(acquisitions, originals):
            observed = acquisition.read().cpu().numpy()
            if scaled:
                np.testing.assert_allclose(observed, original, rtol=0, atol=0.001)
            else:
                np.testing.assert_array_equal(observed, original)
    finally:
        for acquisition in acquisitions:
            acquisition.close()


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
    monkeypatch.setattr(arrays, "MAX_INGEST_BYTES", limit)

    def forbid_reference(*args, **kwargs):
        raise AssertionError(
            "Default ingestion must never use the CPU reference encoder."
        )

    monkeypatch.setattr(reference, "_encode_stream", forbid_reference)
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
        from quantem.gpu.resident.mps import precision
        from quantem.gpu.resident.mps.counts import MPSStreamedCounts

        monkeypatch.setattr(precision, "upload", bounded_upload(precision.upload))
        resident_type = MPSStreamedCounts
    else:
        import cupy as cp
        from quantem.gpu.resident.cuda.counts import StreamedCounts

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
    with pytest.raises(NotImplementedError, match="encoded|ANS"):
        io.load(original, backend=backend, representation="dense", verbose=False)
    assert (
        len(uploads) == count
    ), "Unsupported residency must fail before uploading measurements."
