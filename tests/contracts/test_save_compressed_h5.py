import h5py
import numpy as np
import pytest

from quantem.gpu.io.save import save_compressed_arina_h5


def test_save_compressed_arina_h5_uses_master_data_layout(tmp_path):
    data = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)

    master = tmp_path / "merged_master.h5"
    save_compressed_arina_h5(
        master,
        data,
        dtype="u16",
        batch_size=2,
        frames_per_file=4,
        metadata={"det_bin": 1},
    )

    data1 = tmp_path / "merged_data_000001.h5"
    data2 = tmp_path / "merged_data_000002.h5"
    assert data1.exists()
    assert data2.exists()

    with h5py.File(master, "r") as handle:
        assert handle.attrs["scan_shape"].tolist() == [3, 2]
        assert handle.attrs["detector_shape"].tolist() == [4, 5]
        assert handle.attrs["det_bin"] == 1
        group = handle["entry/data"]
        assert group.attrs["n_frames"] == 6
        assert isinstance(group.get("data_000001", getlink=True), h5py.ExternalLink)
        assert isinstance(group.get("data_000002", getlink=True), h5py.ExternalLink)

    with h5py.File(data1, "r") as handle:
        dset = handle["entry/data/data"]
        assert dset.shape == (4, 4, 5)
        assert dset.chunks == (1, 4, 5)
        assert dset.dtype == np.dtype("uint16")
        assert dset.id.get_create_plist().get_nfilters() > 0
        np.testing.assert_array_equal(dset[:], data.reshape(6, 4, 5)[:4].astype(np.uint16))

    with h5py.File(data2, "r") as handle:
        dset = handle["entry/data/data"]
        assert dset.shape == (2, 4, 5)
        assert dset.chunks == (1, 4, 5)
        np.testing.assert_array_equal(dset[:], data.reshape(6, 4, 5)[4:].astype(np.uint16))


def test_save_compressed_arina_h5_uint8_display_export(tmp_path):
    data = np.array([[[0.2, 12.6], [255.4, 400.0]]], dtype=np.float32)

    master = tmp_path / "browse_master.h5"
    save_compressed_arina_h5(
        master,
        data,
        scan_shape=(1, 1),
        dtype="u8",
        compression_backend="hdf5",
    )

    data_file = tmp_path / "browse_data_000001.h5"
    with h5py.File(data_file, "r") as handle:
        dset = handle["entry/data/data"]
        assert dset.dtype == np.dtype("uint8")
        np.testing.assert_array_equal(
            dset[:],
            np.array([[[0, 13], [255, 255]]], dtype=np.uint8),
        )


def test_save_public_api_routes_numpy_to_arina_h5(tmp_path):
    from quantem.gpu.io import save

    data = np.array(
        [
            [[[0.2, 12.6], [255.4, 400.0]]],
            [[[2.5, 3.5], [-4.0, 4.49]]],
        ],
        dtype=np.float32,
    )

    master = tmp_path / "public_master.h5"
    result = save(
        master,
        data,
        dtype="u16",
        backend="cpu",
        frames_per_file=1,
        wait=False,
    )
    assert result.complete is True
    assert result.wait() is result

    with h5py.File(master, "r") as handle:
        assert handle.attrs["scan_shape"].tolist() == [2, 1]
        group = handle["entry/data"]
        assert isinstance(group.get("data_000001", getlink=True), h5py.ExternalLink)

    with h5py.File(tmp_path / "public_data_000001.h5", "r") as handle:
        dset = handle["entry/data/data"]
        assert dset.dtype == np.dtype("uint16")
        np.testing.assert_array_equal(
            dset[:],
            np.array([[[0, 13], [255, 400]]], dtype=np.uint16),
        )


def test_save_public_api_rejects_cuda_backend_for_numpy(tmp_path):
    from quantem.gpu.io import save

    data = np.zeros((1, 1, 2, 2), dtype=np.uint16)

    with pytest.raises(ValueError, match="backend='cuda' requires"):
        save(tmp_path / "wrong_master.h5", data, backend="cuda")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_arina_compression_choices_preserve_bitshuffle_chunks_and_counts(tmp_path, dtype):
    from quantem.gpu.io import save

    data = np.arange(3 * 2 * 4 * 5, dtype=dtype).reshape(3, 2, 4, 5)
    data[0, 0, 0, 0] = np.iinfo(dtype).max
    reference = None
    for label, options in (
        ("default", {}),
        ("arina", {"format": "arina", "compression": "bitshuffle_lz4"}),
        ("explicit_lz4", {"format": "arina", "compression": "lz4"}),
        ("explicit_bslz4", {"format": "arina", "compression": "bslz4"}),
    ):
        master = tmp_path / f"{label}_master.h5"
        save(
            master,
            data,
            backend="cpu",
            frames_per_file=4,
            metadata={"detector_step_mrad": 0.05},
            **options,
        )
        frames, filters, chunks = [], [], []
        with h5py.File(master, "r") as handle:
            assert handle.attrs["scan_shape"].tolist() == [3, 2]
            assert handle.attrs["detector_step_mrad"] == 0.05
            for dataset in handle["entry/data"].values():
                assert dataset.dtype == np.dtype(dtype)
                assert dataset.chunks == (1, 4, 5)
                frames.append(dataset[:])
                filters.append(dataset.id.get_create_plist().get_filter(0))
                chunks.extend(
                    dataset.id.read_direct_chunk((frame, 0, 0))
                    for frame in range(dataset.shape[0])
                )
        np.testing.assert_array_equal(np.concatenate(frames), data.reshape(6, 4, 5))
        if reference is None:
            reference = filters, chunks
        assert (filters, chunks) == reference


def test_io_save_package_attribute_remains_callable_after_submodule_import():
    import quantem.gpu.io as io
    import quantem.gpu.io.save  # noqa: F401

    assert callable(io.save)


def test_cuda_save_uint8_display_export(tmp_path):
    cp = pytest.importorskip("cupy")

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime is not available: {exc}")

    from quantem.gpu.io import save

    data = np.array(
        [
            [[[0.2, 12.6, 255.4, 400.0], [2.5, 3.5, -4.0, 4.49]]],
            [[[7.1, 8.9, 9.2, 10.8], [11.0, 12.2, 13.6, 14.4]]],
        ],
        dtype=np.float32,
    )

    master = tmp_path / "cuda_browse_master.h5"
    save(
        str(master),
        cp.asarray(data),
        dtype="u8",
        batch_size=1,
        frames_per_file=1,
    )

    with h5py.File(tmp_path / "cuda_browse_data_000001.h5", "r") as handle:
        dset = handle["entry/data/data"]
        assert dset.dtype == np.dtype("uint8")
        np.testing.assert_array_equal(
            dset[:],
            np.array([[[0, 13, 255, 255], [2, 4, 0, 4]]], dtype=np.uint8),
        )


@pytest.mark.parametrize("backend", ["cpu", "cuda", "mps"])
def test_narrower_integer_save_is_exact_or_refused(tmp_path, backend):
    """uint16 counts saved as uint8 are written unchanged when they fit and refused when one does not.

    A plain cast wrote 300 as 44 (300 mod 256) without a message.
    """
    from quantem.gpu import io

    counts = np.arange(16, dtype=np.uint16).reshape(1, 1, 4, 4) * 20
    if backend == "cuda":
        cp = pytest.importorskip("cupy")
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device is not available.")
        convert = cp.asarray
    elif backend == "mps":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS device is not available.")

        def convert(values):
            return torch.from_numpy(values).to("mps")
    else:
        convert = np.asarray
    with pytest.raises(ValueError, match="outside the uint8 range"):
        io.save(str(tmp_path / "wrapped_master.h5"), convert(counts), dtype="u8", backend=backend, verbose=False)
    fitting = np.minimum(counts, 255)
    io.save(str(tmp_path / "exact_master.h5"), convert(fitting), dtype="u8", backend=backend, verbose=False)
    with h5py.File(tmp_path / "exact_data_000001.h5", "r") as handle:
        np.testing.assert_array_equal(handle["entry/data/data"][:], fitting.reshape(1, 4, 4).astype(np.uint8))
