from __future__ import annotations

import h5py
import numpy as np
import pytest

from quantem.gpu.io.save import (
    save_compressed_arina_h5,
    save_compressed_h5,
    write_compressed_h5_dataset,
)


def test_save_compressed_h5_rounds_float_to_uint16(tmp_path):
    data = np.array(
        [
            [[[0.2, 1.6], [65534.6, 70000.0]]],
            [[[2.5, 3.5], [-4.0, 4.49]]],
        ],
        dtype=np.float32,
    )

    path = tmp_path / "portable.h5"
    save_compressed_h5(
        path,
        {"merged": data},
        dtype="u16",
        chunks=(1, 1, 2, 2),
        batch_size=1,
        compression="gzip",
        metadata={"kind": "test"},
        dataset_metadata={"merged": {"quantization": "rint"}},
    )

    with h5py.File(path, "r") as handle:
        dset = handle["merged"]
        assert handle.attrs["kind"] == "test"
        assert dset.attrs["quantization"] == "rint"
        assert dset.dtype == np.dtype("uint16")
        assert dset.compression == "gzip"
        np.testing.assert_array_equal(
            dset[:],
            np.array(
                [
                    [[[0, 2], [65535, 65535]]],
                    [[[2, 4], [0, 4]]],
                ],
                dtype=np.uint16,
            ),
        )


def test_write_compressed_h5_dataset_accepts_existing_handle(tmp_path):
    data = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)

    path = tmp_path / "existing.h5"
    with h5py.File(path, "w") as handle:
        write_compressed_h5_dataset(
            handle,
            "before",
            data,
            dtype="u16",
            chunks=(1, 2, 2),
            batch_size=1,
            compression="lzf",
        )

    with h5py.File(path, "r") as handle:
        assert handle["before"].compression == "lzf"
        np.testing.assert_array_equal(handle["before"][:], data)


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
