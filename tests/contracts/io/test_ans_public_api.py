"""Public exact file round trips and honest unsupported conversion directions."""

import hashlib

import numpy as np
import pytest

from quantem.gpu import io


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_public_ans_save_and_dense_reference_load_preserve_provenance(tmp_path, dtype):
    values = np.arange(5 * 7 * 2 * 3, dtype=dtype).reshape(5, 7, 2, 3)
    values[0, 0, 0, 0] = np.iinfo(values.dtype).max
    metadata = {
        "source_shape": [5, 7, 4, 6],
        "working_shape": list(values.shape),
        "source_dtype": dtype,
        "working_dtype": dtype,
        "source_logical_tensor_bytes": 5 * 7 * 4 * 6 * values.dtype.itemsize,
        "scan_bin": 1,
        "detector_bin": 2,
        "crop": None,
        "calibration": {"scan_step_nm": 0.2, "working_detector_step_mrad": 0.1},
        "lossless_exact": True,
        "source_path": "original-master.h5",
        "source_file_sha256": "a" * 64,
    }
    source = io.FourDSTEMData(values, metadata)
    path = tmp_path / "exact.qgpu"
    saved = io.save(path, source, format="quantem", compression="ans", backend="cpu")
    assert saved.complete and saved.path == str(path)
    assert path.read_bytes()[:8] == b"QGANS\0\1\0"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with io.load(
        saved.path, representation="dense", backend="cpu", expected_source_sha256=digest
    ) as loaded:
        np.testing.assert_array_equal(loaded.data, values)
        assert loaded.dtype == values.dtype
        assert loaded.logical_bytes == values.nbytes
        assert loaded.metadata["source_shape"] == [5, 7, 4, 6]
        assert loaded.metadata["detector_bin"] == 2
        assert loaded.metadata["calibration"] == metadata["calibration"]
        assert loaded.metadata["file_counts_exact"]
        assert loaded.metadata["container_authentication"] == "external-sha256"
        assert loaded.metadata["detector_mask_policy"] == "preserve-stored-counts"
        assert loaded.metadata["source_path"] == "original-master.h5"
        assert loaded.metadata["source_file_sha256"] == "a" * 64
        assert loaded.metadata["container_sha256"] == digest
        assert loaded.metadata["logical_count_hash_verified"]
        assert loaded.to_representation("dense") is loaded
        with pytest.raises(NotImplementedError, match="not implemented"):
            loaded.to_representation("ans")
    assert source.metadata == metadata


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_quantem_auto_and_explicit_compression_write_identical_files(tmp_path, dtype):
    counts = np.zeros((17, 19, 2, 3), dtype=dtype)
    counts[:, :, 0, 1] = np.arange(17 * 19).reshape(17, 19) % 4
    counts[1, 2, 1, 2] = np.iinfo(counts.dtype).max
    metadata = {"calibration": {"detector_sampling_mrad": 0.05}}
    paths = []
    for label, options in (
        ("explicit.qgpu", {"format": "quantem", "compression": "ans"}),
        ("uppercase.qgpu", {"format": "QUANTEM", "compression": "ANS"}),
        ("automatic.qgpu", {"format": "quantem"}),
    ):
        path = tmp_path / label
        io.save(path, counts, backend="cpu", metadata=metadata, **options)
        paths.append(path)
    for path in paths:
        assert path.read_bytes() == paths[0].read_bytes()
        assert io.DataRepresentation.detect_source(path).value == "ans"
        with io.load(path, representation="dense", backend="cpu") as loaded:
            np.testing.assert_array_equal(loaded.data, counts)
            assert loaded.metadata["calibration"] == metadata["calibration"]


def test_compression_cannot_silently_choose_another_container(tmp_path):
    counts = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    for file_format, compression, correction in (
        ("arina", "ans", "format='quantem'"),
        ("quantem", "bitshuffle_lz4", "compression='ans'"),
    ):
        path = tmp_path / f"{file_format}.data"
        with pytest.raises(ValueError, match=correction):
            io.save(
                path, counts, format=file_format, compression=compression, backend="cpu"
            )
        assert not path.exists()


def test_source_memory_is_distinct_from_stored_memory(tmp_path):
    values = np.zeros((2, 3, 2, 2), dtype=np.uint16)
    metadata = {
        "source_shape": [2, 3, 4, 4],
        "source_dtype": "uint16",
        "detector_bin": 2,
    }
    path = tmp_path / "bin2.ans"
    io.save(
        path,
        values,
        metadata=metadata,
        format="quantem",
        compression="ans",
        backend="cpu",
    )
    with io.load(path, representation="dense", backend="cpu") as loaded:
        assert loaded.metadata["source_logical_tensor_bytes"] == 2 * 3 * 4 * 4 * 2
        assert loaded.metadata["working_logical_tensor_bytes"] == values.nbytes
        assert loaded.resident_bytes == values.nbytes


def test_new_counts_controls_fail_before_implicit_transform(tmp_path):
    values = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    saved = io.save(
        tmp_path / "source.ans",
        values,
        format="quantem",
        compression="ans",
        backend="cpu",
    )
    for options, exception, pattern in [
        ({"apply_mask": True}, ValueError, "original counts"),
        ({"dtype": "uint16"}, ValueError, "native integer dtype"),
        ({"scan_indices": [0]}, NotImplementedError, "geometry"),
        ({"detector_bin": 2}, NotImplementedError, "geometry"),
    ]:
        with pytest.raises(exception, match=pattern):
            io.load(saved.path, representation="dense", backend="cpu", **options)
    with pytest.raises(NotImplementedError, match="CPU reference"):
        io.load(saved.path, representation="ans", backend="cpu")
    with pytest.raises(NotImplementedError, match="reference encoder"):
        io.save(
            tmp_path / "no-gpu-fallback.ans",
            values,
            format="quantem",
            compression="ans",
        )
    assert not (tmp_path / "no-gpu-fallback.ans").exists()


def test_ans_request_requires_an_ans_source(tmp_path):
    prepared = tmp_path / "packed.h5"
    prepared.write_bytes(b"QGPUH5\0\x01")
    assert io.DataRepresentation.detect_source(prepared).value == "packed"
    with pytest.raises(NotImplementedError, match="requires an ANS source"):
        io.load(prepared, representation="ans")


def test_removed_names_are_rejected_before_saving_or_loading(tmp_path):
    counts = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    for name in ("ans", "arina-h5", "arina_h5", "h5"):
        with pytest.raises(ValueError, match="use format='arina' or 'quantem'"):
            io.save(tmp_path / "unused", counts, format=name, backend="cpu")
    with pytest.raises(ValueError, match="representation must be one of"):
        io.load(tmp_path / "unused", representation="lossless_packed")
    assert not (tmp_path / "unused").exists()


def test_failed_conversion_publication_releases_output_not_source():
    class Output:
        released = False

        @property
        def nbytes(self):
            raise RuntimeError("injected publication failure")

        def release(self):
            self.released = True

    output = Output()

    class Source:
        def to_packed(self):
            return output

        def release(self):
            raise AssertionError("caller-owned input must not be released")

    loaded = io.FourDSTEMData(Source(), {"representation": "ans"})
    with pytest.raises(RuntimeError, match="publication failure"):
        loaded.to_representation("packed")
    assert output.released
