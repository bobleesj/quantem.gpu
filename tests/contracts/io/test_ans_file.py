"""Self-contained ANS IO checked against a separately written inverse."""

import hashlib
import json
import struct

import numpy as np
import pytest

from quantem.gpu.io._ans import ANSFile, write_ans_reference
from tests.parity.ans_counts_fixture import decode_reference


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("block_frames", [1, 32, 256])
def test_exact_file_roundtrip(tmp_path, dtype, block_frames):
    data = np.zeros((17, 19, 3, 5), dtype=dtype)
    rng = np.random.default_rng(713)
    data[:, :, 1, 2] = rng.integers(0, 4, size=data.shape[:2])
    data[1, 2, 2, 4] = np.iinfo(data.dtype).max
    data[4, 5, 0, 0] = 217
    metadata = {
        "calibration": {"scan_step_nm": 0.12, "detector_step_mrad": 0.04},
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "excluded_detector_pixels": [14],
        "source": "independent-test",
    }
    path = write_ans_reference(
        tmp_path / "counts.ans", data, metadata=metadata, block_frames=block_frames
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with ANSFile(path, expected_sha256=digest) as source:
        parameters = source.runtime_arguments()
        # Independent fixture decoder never imports the production recurrence.
        np.testing.assert_array_equal(decode_reference(parameters), data)
        actual = np.concatenate(
            [
                source.decode_block_reference(block)
                for block in range((17 * 19 + block_frames - 1) // block_frames)
            ]
        ).reshape(data.shape)
        np.testing.assert_array_equal(actual, data)
        assert actual.dtype == data.dtype
        assert source.metadata == metadata
        assert (
            source.manifest["logical_sha256"]
            == hashlib.sha256(data.tobytes()).hexdigest()
        )
        assert source.logical_nbytes == data.nbytes
        if block_frames == 256:
            assert 0 in source.arrays["literal"]  # Entropy as well as literal coverage.
        assert source.encoded_nbytes == sum(
            array.nbytes for array in source.arrays.values()
        )
    with pytest.raises(RuntimeError, match="closed"):
        source.runtime_arguments()


def test_noncontiguous_and_transactional_output(tmp_path):
    data = np.arange(2 * 7 * 3 * 5, dtype=np.uint16).reshape(2, 7, 3, 5)[:, ::2]
    path = write_ans_reference(tmp_path / "stride.ans", data)
    with ANSFile(path) as source:
        np.testing.assert_array_equal(
            decode_reference(source.runtime_arguments()), data
        )
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_ans_reference(path, data)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="64 KiB"):
        write_ans_reference(
            tmp_path / "large-meta.ans", data, metadata={"oversized": "a" * 70000}
        )
    assert not (tmp_path / "large-meta.ans").exists()
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(ValueError, match="uint8/uint16"):
        write_ans_reference(tmp_path / "float.ans", data.astype(np.float32))
    with pytest.raises(ValueError, match="block_frames"):
        write_ans_reference(tmp_path / "bool.ans", data, block_frames=True)


def _rewrite_manifest(path, change):
    raw = bytearray(path.read_bytes())
    magic, length, start = struct.unpack_from("<8sQQ", raw)
    document = json.loads(raw[24 : 24 + length])
    change(document)
    encoded = json.dumps(document, separators=(",", ":")).encode()
    raw[:24] = struct.pack("<8sQQ", magic, len(encoded), start)
    raw[24:start] = encoded + bytes(start - 24 - len(encoded))
    path.write_bytes(raw)


def test_corruption_and_bounds_fail_closed(tmp_path):
    data = np.zeros((2, 2, 3, 5), dtype=np.uint16)
    for label, change, message in [
        ("bounds", lambda m: m["sections"]["payload"].update(offset=2**63), "bounds"),
        ("shape", lambda m: m.update(shape=[True, 2, 3, 5]), "shape"),
        ("dtype", lambda m: m.update(dtype="float16"), "dtype"),
        ("scale", lambda m: m.update(scale=True), "scale"),
        ("section", lambda m: m["sections"].pop("offsets"), "sections"),
    ]:
        path = write_ans_reference(tmp_path / f"{label}.ans", data)
        _rewrite_manifest(path, change)
        with pytest.raises(ValueError, match=message):
            ANSFile(path)
    path = write_ans_reference(tmp_path / "checksum.ans", data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = bytearray(path.read_bytes())
    raw[65536] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        ANSFile(path)
    with pytest.raises(ValueError, match="SHA-256"):
        ANSFile(path, expected_sha256=digest)


def test_resealed_literal_out_of_range_is_not_accepted(tmp_path):
    data = np.full((2, 2, 1, 1), 65535, dtype=np.uint16)
    path = write_ans_reference(tmp_path / "lying.ans", data)
    _rewrite_manifest(path, lambda m: m.update(dtype="uint8"))
    # File/table integrity alone is not numerical stream qualification.
    with ANSFile(path) as source, pytest.raises(ValueError, match="literal count"):
        source.decode_block_reference(0)


@pytest.mark.parametrize(
    "metadata",
    [
        {"detector_bin": 0},
        {"detector_bin": True},
        {"scan_bin": 2},
        {"source_shape": [2, 2, 3, 5]},
        {"working_dtype": "uint8"},
        {"lossless_exact": "False"},
        {"working_logical_tensor_bytes": 0},
    ],
)
def test_invalid_scientific_metadata_is_not_persisted(tmp_path, metadata):
    data = np.zeros((2, 2, 2, 3), dtype=np.uint16)
    path = tmp_path / "invalid.ans"
    with pytest.raises(ValueError, match="ANS|Count-ANS"):
        write_ans_reference(path, data, metadata=metadata)
    assert not path.exists()


def test_source_mutation_is_rejected_before_publication(tmp_path):
    path = write_ans_reference(
        tmp_path / "mutable.ans", np.zeros((2, 2, 1, 1), dtype=np.uint8)
    )
    with ANSFile(path) as source:
        with path.open("r+b") as writer:
            writer.seek(65536)
            writer.write(b"\xff")
            writer.flush()
        with pytest.raises(ValueError, match="changed during"):
            source.assert_unchanged()
