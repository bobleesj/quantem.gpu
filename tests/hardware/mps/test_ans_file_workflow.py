"""Small physical file-to-GPU-to-detector workflow, not a speed benchmark."""

import hashlib

import numpy as np
import pytest

from quantem.gpu import detector, io


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_public_ans_file_packed_conversion_and_detector_parity(tmp_path, dtype):
    pytest.importorskip("Metal")
    counts = np.zeros((17, 19, 2, 3), dtype=dtype)
    counts[:, :, 0, 1] = np.arange(17 * 19).reshape(17, 19) % 4
    counts[1, 2, 1, 2] = np.iinfo(counts.dtype).max
    path = tmp_path / "exact.qgpu"
    io.save(
        path,
        counts,
        format="quantem",
        compression="ans",
        backend="cpu",
        batch_size=64,
        metadata={"calibration": {"detector_sampling_mrad": 0.05}},
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mask = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    expected_image = (counts * mask).sum(axis=(2, 3), dtype=np.uint64)
    with io.load(path, backend="mps", expected_source_sha256=digest) as source:
        assert source.representation.value == "encoded"
        with source.to_representation("packed") as packed:
            for loaded in (source, packed):
                assert loaded.dtype == counts.dtype
                assert loaded.shape == counts.shape
                session = detector.prepare(loaded)
                np.testing.assert_array_equal(session.frame(1 * 19 + 2), counts[1, 2])
                np.testing.assert_array_equal(
                    session.masked_sum_exact(mask), expected_image
                )
                np.testing.assert_array_equal(
                    session.masked_sum_exact(np.zeros((2, 3))), 0
                )
                assert loaded.metadata["calibration"] == {
                    "detector_sampling_mrad": 0.05
                }
            source.close()
            np.testing.assert_array_equal(
                detector.prepare(packed).frame(21), counts[1, 2]
            )
    with io.load(path, representation="packed", backend="mps") as loaded:
        np.testing.assert_array_equal(
            detector.prepare(loaded).masked_sum_exact(mask), expected_image
        )


def test_public_ans_series_keeps_sources_independent_and_exact(tmp_path):
    """A compatible ANS list loads without a dense stack and preserves axes."""
    pytest.importorskip("Metal")
    counts = [
        np.arange(8 * 8 * 2 * 3, dtype=np.uint16).reshape(8, 8, 2, 3),
        np.arange(8 * 8 * 2 * 3, dtype=np.uint16).reshape(8, 8, 2, 3) + 100,
        np.arange(8 * 8 * 2 * 3, dtype=np.uint16).reshape(8, 8, 2, 3) + 200,
    ]
    paths = []
    for index, values in enumerate(counts):
        path = tmp_path / f"series-{index}.qgpu"
        io.save(path, values, format="quantem", compression="ans", backend="cpu")
        paths.append(path)

    loaded = io.load(
        paths,
        backend="mps",
        representation="encoded",
        stack=False,
        verbose=False,
    )
    try:
        session = detector.prepare(loaded)
        assert session.series_shape == (3,)
        np.testing.assert_array_equal(
            session.frame(17), np.stack([values.reshape(64, 2, 3)[17] for values in counts])
        )
        mask = np.ones((2, 3), dtype=np.uint8)
        expected = np.stack(
            [values.sum(axis=(2, 3), dtype=np.uint64) for values in counts], axis=0
        )
        np.testing.assert_array_equal(session.masked_sum_exact(mask), expected)
    finally:
        for item in loaded:
            item.close()
