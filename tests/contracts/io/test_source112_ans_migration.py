"""Bounded native retained source112 archive conversion compared with original acquisitions."""

import os

import numpy as np
import pytest

from quantem.gpu.io import load
from quantem.gpu.io._ans import _write_ans_blocks


def _original_window(path, first, count):
    import h5py
    import hdf5plugin  # noqa: F401 - registers original compression filters

    blocks = []
    with h5py.File(path) as source:
        for key in sorted(source["entry/data"]):
            dataset = source["entry/data"][key]
            if first >= len(dataset):
                first -= len(dataset)
                continue
            take = min(count, len(dataset) - first)
            blocks.append(dataset[first:first + take])
            count -= take
            first = 0
            if not count:
                break
    if count:
        raise ValueError("Original acquisition ended before the selected validation window.")
    return np.concatenate(blocks)


def test_source112_archive_native_counts_and_canonical_conversion(tmp_path):
    archive = os.environ.get("QUANTEM_SOURCE112_ARCHIVE")
    if os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1" or not archive:
        pytest.skip("Set the CUDA gate and preserved retained source112 source112 archive path.")
    cp = pytest.importorskip("cupy")
    from quantem.gpu.io._source112_archive import _Source112Archive

    with _Source112Archive(archive, device=0) as source:
        for chunk, first in ((0, 0), (523, 12288), (1055, 15872)):
            decoded = source.decode_window(chunk, first)
            acquisition = source.manifest["original_acquisitions"]["acquisitions"][chunk // 16]
            expected = _original_window(
                acquisition["original_path_provenance"],
                chunk % 16 * 16384 + first, 512,
            )
            np.testing.assert_array_equal(decoded, expected)
            if chunk != 0:
                continue
            # This is an explicitly labeled validation window, not a reduced
            # substitute for the full-acquisition production converter.
            path = _write_ans_blocks(
                tmp_path / "validation-window.ans",
                iter((decoded[:256], decoded[256:])),
                shape=(1, 512, 192, 192), dtype=decoded.dtype,
                metadata={"legacy_validation_window": {"chunk": chunk, "first_scan": first}},
                block_frames=256, scale=15,
            )
            with cp.cuda.Device(0):
                migrated = load(path, backend="cuda", representation="encoded", device=0).data
                try:
                    actual = np.concatenate([
                        migrated.decode_block_device(block).get() for block in range(2)
                    ])
                    np.testing.assert_array_equal(actual, expected)
                finally:
                    migrated.release()
