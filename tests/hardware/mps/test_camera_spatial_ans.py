"""Exact camera encode, spatial-query and saved-snapshot workflows on Metal."""

from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu import io


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_native_camera_spatial_roundtrip(tmp_path, dtype):
    """Retain tails, saturated counts and validity through encode/save/reopen."""
    pytest.importorskip("Metal")
    from quantem.gpu.io.backends.mps._spatial import build_index
    from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts, _upload
    from quantem.gpu.io.backends.mps.packed import _release

    rng = np.random.default_rng(20260915)
    shape = (1, 519, 37, 43)
    counts = rng.poisson(0.3, shape).astype(dtype)
    counts[:, :, 0, 0] = 0
    counts[:, :, 0, 1] = 7
    counts[:, :, 0, 2] = np.iinfo(dtype).max
    counts[0, 511:, 2, 3] = np.iinfo(dtype).max
    valid = rng.random(shape[2:]) > 0.08
    resident = MPSStreamedCounts(shape, np.dtype(dtype), valid)
    resident.spatial_chunks = []
    try:
        for first in (0, 512):
            raw = np.ascontiguousarray(counts[0, first : first + 512])
            buffer = _upload(
                resident._device, resident._metal, raw, "test native counts"
            )
            try:
                resident.append(
                    SimpleNamespace(
                        _mtl=buffer, ndim=3, shape=raw.shape, dtype=raw.dtype
                    )
                )
                resident.spatial_chunks.append(build_index(resident, buffer, len(raw)))
            finally:
                _release(buffer)
        with pytest.raises(ValueError, match=".qem extension"):
            io.save(tmp_path / "camera.ans", resident, format="quantem", backend="mps")
        assert not (tmp_path / "camera.ans").exists()
        path = tmp_path / "camera.qem"
        io.save(path, resident, format="quantem", backend="mps")
        reopened = io.load(path, backend="mps")
        try:
            assert reopened.metadata["container"] == "quantem.qem"
            axes = reopened.metadata["scientific_metadata"]["axes"]
            assert [axis["size"] for axis in axes] == list(shape)
            for source in (resident, reopened.data):
                decoded = source.decode_scan_range_device(0, 519)
                try:
                    np.testing.assert_array_equal(
                        decoded.to_numpy().reshape(shape), counts
                    )
                finally:
                    decoded.release()
                masks = [
                    np.zeros(shape[2:], bool),
                    np.ones(shape[2:], bool),
                    rng.random(shape[2:]) > 0.4,
                ]
                for scan in (0, 255, 511, 512, 518):
                    frame = source.decode_scan_range_device(scan, scan + 1)
                    try:
                        np.testing.assert_array_equal(
                            frame.to_numpy(), counts[0, scan : scan + 1]
                        )
                    finally:
                        frame.release()
                for mask in masks:
                    result = source.detector_sum_device(mask)
                    try:
                        expected = (counts * (mask & valid)).sum(
                            axis=(2, 3), dtype=np.uint64
                        )
                        np.testing.assert_array_equal(result.to_numpy(), expected)
                    finally:
                        result.release()
                first = masks[-1]
                product = source.detector_delta_device(first)
                try:
                    for mask in (np.roll(first, 1, axis=0), ~first, masks[0], masks[1]):
                        source.detector_delta_device(mask, first, product)
                        expected = (counts * (mask & valid)).sum(
                            axis=(2, 3), dtype=np.uint64
                        )
                        np.testing.assert_array_equal(product.to_numpy(), expected)
                        first = mask
                finally:
                    product.release()
        finally:
            reopened.close()
    finally:
        resident.release()


def test_camera_delta_preserves_uint64_carry_and_borrow():
    """Saturated detector sums remain exact when a drag crosses 32-bit range."""
    pytest.importorskip("Metal")
    from quantem.gpu.io.backends.mps._spatial import build_index
    from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts, _upload
    from quantem.gpu.io.backends.mps.packed import _release

    counts = np.full((3, 257, 257), 65535, np.uint16)
    resident = MPSStreamedCounts((1, *counts.shape), counts.dtype)
    buffer = _upload(
        resident._device, resident._metal, counts, "saturated native counts"
    )
    try:
        resident.append(
            SimpleNamespace(_mtl=buffer, ndim=3, shape=counts.shape, dtype=counts.dtype)
        )
        resident.spatial_chunks = [build_index(resident, buffer, 3)]
        previous = np.zeros((257, 257), bool)
        previous.ravel()[:65536] = True
        output = resident.detector_delta_device(previous)
        try:
            for count in (65538, 65535, 65539, 65534):
                mask = np.zeros_like(previous)
                mask.ravel()[:count] = True
                resident.detector_delta_device(mask, previous, output)
                np.testing.assert_array_equal(
                    output.to_numpy(), np.full((1, 3), count * 65535, np.uint64)
                )
                previous = mask
        finally:
            output.release()
    finally:
        resident.release()
        _release(buffer)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dm4_mps_preserves_counts_and_saved_calibration(tmp_path, dtype):
    """Open a real DM4 structure on MPS and reopen the saved calibrated snapshot."""
    pytest.importorskip("Metal")
    from tests.contracts.io.test_digitalmicrograph import write_dm4

    shape = (23, 25, 24, 32)
    counts = np.random.default_rng(5).poisson(0.3, shape).astype(dtype)
    counts[22, 24, 3, 5] = np.iinfo(dtype).max
    path = write_dm4(tmp_path / "camera.dm4", counts)
    with io.load(path, backend="mps", verbose=False) as loaded:
        saved = tmp_path / "camera.qem"
        io.save(saved, loaded, format="quantem", backend="mps")
    with io.load(saved, backend="mps", verbose=False) as loaded:
        assert loaded.metadata["scan_sampling_A"] == [2.5, 2.5]
        assert loaded.metadata["detector_sampling_inv_A"] == [0.025, 0.025]
        output = loaded.data.decode_scan_range_device(0, 575)
        try:
            np.testing.assert_array_equal(output.to_numpy().reshape(shape), counts)
        finally:
            output.release()


def test_public_detector_session_reads_camera_counts(tmp_path):
    """The public detector session uses exact Metal camera frames and products."""
    pytest.importorskip("Metal")
    from quantem.gpu import detector
    from tests.contracts.io.test_digitalmicrograph import write_dm4

    counts = np.arange(12 * 16 * 16, dtype=np.uint16).reshape(3, 4, 16, 16)
    path = write_dm4(tmp_path / "camera.dm4", counts)
    with io.load(path, backend="mps", verbose=False) as loaded:
        session = detector.prepare(loaded)
        try:
            np.testing.assert_array_equal(session.frame(5), counts[1, 1])
            mask = np.indices((16, 16))[0] < 8
            np.testing.assert_array_equal(
                session.masked_sum_exact(mask),
                (counts * mask).sum(axis=(2, 3), dtype=np.uint64),
            )
        finally:
            session.close()
