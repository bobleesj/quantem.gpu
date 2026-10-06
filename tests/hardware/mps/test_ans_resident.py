"""Metal tests for corrected ANS residency and bounded public reads."""

from quantem.gpu.io.dataset import Dataset4dstemGPU

import h5py
import hdf5plugin
import numpy as np
import pytest

pytest.importorskip("Metal")
pytest.importorskip("torch")

from quantem.gpu import io
from quantem.gpu.resident.mps.counts import MPSStreamedCounts
from quantem.gpu.resident.mps.precision import upload
from quantem.gpu.device.metal_runtime import shared_array


def _median_corrected(raw, pixel_mask):
    expected = raw.copy()
    height, width = pixel_mask.shape
    for row, column in np.argwhere(pixel_mask != 0):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, column + dc
                if (
                    (dr or dc)
                    and 0 <= rr < height
                    and 0 <= cc < width
                    and pixel_mask[rr, cc] == 0
                ):
                    neighbors.append(raw[..., rr, cc])
        expected[..., row, column] = np.median(
            np.stack(neighbors, axis=-1), axis=-1
        ).astype(raw.dtype)
    return expected


def test_h5_ans_defaults_to_gpu_median_hot_pixel_correction(tmp_path):
    raw = (np.arange(2 * 5 * 6 * 8).reshape(2, 5, 6, 8) * 7 % 251).astype(
        np.uint16
    )
    mask = np.zeros((6, 8), np.uint8)
    mask[0, 0] = 16
    mask[2, 3] = 20
    raw[..., mask != 0] = np.iinfo(np.uint16).max
    expected = _median_corrected(raw, mask)
    path = tmp_path / "hot-pixels.h5"
    with h5py.File(path, "w") as handle:
        data = handle.require_group("entry/data")
        data.create_dataset(
            "data_000001",
            data=raw.reshape(-1, 6, 8),
            chunks=(1, 6, 8),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        detector = handle.require_group("entry/instrument/detector")
        detector_specific = detector.require_group("detectorSpecific")
        detector_specific["ntrigger"] = 10
        detector_specific["y_pixels_in_detector"] = 6
        detector_specific["x_pixels_in_detector"] = 8
        detector_specific["pixel_mask"] = mask

    loaded = io.load(
        path,
        backend="mps",
        scan_shape=(2, 5),
        apply_mask=False,
        verbose=False,
    )
    try:
        decoded = loaded.data.decode_scan_range_device(0, 10)
        try:
            np.testing.assert_array_equal(
                decoded.get().reshape(raw.shape), expected
            )
        finally:
            decoded.release()
        correction = loaded.metadata["hot_pixel_correction"]
        assert correction["method"] == "median"
        assert correction["pixel_count"] == 2
        assert correction["coordinates_row_column"] == [[0, 0], [2, 3]]
        assert correction["applied"] is True
    finally:
        loaded.close()


def test_resident_read_matches_numpy_region():
    """The public read contract restores requested MPS rows and columns."""
    shape = (5, 6, 4, 7)
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) % 251
    valid = np.ones(shape[2:], dtype=bool)
    valid[2, 4] = False
    source = MPSStreamedCounts(shape, np.uint16, valid)
    raw = upload(values.reshape(-1, *shape[2:]))
    try:
        source.append(shared_array(raw._mtl, raw.dtype, raw.shape))
        loaded = Dataset4dstemGPU(
            source,
            {
                "working_shape": shape,
                "working_dtype": "uint16",
                "representation": "encoded",
            },
        )
        observed = loaded.read(
            scan_region=(1, 4, 2, 6),
            detector_region=(1, 4, 3, 7),
        )
        assert observed.device.type == "mps"
        # read() returns the stored counts; detector validity applies to
        # the exact detector reductions, as on CUDA.
        expected = values[1:4, 2:6, 1:4, 3:7]
        np.testing.assert_array_equal(observed.cpu().numpy(), expected)
    finally:
        raw.release()
        source.release()


def test_native_gpu_probability_tables_match_numpy():
    """Swift's count_tables.msl constants and the uploaded Python tables are one model."""
    from pathlib import Path

    from quantem.gpu.device.metal_runtime import (
        allocate_shared,
        buffer_view,
        complete_command,
        metal_module,
        metal_pipelines,
        metal_queue,
        release_buffer,
    )
    from quantem.gpu.formats.qem.reference import count_tables
    from quantem.gpu.resident.mps import counts

    encoding, decoding = count_tables()
    source = MPSStreamedCounts((1, 1, 4, 7), np.uint16)
    try:
        np.testing.assert_array_equal(
            np.frombuffer(buffer_view(source._encoding), np.uint32).reshape(64, 33),
            encoding,
        )
        np.testing.assert_array_equal(
            np.frombuffer(buffer_view(source._decoding), np.uint32).reshape(64, 1024),
            decoding,
        )
    finally:
        source.release()
    kernel = (Path(counts.__file__).parent / "kernels" / "count_tables.msl").read_text()
    pipeline = metal_pipelines(kernel, ("count_tables",), fast_math=False)["count_tables"]
    native = [allocate_shared(encoding.nbytes), allocate_shared(decoding.nbytes)]
    try:
        command = metal_queue().commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate(native):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        size = metal_module().MTLSizeMake
        encoder.dispatchThreads_threadsPerThreadgroup_(size(64, 1, 1), size(128, 1, 1))
        encoder.endEncoding()
        complete_command(command, "count table check")
        np.testing.assert_array_equal(
            np.frombuffer(buffer_view(native[0]), np.uint32).reshape(64, 33), encoding
        )
        np.testing.assert_array_equal(
            np.frombuffer(buffer_view(native[1]), np.uint32).reshape(64, 1024), decoding
        )
    finally:
        for buffer in native:
            release_buffer(buffer)


def test_h5_ans_resident_carries_spatial_index_and_reopens(tmp_path):
    """An encoded H5 load owns its spatial index, so it can be saved and reopened.

    Without the index the resident still decodes, but ``detector_sum_device``
    silently falls back to a full decode and ``io.save`` refuses the snapshot.
    """
    raw = (np.arange(2 * 5 * 6 * 8).reshape(2, 5, 6, 8) * 7 % 251).astype(
        np.uint16
    )
    path = tmp_path / "indexed.h5"
    with h5py.File(path, "w") as handle:
        data = handle.require_group("entry/data")
        data.create_dataset(
            "data_000001",
            data=raw.reshape(-1, 6, 8),
            chunks=(1, 6, 8),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        detector = handle.require_group("entry/instrument/detector")
        detector_specific = detector.require_group("detectorSpecific")
        detector_specific["ntrigger"] = 10
        detector_specific["y_pixels_in_detector"] = 6
        detector_specific["x_pixels_in_detector"] = 8

    loaded = io.load(path, backend="mps", scan_shape=(2, 5), verbose=False)
    saved = tmp_path / "indexed.qem"
    try:
        source = loaded.data
        assert len(source.spatial_chunks) == len(source.chunks)
        assert loaded.metadata["index_bytes"] > 0
        io.save(saved, loaded, format="quantem", backend="mps")
    finally:
        loaded.close()

    mask = np.zeros((6, 8), np.float32)
    mask[1:5, 2:6] = 1.0
    expected = (raw * mask).sum(axis=(2, 3), dtype=np.uint64)
    reopened = io.load(saved, backend="mps", verbose=False)
    try:
        assert len(reopened.data.spatial_chunks) == len(reopened.data.chunks)
        decoded = reopened.data.decode_scan_range_device(0, 10)
        try:
            np.testing.assert_array_equal(decoded.get().reshape(raw.shape), raw)
        finally:
            decoded.release()
        from quantem.gpu import detector

        session = detector.prepare(reopened)
        try:
            np.testing.assert_array_equal(session.masked_sum_exact(mask), expected)
        finally:
            session.close()
    finally:
        reopened.close()
