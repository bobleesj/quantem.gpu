"""MPS Metal buffers must be returned to the system when a load is dropped.

PyObjC does not release buffers created by ``newBufferWithLength_options_`` when
the Python wrapper is collected, so without an explicit release every
``load(backend="mps")`` permanently retains its output. Seven no-bin tilts then
exhaust a 128 GB Mac even though only one tilt is ever meant to be resident.
"""

import gc
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="needs an Apple GPU",
)


def _allocated_bytes() -> int:
    return int(torch.mps.driver_allocated_memory())


def test_owner_releases_buffer_when_dropped():
    """An owned buffer returns its memory once the owner goes away.

    ``newBufferWithLength_options_`` hands back a +1-retained object on top of
    the retain PyObjC takes for its wrapper, so the memory comes back only when
    both are undone: the explicit release, then the dropped Python reference.
    Dropping the reference alone is what leaked ~45 GB per tilt load.
    """
    from quantem.gpu.io.backends.mps import decoder as be

    nbytes = 1 << 30
    baseline = _allocated_bytes()
    owner = be._MtlOwner(be._metal_buffer_alloc(nbytes))
    assert _allocated_bytes() - baseline >= nbytes // 2
    del owner
    gc.collect()
    assert _allocated_bytes() - baseline < nbytes // 2


def test_release_is_idempotent():
    """The owner wrapper detaches its buffer before a second release.

    Raw ``MTLBuffer.release()`` is not idempotent. The wrapper makes its own
    repeated call safe by clearing the one owned reference after the first.
    """
    from quantem.gpu.io.backends.mps import decoder as be

    owner = be._MtlOwner(be._metal_buffer_alloc(1 << 20))
    owner.release()
    owner.release()
    be._release_metal_buffer(None)


def test_mtl_array_view_keeps_buffer_alive():
    """Slices must not outlive the buffer they read from.

    ``_MtlArray`` had no ``__array_finalize__``, so a view silently lost ``_mtl``.
    That was invisible while every buffer leaked; once release works it is a
    use-after-free, so views must carry the owner.
    """
    from quantem.gpu.io.backends.mps import decoder as be

    buf = be._metal_buffer_alloc(4096)
    arr = be._mtl_array_from_buffer(buf, np.uint16, (32, 32))
    view = arr[:8]
    assert view._mtl is arr._mtl, "view dropped its Metal buffer owner"
    assert arr.reshape(16, 64)._mtl is arr._mtl


def test_compressed_buffer_bound_uses_file_metadata(tmp_path):
    """Sizing compressed input must not pre-scan every HDF5 chunk layout."""
    from quantem.gpu.io.backends.mps import decoder as be

    small = tmp_path / "small.h5"
    large = tmp_path / "large.h5"
    small.write_bytes(b"0")
    large.write_bytes(b"0" * 1024)
    plan = SimpleNamespace(chunk_files=(str(small), str(large)))

    assert be._max_compressed_bytes_for_plan(plan) == 150 * 1024 * 1024

    os.truncate(large, 200 * 1024 * 1024)
    assert be._max_compressed_bytes_for_plan(plan) == 201 * 1024 * 1024


def test_exact_fused_loaders_are_default_on_and_can_be_disabled(monkeypatch):
    """The verified exact paths are defaults, with an explicit diagnostic opt-out."""
    from quantem.gpu.io.backends.mps import decoder as be

    monkeypatch.delenv("QT_MPS_FUSED_FULL_U16", raising=False)
    monkeypatch.delenv("QT_MPS_FUSED_BIN", raising=False)
    assert be._fused_full_u16_enabled()
    assert be._fused_bin_enabled()

    monkeypatch.setenv("QT_MPS_FUSED_FULL_U16", "0")
    monkeypatch.setenv("QT_MPS_FUSED_BIN", "0")
    assert not be._fused_full_u16_enabled()
    assert not be._fused_bin_enabled()


def test_exact_detector_bins_route_to_distinct_fused_pipelines():
    """Measured bin 4 and candidate bin 8 keep distinct specializations."""
    from quantem.gpu.io.backends.mps import decoder as be

    pipelines = [be._fused_bin_pipeline_for(factor) for factor in (2, 4, 8)]

    assert all(pipeline is not None for pipeline in pipelines)
    assert len({id(pipeline) for pipeline in pipelines}) == 3
    with pytest.raises(KeyError):
        be._fused_bin_pipeline_for(3)


def test_exact_detector_bins_use_measured_output_tile_shapes():
    """Each exact detector-bin pipeline declares its measured output tile."""
    from quantem.gpu.io.backends.mps import decoder as be

    assert be._fused_bin_output_tile_shape(2) == (16, 32)
    assert be._fused_bin_output_tile_shape(4) == (8, 16)
    assert be._fused_bin_output_tile_shape(8) == (8, 8)


def test_binned_uint16_overflow_releases_rejected_output(monkeypatch):
    """An unrepresentable exact sum releases memory before failing closed."""
    from quantem.gpu.io.backends.mps import decoder as be

    rejected = object()
    released = []
    monkeypatch.setattr(be, "_release_metal_buffer", released.append)
    monkeypatch.setattr(be, "_buffer_key", id)

    be._release_and_raise_if_binned_integer_overflow(
        np.array([0], dtype=np.uint32),
        4,
        np.dtype(np.uint16),
        rejected,
    )
    assert released == []

    with pytest.raises(OverflowError, match="no saturated uint16 result"):
        be._release_and_raise_if_binned_integer_overflow(
            np.array([1], dtype=np.uint32),
            4,
            np.dtype(np.uint16),
            rejected,
            rejected,
        )
    assert released == [rejected]


def _dispatch_scalar_integer_bin(values: np.ndarray, bin_factor: int):
    """Run the fallback Metal sum kernel and return output plus range flag."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = np.asarray(values)
    assert values.dtype in (np.dtype(np.uint16), np.dtype(np.uint32))
    in_rows, in_cols = values.shape
    out_rows, out_cols = in_rows // bin_factor, in_cols // bin_factor
    out_size = out_rows * out_cols
    buffers = [
        be._metal_buffer_alloc(values.nbytes),
        be._metal_buffer_alloc(out_size * values.dtype.itemsize),
        be._metal_buffer_alloc(values.size),
        be._metal_buffer_alloc(np.dtype(np.uint32).itemsize),
    ]
    in_mtl, out_mtl, mask_mtl, overflow_mtl = buffers
    try:
        be._numpy_view(in_mtl, values.dtype, values.size)[:] = values.ravel()
        be._numpy_view(mask_mtl, np.uint8, values.size)[:] = 0
        overflow = be._numpy_view(overflow_mtl, np.uint32, 1)
        overflow[0] = 0

        command = be._queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        decoder = SimpleNamespace(
            _mask_mtl=mask_mtl,
            _cast_overflow_mtl=overflow_mtl,
        )
        be.MPSDecompressor._encode_detector_bin_sum(
            decoder,
            encoder,
            in_mtl=in_mtl,
            in_byte_offset=0,
            out_mtl=out_mtl,
            out_byte_offset=0,
            n_frames=1,
            elem_size=values.dtype.itemsize,
            det_row=in_rows,
            det_col=in_cols,
            bin_factor=bin_factor,
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

        output = np.array(
            be._numpy_view(out_mtl, values.dtype, out_size),
            copy=True,
        ).reshape(out_rows, out_cols)
        return output, int(overflow[0])
    finally:
        for buffer in buffers:
            be._release_metal_buffer(buffer)


def test_scalar_bin_kernel_checks_range_and_rectangular_rows():
    """The fallback kernel reports overflow and keeps independent row bounds."""
    safe = np.arange(1, 9, dtype=np.uint16).reshape(4, 2)
    output, overflow = _dispatch_scalar_integer_bin(safe, 2)
    np.testing.assert_array_equal(output, np.array([[10], [26]], dtype=np.uint16))
    assert overflow == 0

    too_large = np.full((4, 4), 20_000, dtype=np.uint16)
    output, overflow = _dispatch_scalar_integer_bin(too_large, 2)
    np.testing.assert_array_equal(output, np.full((2, 2), 65_535, dtype=np.uint16))
    assert overflow == 1


def test_scalar_uint32_bin_uses_widened_accumulator_and_fails_closed():
    """Native uint32 sums may not wrap before their output range check."""
    safe = np.arange(1, 9, dtype=np.uint32).reshape(4, 2)
    output, overflow = _dispatch_scalar_integer_bin(safe, 2)
    np.testing.assert_array_equal(output, np.array([[10], [26]], dtype=np.uint32))
    assert overflow == 0

    too_large = np.full((2, 2), np.iinfo(np.uint32).max, dtype=np.uint32)
    output, overflow = _dispatch_scalar_integer_bin(too_large, 2)
    np.testing.assert_array_equal(
        output,
        np.array([[np.iinfo(np.uint32).max]], dtype=np.uint32),
    )
    assert overflow == 1


def _write_bslz4_master(
    root,
    name: str,
    values: np.ndarray,
    *,
    pixel_mask: np.ndarray | None = None,
):
    """Write one real bitshuffle-LZ4 detector frame and Arina-style master."""
    import h5py
    import hdf5plugin

    data_path = root / f"{name}_data_000001.h5"
    master_path = root / f"{name}_master.h5"
    with h5py.File(data_path, "w") as data_file:
        data_file.create_dataset(
            "entry/data/data",
            data=values,
            chunks=(1, values.shape[-2], values.shape[-1]),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    with h5py.File(master_path, "w") as master_file:
        master_file["entry/data/data_000001"] = h5py.ExternalLink(
            str(data_path),
            "/entry/data/data",
        )
        master_file.create_dataset(
            "entry/instrument/detector/detectorSpecific/ntrigger",
            data=np.uint32(values.shape[0]),
        )
        if pixel_mask is not None:
            master_file.create_dataset(
                "entry/instrument/detector/detectorSpecific/pixel_mask",
                data=np.asarray(pixel_mask, dtype=np.uint8),
            )
    return master_path


def _write_bslz4_sharded_master(
    root,
    name: str,
    values: np.ndarray,
    split: int,
):
    """Write two external bitshuffle-LZ4 shards for output-offset coverage."""
    import h5py
    import hdf5plugin

    master_path = root / f"{name}_master.h5"
    data_paths = []
    for index, shard in enumerate((values[:split], values[split:]), start=1):
        data_path = root / f"{name}_data_{index:06d}.h5"
        with h5py.File(data_path, "w") as data_file:
            data_file.create_dataset(
                "entry/data/data",
                data=shard,
                chunks=(1, values.shape[-2], values.shape[-1]),
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
        data_paths.append(data_path)
    with h5py.File(master_path, "w") as master_file:
        for index, data_path in enumerate(data_paths, start=1):
            master_file[f"entry/data/data_{index:06d}"] = h5py.ExternalLink(
                str(data_path),
                "/entry/data/data",
            )
        master_file.create_dataset(
            "entry/instrument/detector/detectorSpecific/ntrigger",
            data=np.uint32(values.shape[0]),
        )
    return master_path


def _release_test_arrays(be, *groups) -> None:
    """Release each returned Metal buffer exactly once."""
    released = set()
    for group in groups:
        arrays = group if isinstance(group, (list, tuple)) else [group]
        for array in arrays:
            buffer = getattr(array, "_mtl", None)
            if buffer is None:
                continue
            key = be._buffer_key(buffer)
            array._mtl = None
            if key not in released:
                released.add(key)
                be._release_metal_buffer(buffer)


@pytest.mark.parametrize("bin_factor", [2, 4, 8])
def test_fused_bslz4_bins_preserve_exact_counts_and_declared_u8(
    tmp_path,
    bin_factor,
):
    """Production fused kernels prove safe, overflow, and explicit-u8 paths."""
    from quantem.gpu.io.backends.mps import decoder as be

    safe_values = (np.arange(4096, dtype=np.uint16) % 100).reshape(1, 64, 64)
    high_values = np.full((1, 64, 64), 20_000, dtype=np.uint16)
    safe_master = _write_bslz4_master(tmp_path, "safe", safe_values)
    high_master = _write_bslz4_master(tmp_path, "high", high_values)
    try:
        safe = be.load_master(
            str(safe_master),
            det_bin=bin_factor,
            verbose=False,
        )
        expected = safe_values.reshape(
            1,
            64 // bin_factor,
            bin_factor,
            64 // bin_factor,
            bin_factor,
        ).sum(axis=(2, 4), dtype=np.uint64)
        np.testing.assert_array_equal(safe, expected.astype(np.uint16))
        _release_test_arrays(be, safe)

        with pytest.raises(OverflowError, match="no saturated uint16 result"):
            be.load_master(
                str(high_master),
                det_bin=bin_factor,
                verbose=False,
            )

        clipped = be.load_master(
            str(high_master),
            det_bin=bin_factor,
            output_dtype="u8",
            verbose=False,
        )
        np.testing.assert_array_equal(
            clipped,
            np.full(clipped.shape, 255, dtype=np.uint8),
        )
        _release_test_arrays(be, clipped)
    finally:
        be.clear_mps_cache()


@pytest.mark.parametrize("detector_shape", [(96, 128), (128, 96)])
def test_fused_bin8_rectangular_partial_tiles_preserve_masked_counts(
    tmp_path,
    detector_shape,
):
    """Row- and column-partial bin-8 tiles preserve mask-before-sum exactness."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (
        np.arange(np.prod(detector_shape), dtype=np.uint16) % 100
    ).reshape(1, *detector_shape)
    values[:, :8, :8] = 20_000
    pixel_mask = np.zeros(detector_shape, dtype=bool)
    pixel_mask[:8, :8] = True
    master = _write_bslz4_master(
        tmp_path,
        f"rectangular_{detector_shape[0]}_{detector_shape[1]}",
        values,
    )
    expected_source = values.copy()
    expected_source[:, pixel_mask] = 0
    expected = expected_source.reshape(
        1,
        detector_shape[0] // 8,
        8,
        detector_shape[1] // 8,
        8,
    ).sum(axis=(2, 4), dtype=np.uint64)
    try:
        result = be.load_master(
            str(master),
            det_bin=8,
            pixel_mask=pixel_mask,
            verbose=False,
        )
        np.testing.assert_array_equal(result, expected.astype(np.uint16))
        _release_test_arrays(be, result)

        with pytest.raises(OverflowError, match="no saturated uint16 result"):
            be.load_master(str(master), det_bin=8, verbose=False)
    finally:
        be.clear_mps_cache()


@pytest.mark.parametrize("detector_shape", [(48, 256), (128, 96)])
def test_fused_bin2_rectangular_partial_tiles_preserve_masked_counts(
    tmp_path,
    detector_shape,
):
    """Bin-2 row and column edge tiles preserve mask-before-sum exactness."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (
        np.arange(np.prod(detector_shape), dtype=np.uint16) % 100
    ).reshape(1, *detector_shape)
    values[:, -2:, -2:] = 20_000
    pixel_mask = np.zeros(detector_shape, dtype=bool)
    pixel_mask[-2:, -2:] = True
    master = _write_bslz4_master(
        tmp_path,
        f"bin2_rectangular_{detector_shape[0]}_{detector_shape[1]}",
        values,
    )
    expected_source = values.copy()
    expected_source[:, pixel_mask] = 0
    expected = expected_source.reshape(
        1,
        detector_shape[0] // 2,
        2,
        detector_shape[1] // 2,
        2,
    ).sum(axis=(2, 4), dtype=np.uint64)
    try:
        result = be.load_master(
            str(master),
            det_bin=2,
            pixel_mask=pixel_mask,
            verbose=False,
        )
        np.testing.assert_array_equal(result, expected.astype(np.uint16))
        _release_test_arrays(be, result)

        with pytest.raises(OverflowError, match="no saturated uint16 result"):
            be.load_master(str(master), det_bin=2, verbose=False)
    finally:
        be.clear_mps_cache()


@pytest.mark.parametrize("detector_shape", [(48, 96), (96, 48), (32, 64)])
def test_partial_bitshuffle_tail_preserves_native_mask_and_explicit_u8(
    tmp_path,
    detector_shape,
):
    """The canonical tail stride preserves native, masked, and u8 outputs."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (
        np.arange(2 * np.prod(detector_shape), dtype=np.uint16) % 600
    ).reshape(2, *detector_shape)
    values[:, -2:, -2:] = 20_000
    pixel_mask = np.zeros(detector_shape, dtype=bool)
    pixel_mask[-2:, -2:] = True
    master = _write_bslz4_master(
        tmp_path,
        f"tail_native_{detector_shape[0]}_{detector_shape[1]}",
        values,
    )
    expected_masked = values.copy()
    expected_masked[:, pixel_mask] = 0
    try:
        native = be.load_master(str(master), verbose=False)
        np.testing.assert_array_equal(native, values)
        _release_test_arrays(be, native)

        masked = be.load_master(
            str(master),
            pixel_mask=pixel_mask,
            verbose=False,
        )
        np.testing.assert_array_equal(masked, expected_masked)
        _release_test_arrays(be, masked)

        clipped = be.load_master(
            str(master),
            pixel_mask=pixel_mask,
            output_dtype="u8",
            verbose=False,
        )
        np.testing.assert_array_equal(
            clipped,
            np.minimum(expected_masked, 255).astype(np.uint8),
        )
        _release_test_arrays(be, clipped)
    finally:
        be.clear_mps_cache()


@pytest.mark.parametrize("detector_shape", [(48, 96), (96, 48)])
@pytest.mark.parametrize("bin_factor", [2, 4, 8])
def test_partial_bitshuffle_tail_preserves_exact_detector_bins(
    tmp_path,
    detector_shape,
    bin_factor,
):
    """Tail unshuffle feeds exact mask-before-sum detector binning."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (
        np.arange(np.prod(detector_shape), dtype=np.uint16) % 100
    ).reshape(1, *detector_shape)
    values[:, -2:, -2:] = 20_000
    pixel_mask = np.zeros(detector_shape, dtype=bool)
    pixel_mask[-2:, -2:] = True
    master = _write_bslz4_master(
        tmp_path,
        f"tail_bin{bin_factor}_{detector_shape[0]}_{detector_shape[1]}",
        values,
    )
    expected_source = values.copy()
    expected_source[:, pixel_mask] = 0
    expected = expected_source.reshape(
        1,
        detector_shape[0] // bin_factor,
        bin_factor,
        detector_shape[1] // bin_factor,
        bin_factor,
    ).sum(axis=(2, 4), dtype=np.uint64)
    try:
        result = be.load_master(
            str(master),
            det_bin=bin_factor,
            pixel_mask=pixel_mask,
            verbose=False,
        )
        np.testing.assert_array_equal(result, expected.astype(np.uint16))
        _release_test_arrays(be, result)

        with pytest.raises(OverflowError, match="no saturated uint16 result"):
            be.load_master(
                str(master),
                det_bin=bin_factor,
                verbose=False,
            )
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_uint32(tmp_path):
    """The uint32 tail port matches the canonical CUDA element mapping."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (
        np.arange(2 * 48 * 48, dtype=np.uint32).reshape(2, 48, 48) * 1009
    )
    master = _write_bslz4_master(tmp_path, "tail_uint32", values)
    try:
        result = be.load_master(str(master), verbose=False)
        np.testing.assert_array_equal(result, values)
        _release_test_arrays(be, result)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_fused_u8_detector_sum(tmp_path):
    """Lossless-u8 tail decode and its detector sum consume every count."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(3 * 48 * 96, dtype=np.uint16) % 200).reshape(
        3,
        48,
        96,
    )
    values[:, -2:, -2:] = np.iinfo(np.uint16).max
    pixel_mask = np.zeros((48, 96), dtype=bool)
    pixel_mask[-2:, -2:] = True
    expected = values.copy()
    expected[:, pixel_mask] = 0
    master = _write_bslz4_master(tmp_path, "tail_u8_detector_sum", values)
    try:
        chunks, detector_sum = be.load_master_chunked(
            str(master),
            pixel_mask=pixel_mask,
            output_dtype="u8",
            precompute_detector_sum=True,
            verbose=False,
        )
        actual = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=0)
        np.testing.assert_array_equal(actual, expected.astype(np.uint8))
        np.testing.assert_array_equal(
            detector_sum,
            expected.sum(axis=0, dtype=np.uint64).astype(np.uint32),
        )
        _release_test_arrays(be, chunks)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_exact_fast_sidecar(tmp_path):
    """Native chunks and their eager exact-bin sidecar include the tail."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(3 * 48 * 96, dtype=np.uint16) % 100).reshape(
        3,
        48,
        96,
    )
    values[:, -2:, -2:] = 20_000
    pixel_mask = np.zeros((48, 96), dtype=bool)
    pixel_mask[-2:, -2:] = True
    expected = values.copy()
    expected[:, pixel_mask] = 0
    expected_sidecar = expected.reshape(3, 24, 2, 48, 2).sum(
        axis=(2, 4),
        dtype=np.uint64,
    )
    master = _write_bslz4_master(tmp_path, "tail_fast_sidecar", values)
    try:
        chunks, fast_chunks = be.load_master_chunked(
            str(master),
            pixel_mask=pixel_mask,
            fast_det_bin=2,
            verbose=False,
        )
        actual = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=0)
        actual_sidecar = np.concatenate(
            [np.asarray(chunk) for chunk in fast_chunks],
            axis=0,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual_sidecar,
            expected_sidecar.astype(np.uint16),
        )
        _release_test_arrays(be, chunks, fast_chunks)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_selective_order_and_duplicates(
    tmp_path,
):
    """Selective MPS IO reads only requested tail frames in requested order."""
    from quantem.gpu.io import load
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(4 * 48 * 96, dtype=np.uint16) % 1000).reshape(
        4,
        48,
        96,
    )
    master = _write_bslz4_master(tmp_path, "tail_selective", values)
    try:
        result = load(
            str(master),
            scan_indices=[3, 1, 3, 0],
            scan_shape=(2, 2),
            backend="mps",
            verbose=False,
        )
        np.testing.assert_array_equal(result.data, values[[3, 1, 3, 0]])
        np.testing.assert_array_equal(
            result.metadata["scan_indices"],
            [3, 1, 3, 0],
        )
        _release_test_arrays(be, result.data)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_after_multiple_full_blocks_and_direct_load(
    tmp_path,
):
    """The tail remains exact after two full blocks in the direct decoder."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(2 * 96 * 96, dtype=np.uint16) % 1000).reshape(
        2,
        96,
        96,
    )
    master = _write_bslz4_master(tmp_path, "tail_direct", values)
    data_path = tmp_path / "tail_direct_data_000001.h5"
    decoder = be.MPSDecompressor(
        max_compressed_bytes=max(1 << 20, data_path.stat().st_size),
        max_frames=values.shape[0],
        frame_bytes=values[0].nbytes,
        n_blocks_per_frame=3,
        gpu_batch=values.shape[0],
    )
    try:
        direct = decoder.load(str(data_path), verbose=False)
        np.testing.assert_array_equal(direct, values)

        public = be.load_master(str(master), verbose=False)
        np.testing.assert_array_equal(public, values)
        _release_test_arrays(be, public)
    finally:
        decoder.free()
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_row_prefix(tmp_path):
    """Mask-before-prefix remains exact for a multi-block tail frame."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(2 * 96 * 96, dtype=np.uint16) % 5).reshape(
        2,
        96,
        96,
    )
    pixel_mask = np.zeros((96, 96), dtype=bool)
    pixel_mask[:, -1] = True
    expected = values.copy()
    expected[:, pixel_mask] = 0
    expected = np.cumsum(expected, axis=2, dtype=np.uint32).astype(np.uint16)
    master = _write_bslz4_master(tmp_path, "tail_prefix", values)
    try:
        chunks = be.load_master_chunked(
            str(master),
            pixel_mask=pixel_mask,
            row_prefix=True,
            verbose=False,
        )
        actual = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=0)
        np.testing.assert_array_equal(actual, expected)
        assert all(chunk._row_prefix for chunk in chunks)
        _release_test_arrays(be, chunks)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_sharded_output_offsets(tmp_path):
    """Two source shards write exact binned results at distinct offsets."""
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(5 * 96 * 96, dtype=np.uint16) % 100).reshape(
        5,
        96,
        96,
    )
    master = _write_bslz4_sharded_master(
        tmp_path,
        "tail_sharded",
        values,
        split=2,
    )
    expected = values.reshape(5, 48, 2, 48, 2).sum(
        axis=(2, 4),
        dtype=np.uint64,
    )
    try:
        result = be.load_master(str(master), det_bin=2, verbose=False)
        np.testing.assert_array_equal(result, expected.astype(np.uint16))
        _release_test_arrays(be, result)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_prepared_bin_and_masked_u8(
    tmp_path,
):
    """Selective exact binning and explicit clipped output use the real mask."""
    from quantem.gpu.io import load
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(4 * 96 * 96, dtype=np.uint16) % 600).reshape(
        4,
        96,
        96,
    )
    values[:, -2:, -2:] = np.iinfo(np.uint16).max
    pixel_mask = np.zeros((96, 96), dtype=bool)
    pixel_mask[-2:, -2:] = True
    master = _write_bslz4_master(
        tmp_path,
        "tail_prepared",
        values,
        pixel_mask=pixel_mask,
    )
    order = [3, 1, 3, 0]
    expected = values[order].copy()
    expected[:, pixel_mask] = 0
    expected_bin = expected.reshape(4, 48, 2, 48, 2).sum(
        axis=(2, 4),
        dtype=np.uint64,
    )
    try:
        binned = load(
            str(master),
            scan_indices=order,
            scan_shape=(2, 2),
            det_bin=2,
            backend="mps",
            verbose=False,
        )
        np.testing.assert_array_equal(
            binned.data,
            expected_bin.astype(np.uint16),
        )

        clipped = load(
            str(master),
            scan_indices=order,
            scan_shape=(2, 2),
            dtype="u8",
            backend="mps",
            verbose=False,
        )
        np.testing.assert_array_equal(
            clipped.data,
            np.minimum(expected, 255).astype(np.uint8),
        )
        _release_test_arrays(be, binned.data, clipped.data)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_preserves_prepared_uint32_narrow(tmp_path):
    """Selective uint32-to-uint16 narrowing stays exact after mask application."""
    from quantem.gpu.io import load
    from quantem.gpu.io.backends.mps import decoder as be

    values = (np.arange(4 * 48 * 48, dtype=np.uint32) * 7).reshape(
        4,
        48,
        48,
    )
    pixel_mask = np.zeros((48, 48), dtype=bool)
    pixel_mask[-1, -1] = True
    values[:, -1, -1] = np.iinfo(np.uint32).max
    master = _write_bslz4_master(
        tmp_path,
        "tail_prepared_u32",
        values,
        pixel_mask=pixel_mask,
    )
    expected = values[[2, 0]].copy()
    expected[:, pixel_mask] = 0
    try:
        result = load(
            str(master),
            scan_indices=[2, 0],
            scan_shape=(2, 2),
            dtype="u16",
            backend="mps",
            verbose=False,
        )
        np.testing.assert_array_equal(result.data, expected.astype(np.uint16))
        _release_test_arrays(be, result.data)
    finally:
        be.clear_mps_cache()


def test_partial_bitshuffle_tail_rejects_non_byte_aligned_elements():
    """Unsupported tail geometry fails closed with a corrective next step."""
    from quantem.gpu.io.backends.mps import decoder as be

    with pytest.raises(ValueError, match="multiple of 8 elements"):
        be._bitshuffle_tail_elements(33 * 33 * 2, 2)


def test_partial_bitshuffle_tail_rejects_unsupported_source_before_allocation(
    monkeypatch,
):
    """Public prepared IO validates source width and tail before construction."""
    from quantem.gpu.io.backends.mps import decoder as be

    def reject_construction(*args, **kwargs):
        pytest.fail("invalid source reached Metal allocation")

    monkeypatch.setattr(be, "MPSDecompressor", reject_construction)
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        be.load_prepared_frames(
            {
                "frame_bytes": 33 * 33 * 2,
                "dtype": np.dtype(np.uint16),
            }
        )
    with pytest.raises(ValueError, match="supports only 2-byte uint16"):
        be.load_prepared_frames(
            {
                "frame_bytes": 32 * 32,
                "dtype": np.dtype(np.uint8),
            }
        )


def test_partial_bitshuffle_tail_rejects_class_load_before_output_allocation(
    monkeypatch,
):
    """The eager class entry point validates before its full output buffer."""
    from quantem.gpu.io.backends.mps import decoder as be

    plan = SimpleNamespace(
        detector_shape=(33, 33),
        dtype=np.dtype(np.uint16),
        frame_bytes=33 * 33 * 2,
        elem_size=2,
    )
    monkeypatch.setattr(be, "plan_master", lambda master_path: plan)
    decoder = SimpleNamespace(
        _ensure_output_buffer=lambda nbytes: pytest.fail(
            "invalid tail reached full-output allocation"
        )
    )

    with pytest.raises(ValueError, match="multiple of 8 elements"):
        be.MPSDecompressor.load_master(decoder, "invalid_master.h5")


def test_fast_sidecar_overflow_fails_closed_then_resets(tmp_path):
    """The eager sidecar rejects overflow and recovers on the next exact load."""
    from quantem.gpu.io.backends.mps import decoder as be

    safe_values = (np.arange(4096, dtype=np.uint16) % 100).reshape(1, 64, 64)
    high_values = np.full((1, 64, 64), 20_000, dtype=np.uint16)
    safe_master = _write_bslz4_master(tmp_path, "sidecar_safe", safe_values)
    high_master = _write_bslz4_master(tmp_path, "sidecar_high", high_values)
    expected = safe_values.reshape(1, 32, 2, 32, 2).sum(
        axis=(2, 4),
        dtype=np.uint64,
    )
    try:
        raw, sidecar = be.load_master_chunked(
            str(safe_master),
            fast_det_bin=2,
            verbose=False,
        )
        np.testing.assert_array_equal(raw[0], safe_values)
        np.testing.assert_array_equal(sidecar[0], expected.astype(np.uint16))
        _release_test_arrays(be, raw, sidecar)

        with pytest.raises(OverflowError, match="no saturated uint16 result"):
            be.load_master_chunked(
                str(high_master),
                fast_det_bin=2,
                verbose=False,
            )

        raw, sidecar = be.load_master_chunked(
            str(safe_master),
            fast_det_bin=2,
            verbose=False,
        )
        np.testing.assert_array_equal(raw[0], safe_values)
        np.testing.assert_array_equal(sidecar[0], expected.astype(np.uint16))
        _release_test_arrays(be, raw, sidecar)
    finally:
        be.clear_mps_cache()


def test_bslz4_uint32_bin_uses_widened_accumulator(tmp_path):
    """The public uint32 fallback preserves low sums and rejects wraparound."""
    from quantem.gpu.io.backends.mps import decoder as be

    safe_values = np.arange(4096, dtype=np.uint32).reshape(1, 64, 64)
    high_values = np.full(
        (1, 64, 64),
        np.iinfo(np.uint32).max,
        dtype=np.uint32,
    )
    safe_master = _write_bslz4_master(tmp_path, "u32_safe", safe_values)
    high_master = _write_bslz4_master(tmp_path, "u32_high", high_values)
    expected = safe_values.reshape(1, 32, 2, 32, 2).sum(
        axis=(2, 4),
        dtype=np.uint64,
    )
    try:
        safe = be.load_master(str(safe_master), det_bin=2, verbose=False)
        np.testing.assert_array_equal(safe, expected.astype(np.uint32))
        _release_test_arrays(be, safe)

        with pytest.raises(OverflowError, match="no saturated uint32 result"):
            be.load_master(str(high_master), det_bin=2, verbose=False)

        clipped = be.load_master(
            str(high_master),
            det_bin=2,
            output_dtype="u8",
            verbose=False,
        )
        np.testing.assert_array_equal(
            clipped,
            np.full(clipped.shape, 255, dtype=np.uint8),
        )
        _release_test_arrays(be, clipped)
    finally:
        be.clear_mps_cache()


def _sparse_decoder_for_overflow_test(be, *, overflow_after_dispatch: int):
    """Construct the smallest selective-load decoder without source IO."""
    decoder = SimpleNamespace()
    decoder.max_frames = 1
    decoder._comp_np = np.zeros(1, dtype=np.uint8)
    decoder._co_np = np.zeros(1, dtype=np.uint32)
    decoder._bs_np = np.zeros(1, dtype=np.uint32)
    decoder._bc_np = np.zeros(1, dtype=np.uint32)
    decoder._bo_np = np.zeros(2, dtype=np.uint32)
    decoder._cast_overflow_np = np.array([99], dtype=np.uint32)
    for name in ("_comp_mtl", "_co_mtl", "_bs_mtl", "_bc_mtl", "_bo_mtl"):
        setattr(decoder, name, object())
    decoder._set_mask = lambda *args: None

    def submit(*args, **kwargs):
        assert decoder._cast_overflow_np[0] == 0, "stale range flag was not reset"
        decoder._cast_overflow_np[0] = overflow_after_dispatch
        return SimpleNamespace(waitUntilCompleted=lambda: None)

    decoder._submit_gpu_binned = submit
    return decoder


def _minimal_sparse_prepared():
    return {
        "read_buffer": bytearray(1),
        "total_frames": 1,
        "frame_shape": (2, 4),
        "dtype": np.dtype(np.uint16),
        "frame_bytes": 16,
        "chunk_offsets": np.array([0], dtype=np.uint64),
        "block_starts": np.array([0], dtype=np.uint32),
        "block_counts": np.array([1], dtype=np.uint32),
        "block_offsets": np.array([0, 1], dtype=np.uint32),
    }


def test_sparse_binned_uint16_resets_checks_and_releases(monkeypatch):
    """Selective exact loads use the same fail-closed ownership contract."""
    from quantem.gpu.io.backends.mps import decoder as be

    released = []
    real_release = be._release_metal_buffer

    def release(buffer):
        released.append(be._buffer_key(buffer))
        real_release(buffer)

    monkeypatch.setattr(be, "_release_metal_buffer", release)
    decoder = _sparse_decoder_for_overflow_test(be, overflow_after_dispatch=1)
    with pytest.raises(OverflowError, match="no saturated uint16 result"):
        be.MPSDecompressor.load_prepared_frames(
            decoder,
            _minimal_sparse_prepared(),
            det_bin=2,
        )
    assert len(released) == 1


def test_sparse_explicit_uint8_keeps_declared_clipping(monkeypatch):
    """A declared uint8 browse load may clip after an exact sum exceeds uint16."""
    from quantem.gpu.io.backends.mps import decoder as be

    decoder = _sparse_decoder_for_overflow_test(be, overflow_after_dispatch=1)
    sentinel = np.array([255], dtype=np.uint8)
    monkeypatch.setattr(be, "_cast_mtl_integer_to_u8", lambda output: sentinel)

    output = be.MPSDecompressor.load_prepared_frames(
        decoder,
        _minimal_sparse_prepared(),
        det_bin=2,
        output_dtype="u8",
    )
    assert output is sentinel


def test_full_explicit_uint8_propagates_clipping_intent(monkeypatch):
    """The full loader declares uint8 intent before the fused sum dispatch."""
    from quantem.gpu.io.backends.mps import decoder as be

    plan = SimpleNamespace(
        detector_shape=(192, 192),
        dtype=np.dtype(np.uint16),
        chunk_n_frames=(1,),
    )
    source = SimpleNamespace(_mtl=object())
    calls = []

    class FakeDecompressor:
        def load_binned_masked(self, *args, **kwargs):
            calls.append(kwargs)
            return source

    monkeypatch.setattr(be, "plan_master", lambda path: plan)
    monkeypatch.setattr(be, "_get_decompressor", lambda *args, **kwargs: FakeDecompressor())
    monkeypatch.setattr(be, "_cast_mtl_integer_to_u8", lambda output: "uint8-result")
    released = []
    monkeypatch.setattr(be, "_release_metal_buffer", released.append)

    result = be.load_master("fixture.h5", det_bin=4, output_dtype="u8", verbose=False)

    assert result == "uint8-result"
    assert calls == [
        {
            "mask": None,
            "verbose": False,
            "allow_integer_saturation_for_u8": True,
        }
    ]
    assert source._mtl is None
    assert len(released) == 1


def test_whole_shard_batch_is_bounded_to_one_gigabyte(monkeypatch):
    """Small fused shards stay intact; unusually large shards remain bounded."""
    from quantem.gpu.io.backends.mps import decoder as be

    created = []

    class FakeDecompressor:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.gpu_batch = kwargs["gpu_batch"]

    monkeypatch.setattr(be, "MPSDecompressor", FakeDecompressor)
    be._decompressor_cache.clear()
    frame_bytes = 192 * 192 * 2

    small = be._get_decompressor(
        frame_bytes,
        max_frames=10_000,
        whole_shard=True,
    )
    large = be._get_decompressor(
        frame_bytes,
        max_frames=100_000,
        whole_shard=True,
    )

    assert small.gpu_batch == 10_000
    assert large.gpu_batch == int(be._GPU_BATCH_TARGET_GB * 1e9 / frame_bytes)
    assert len(created) == 2


MAPED_TEST_DIR = os.environ.get("MAPED_TEST_DIR", "")


@pytest.mark.skipif(not os.path.isdir(MAPED_TEST_DIR), reason="needs MAPED_TEST_DIR")
def test_repeated_load_does_not_accumulate():
    """Loading tilts one at a time must not grow memory without bound."""
    import glob

    from quantem.gpu.io import load

    masters = sorted(glob.glob(os.path.join(MAPED_TEST_DIR, "*_master.h5")))[:3]
    if len(masters) < 2:
        pytest.skip("needs at least 2 masters")

    result = load(masters[0], det_bin=1, backend="mps", verbose=False)
    one_tilt = _allocated_bytes()
    del result
    gc.collect()
    for path in masters[1:]:
        result = load(path, det_bin=1, backend="mps", verbose=False)
        del result
        gc.collect()
    assert _allocated_bytes() <= one_tilt * 1.5, (
        f"memory grew from {one_tilt / 1e9:.1f} GB to "
        f"{_allocated_bytes() / 1e9:.1f} GB across {len(masters)} sequential loads"
    )
