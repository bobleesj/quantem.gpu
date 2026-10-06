from pathlib import Path

import numpy as np
import pytest


def test_removed_detector_compute_aliases_stay_absent() -> None:
    from quantem.gpu.detector import tensors as backends
    from quantem.gpu.resident.mps.virtual_image import MetalVirtualImage

    assert not hasattr(backends, "TorchCompute")
    assert not hasattr(backends, "MetalCompute")
    assert not hasattr(MetalVirtualImage, "bin2_chunks")


def test_mps_integer_reduction_kernel_sources_are_present() -> None:
    source = Path("src/quantem/gpu/resident/mps/kernels/reductions.msl").read_text(
        encoding="utf-8"
    )

    for name in (
        "masked_sum_u8",
        "detector_sum_exact_u8",
        "detector_sum_exact_u16",
        "detector_sum_exact_u32",
        "bin_detector_u8",
        "mean_dp_sum_u8",
        "detector_sum_u8_block_partial",
        "detector_sum_u8_block_merge",
        "rowspan_sum_u8",
        "com_u8",
        "masked_sum_u32",
        "mean_dp_sum_u32",
        "rowspan_sum_u32",
        "com_u32",
    ):
        assert f"kernel void {name}" in source


def test_mps_exact_detector_sum_exceeds_uint32_without_overflow() -> None:
    """The Metal exact reducer preserves sums beyond the uint32 range."""
    pytest.importorskip("Metal")
    from quantem.gpu.device import metal_runtime as decoder
    from quantem.gpu.resident.mps.virtual_image import MetalVirtualImage

    frame_count = 70_000
    value = np.iinfo(np.uint16).max
    buffer = decoder.allocate_shared(frame_count * np.dtype(np.uint16).itemsize)
    chunk = decoder.numpy_view(buffer, np.uint16, frame_count).reshape(
        frame_count, 1, 1
    ).view(decoder.SharedArray)
    chunk._mtl = buffer
    chunk.fill(value)

    result = MetalVirtualImage([chunk]).detector_sum_exact()

    assert result.dtype == np.dtype(np.uint64)
    assert int(result[0, 0]) == frame_count * int(value)
    assert int(result[0, 0]) > np.iinfo(np.uint32).max


def test_mps_u8_mean_dp_uses_resident_metal_detector_sum() -> None:
    """Lossless-u8 MPS data must not fall back to a host chunk reduction."""
    from types import SimpleNamespace

    from quantem.gpu.detector.mps.dense import MetalRawBackend
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    detector_sum = np.asarray([[8, 16], [24, 32]], dtype=np.uint64)
    backend = MetalRawBackend.__new__(MetalRawBackend)
    backend.frames = ChunkedFrames.__new__(ChunkedFrames)
    backend.frames.vi = SimpleNamespace(detector_sum_exact=lambda: detector_sum)
    backend.det_shape = (2, 2)
    backend.n_frames = 4

    np.testing.assert_array_equal(
        backend.mean_dp(),
        np.asarray([[2, 4], [6, 8]], dtype=np.float32),
    )


def test_mps_mean_dp_has_no_host_chunk_fallback() -> None:
    """Unsupported MPS dtypes must raise in Metal code, never reduce on CPU."""
    from types import SimpleNamespace

    from quantem.gpu.detector.mps.dense import MetalRawBackend
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    def unsupported_detector_sum():
        raise NotImplementedError("native Metal reducer required")

    backend = MetalRawBackend.__new__(MetalRawBackend)
    backend.frames = ChunkedFrames.__new__(ChunkedFrames)
    backend.frames.vi = SimpleNamespace(detector_sum_exact=unsupported_detector_sum)
    backend.det_shape = (2, 2)
    backend.n_frames = 4

    with pytest.raises(NotImplementedError, match="native Metal reducer"):
        backend.mean_dp()


def test_mps_production_reductions_do_not_route_to_numba() -> None:
    """Row-prefix and detector reductions must remain on the Metal backend."""
    source = Path("src/quantem/gpu/resident/mps/virtual_image.py").read_text(encoding="utf-8")
    backend_source = Path("src/quantem/gpu/detector/mps/dense.py").read_text(
        encoding="utf-8"
    )

    assert "from numba import" not in source
    assert "_masked_sum_prefix_numba" not in source
    assert "gather_columns_float32" in source
    assert "for chunk in self.frames.chunks" not in backend_source
    assert "MPS reduce_frames(reduce='max') has no Metal kernel" in backend_source
    assert "raise NotImplementedError" in backend_source


def test_mps_dense_mask_uses_total_minus_complement_contract() -> None:
    source = Path("src/quantem/gpu/detector/mps/dense.py").read_text(encoding="utf-8")

    assert "self._totals" in source
    assert '"sidecar"' in source
    assert "return self._totals[totals] - np.asarray(vi.masked_sum(~mask))" in source
    assert "bin_mask" in source
