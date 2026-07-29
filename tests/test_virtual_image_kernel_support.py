from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_cuda_virtual_image_support_includes_future_1024_uint8_shape() -> None:
    from quantem.gpu.compute import virtual_image_kernel_support

    support = virtual_image_kernel_support(
        backend="cuda",
        shape=(1024, 1024, 192, 192),
        dtype=np.uint8,
        bf_radius=30,
    )

    assert support.backend == "cuda"
    assert support.available is True
    assert support.custom_kernel is True
    assert support.scan_shape == (1024, 1024)
    assert support.det_shape == (192, 192)
    assert support.resident_gib == 36.0
    assert support.mask_paths == {
        "BF": "cuda_rawkernel_selected",
        "ADF": "cuda_rawkernel_selected",
        "DF": "cuda_rawkernel_total_minus_complement",
    }


def test_virtual_image_support_tracks_mps_and_webgpu_contracts() -> None:
    from quantem.gpu.compute import virtual_image_kernel_support

    mps_u16 = virtual_image_kernel_support(
        backend="mps",
        shape=(512, 512, 192, 192),
        dtype=np.uint16,
    )
    assert mps_u16.custom_kernel is True
    assert mps_u16.kernel == "quantem.gpu.compute.mps MetalVirtualImage"
    assert mps_u16.mask_paths == {
        "BF": "mps_metal_selected",
        "ADF": "mps_metal_selected",
        "DF": "mps_metal_total_minus_complement",
    }

    mps_u8 = virtual_image_kernel_support(
        backend="mps",
        shape=(1024, 1024, 192, 192),
        dtype=np.uint8,
    )
    assert mps_u8.custom_kernel is True
    assert mps_u8.resident_gib == 36.0
    assert mps_u8.mask_paths == mps_u16.mask_paths

    mps_u32 = virtual_image_kernel_support(
        backend="mps",
        shape=(512, 512, 192, 192),
        dtype=np.uint32,
    )
    assert mps_u32.custom_kernel is True
    assert mps_u32.resident_gib == 36.0
    assert any("uint64 selected-sum" in note for note in mps_u32.notes)

    webgpu_u8 = virtual_image_kernel_support(
        backend="webgpu",
        shape=(1024, 1024, 192, 192),
        dtype=np.uint8,
    )
    assert webgpu_u8.custom_kernel is True
    assert "quantem.gpu.webgpu" in webgpu_u8.kernel
    assert webgpu_u8.mask_paths["BF"] == "webgpu_wgsl_selected"
    assert any("SwiftShader" in note for note in webgpu_u8.notes)

    webgpu_u32 = virtual_image_kernel_support(
        backend="webgpu",
        shape=(512, 512, 192, 192),
        dtype=np.uint32,
    )
    assert webgpu_u32.custom_kernel is True
    assert webgpu_u32.mask_paths == webgpu_u8.mask_paths
    assert any("mode 3" in note for note in webgpu_u32.notes)


def test_cuda_virtual_image_support_includes_uint32_path() -> None:
    from quantem.gpu.compute import virtual_image_kernel_support

    support = virtual_image_kernel_support(
        backend="cuda",
        shape=(512, 512, 192, 192),
        dtype=np.uint32,
        bf_radius=30,
    )

    assert support.available is True
    assert support.custom_kernel is True
    assert support.resident_gib == 36.0
    assert support.mask_paths == {
        "BF": "cuda_rawkernel_selected_u64_to_f32",
        "ADF": "cuda_rawkernel_selected_u64_to_f32",
        "DF": "cuda_rawkernel_total_minus_complement_u64_to_f32",
    }
    assert any("uint64 internal accumulation" in note for note in support.notes)


def test_mps_integer_reduction_kernel_sources_are_present() -> None:
    source = Path("src/quantem/gpu/compute/metal/reductions.msl").read_text(
        encoding="utf-8"
    )

    for name in (
        "masked_sum_u8",
        "detector_sum_u8",
        "bin_detector_u8",
        "mean_dp_sum_u8",
        "detector_sum_u8_block_partial",
        "detector_sum_u8_block_merge",
        "rowspan_sum_u8",
        "radial_cumsum_u8",
        "com_u8",
        "masked_sum_u32",
        "mean_dp_sum_u32",
        "rowspan_sum_u32",
        "radial_cumsum_u32",
        "com_u32",
    ):
        assert f"kernel void {name}" in source


def test_mps_u8_mean_dp_uses_resident_metal_detector_sum() -> None:
    """Lossless-u8 MPS data must not fall back to a host chunk reduction."""
    from types import SimpleNamespace

    from quantem.gpu.compute.backends import MetalRawBackend

    detector_sum = np.asarray([[8, 16], [24, 32]], dtype=np.float32)
    backend = MetalRawBackend.__new__(MetalRawBackend)
    backend._cf = SimpleNamespace(
        _np_dtype=np.dtype(np.uint8),
        vi=SimpleNamespace(detector_sum=lambda: detector_sum),
        chunks=[np.zeros((4, 2, 2), dtype=np.uint8)],
    )
    backend.det_shape = (2, 2)
    backend.n_frames = 4

    np.testing.assert_array_equal(
        backend.mean_dp(),
        np.asarray([[2, 4], [6, 8]], dtype=np.float32),
    )


def test_mps_mean_dp_reuses_exact_decode_side_detector_sum() -> None:
    """A decode-side sum must bypass the later full-stack Metal pass."""
    from types import SimpleNamespace

    from quantem.gpu.compute.backends import MetalRawBackend

    detector_sum = np.asarray([[8, 16], [24, 32]], dtype=np.uint32)

    def unexpected_reduction():
        raise AssertionError("resident detector sum was not reused")

    backend = MetalRawBackend.__new__(MetalRawBackend)
    backend._cf = SimpleNamespace(
        detector_sum=detector_sum,
        vi=SimpleNamespace(detector_sum=unexpected_reduction),
    )
    backend.n_frames = 4

    np.testing.assert_array_equal(
        backend.mean_dp(),
        np.asarray([[2, 4], [6, 8]], dtype=np.float32),
    )


def test_mps_mean_dp_has_no_host_chunk_fallback() -> None:
    """Unsupported MPS dtypes must raise in Metal code, never reduce on CPU."""
    from types import SimpleNamespace

    from quantem.gpu.compute.backends import MetalRawBackend

    def unsupported_detector_sum():
        raise NotImplementedError("native Metal reducer required")

    backend = MetalRawBackend.__new__(MetalRawBackend)
    backend._cf = SimpleNamespace(
        _np_dtype=np.dtype(np.uint32),
        vi=SimpleNamespace(detector_sum=unsupported_detector_sum),
        chunks=[np.ones((4, 2, 2), dtype=np.uint32)],
    )
    backend.det_shape = (2, 2)
    backend.n_frames = 4

    with pytest.raises(NotImplementedError, match="native Metal reducer"):
        backend.mean_dp()


def test_mps_production_reductions_do_not_route_to_numba() -> None:
    """Row-prefix and detector reductions must remain on the Metal backend."""
    source = Path("src/quantem/gpu/compute/mps.py").read_text(encoding="utf-8")
    backend_source = Path("src/quantem/gpu/compute/backends.py").read_text(
        encoding="utf-8"
    )

    assert "from numba import" not in source
    assert "_masked_sum_prefix_numba" not in source
    assert "gather_columns_float32" in source
    assert "for chunk in self._cf.chunks" not in backend_source
    assert "MPS reduce_frames(reduce='max') has no Metal kernel" in backend_source
    assert "raise NotImplementedError" in backend_source


def test_mps_integer_chunked_load_source_contract_is_present() -> None:
    msl = Path("src/quantem/gpu/io/backends/metal/bslz4.msl").read_text(
        encoding="utf-8"
    )
    hdf5_source = Path("src/quantem/gpu/io/hdf5.py").read_text(encoding="utf-8")
    mps_source = Path("src/quantem/gpu/io/backends/mps.py").read_text(
        encoding="utf-8"
    )

    assert "kernel void clip_u16_to_u8" in msl
    assert "kernel void shuf_8192_16_to_u8_batched" in msl
    assert "kernel void shuf_8192_16_to_u8_masked_batched" in msl
    assert "kernel void zero_bad_pixels_u32" in msl
    assert "kernel void clip_u32_to_u8" in msl
    assert "output_dtype=mps_chunk_output_dtype" in hdf5_source
    assert "output_dtype=np.uint8" in mps_source
    assert "np.uint32" in mps_source
    assert "cast_u8_out_mtl" in mps_source
    assert "fused_u8_load" in mps_source
    assert "_shuf16_u8_pipeline" in mps_source
    assert "_shuf16_u8_masked_pipeline" in mps_source
    assert "memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)" in mps_source
    assert "scratch_idx = ci % D" in mps_source
    assert "compact: bool = True" in mps_source
    assert "compact_target_gb: float = 1.5" in mps_source
    assert "and fast_det_bin is None" in mps_source
    assert (
        "grouping_frame_bytes = final_frame_bytes if output_u8 else frame_bytes"
        in mps_source
    )
    assert "cast_u8_out_mtl = u8_out_mtls[oi]" in mps_source
    assert "dec.drop_output_pool_refs()" in mps_source


def test_mps_dense_mask_uses_total_minus_complement_contract() -> None:
    source = Path("src/quantem/gpu/compute/backends.py").read_text(encoding="utf-8")

    assert "_total_cache" in source
    assert "_fast_total_cache" in source
    assert "return total - np.asarray(vi.masked_sum(~mask))" in source
    assert "_bin_mask" in source
