from __future__ import annotations

from pathlib import Path

import numpy as np


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

    webgpu_u8 = virtual_image_kernel_support(
        backend="webgpu",
        shape=(1024, 1024, 192, 192),
        dtype=np.uint8,
    )
    assert webgpu_u8.custom_kernel is True
    assert "quantem.gpu.webgpu" in webgpu_u8.kernel
    assert webgpu_u8.mask_paths["BF"] == "webgpu_wgsl_selected"
    assert any("SwiftShader" in note for note in webgpu_u8.notes)


def test_mps_uint8_reduction_kernel_sources_are_present() -> None:
    source = Path("src/quantem/gpu/compute/metal/reductions.msl").read_text(
        encoding="utf-8"
    )

    for name in (
        "masked_sum_u8",
        "detector_sum_u8",
        "bin_detector_u8",
        "mean_dp_sum_u8",
        "rowspan_sum_u8",
        "radial_cumsum_u8",
        "com_u8",
    ):
        assert f"kernel void {name}" in source


def test_mps_uint8_chunked_load_source_contract_is_present() -> None:
    msl = Path("src/quantem/gpu/io/backends/metal/bslz4.msl").read_text(
        encoding="utf-8"
    )
    hdf5_source = Path("src/quantem/gpu/io/hdf5.py").read_text(encoding="utf-8")
    mps_source = Path("src/quantem/gpu/io/backends/mps.py").read_text(
        encoding="utf-8"
    )

    assert "kernel void clip_u16_to_u8" in msl
    assert "output_dtype=mps_chunk_output_dtype" in hdf5_source
    assert "output_dtype=np.uint8" in mps_source
    assert "cast_u8_out_mtl" in mps_source
    assert "if output_u8 or output_u16_narrow:" in mps_source
    assert "scratch_idx = ci % D" in mps_source
    assert "dec.drop_output_pool_refs()" in mps_source


def test_mps_dense_mask_uses_total_minus_complement_contract() -> None:
    source = Path("src/quantem/gpu/compute/backends.py").read_text(encoding="utf-8")

    assert "_total_cache" in source
    assert "_fast_total_cache" in source
    assert "return total - np.asarray(vi.masked_sum(~mask))" in source
    assert "_bin_mask" in source
