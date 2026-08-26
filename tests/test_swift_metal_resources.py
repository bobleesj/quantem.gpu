from importlib.resources import files
from pathlib import Path


def test_swift_package_names_the_native_backend_explicitly() -> None:
    package = (Path(__file__).parents[1] / "Package.swift").read_text(
        encoding="utf-8"
    )

    assert 'platforms: [.macOS(.v14), .iOS(.v17)]' in package
    assert '.library(name: "Metal4DSTEMKernels"' in package
    assert '.library(name: "MetalDisplayKernels"' in package


def test_native_4dstem_metal_resources_are_packaged() -> None:
    root = (
        files("quantem.gpu")
        / "swift"
        / "Sources"
        / "Metal4DSTEMKernels"
        / "Resources"
    )
    qh5idx = (root / "qh5idx.metal").read_text(encoding="utf-8")
    detector = (root / "detector.metal").read_text(encoding="utf-8")

    assert "kernel void h5lz4dc_unshuffle_source_u8_qh5idx" in qh5idx
    assert "kernel void h5lz4dc_unshuffle_u16_qh5idx" in qh5idx
    assert (
        "kernel void "
        "h5lz4dc_unshuffle_u16_identity_audited_single_block_qh5idx"
        in qh5idx
    )
    assert (
        "kernel void h5lz4dc_unshuffle_u16_single_block_packed_h5"
        in qh5idx
    )
    assert (
        "kernel void "
        "h5lz4dc_bin_u16_audited_low8_scalar_u16_frame_major_row8_qh5idx"
        in qh5idx
    )
    assert (
        "kernel void "
        "h5lz4dc_unshuffle_u16_audited_low8_tile4_octet192_"
        "word_major_products_qh5idx"
        in qh5idx
    )
    assert "kernel void detector_products_u8" in detector
    assert "kernel void detector_products_u8_word_major" in detector
    assert (
        "kernel void contiguous_detector_bin1_u16_products_"
        "detector_partials_tiled32x8" in detector
    )
    assert "kernel void detector_accumulate_u32_partials_u64" in detector
    assert "kernel void transpose_scan_words" in detector
    assert "kernel void signed_delta_u16_word_major" in detector


def test_native_4dstem_resources_exclude_experimental_entry_points() -> None:
    root = (
        files("quantem.gpu")
        / "swift"
        / "Sources"
        / "Metal4DSTEMKernels"
        / "Resources"
    )
    qh5idx = (root / "qh5idx.metal").read_text(encoding="utf-8")
    detector = (root / "detector.metal").read_text(encoding="utf-8")

    assert "kernel void h5lz4dc_qh5idx" not in qh5idx
    assert "kernel void h5lz4dc_unshuffle_u8_qh5idx" not in qh5idx
    assert "kernel void h5lz4dc_frame_low8_qh5idx" not in qh5idx
    assert "kernel void shuf_8192_16_batched" not in detector


def test_packed_u16_decoder_keeps_its_four_simdgroup_launch_contract() -> None:
    root = files("quantem.gpu")
    qh5idx = (
        root
        / "swift"
        / "Sources"
        / "Metal4DSTEMKernels"
        / "Resources"
        / "qh5idx.metal"
    ).read_text(encoding="utf-8")
    decoder = (
        root / "io" / "backends" / "mps" / "decoder.py"
    ).read_text(encoding="utf-8")

    assert "group < 128u; group += 4u" in qh5idx
    assert "exactly four 32-lane SIMD groups (128 threads)" in qh5idx
    assert "Metal.MTLSizeMake(128, 1, 1)" in decoder
    assert "requires exactly 4 x 32 = 128 threads" in decoder
