import re
from importlib.resources import files
from pathlib import Path

# Kernels Python also compiles ship in the wheel; Swift-only kernels stay in native/.
KERNELS = files("quantem.gpu") / "io" / "hdf5" / "mps" / "kernels"
NATIVE_RESOURCES = (
    Path(__file__).parents[2] / "native/swift/Sources/Metal4DSTEMKernels/Resources"
)


def test_swift_package_names_the_native_backend_explicitly() -> None:
    package = (Path(__file__).parents[2] / "Package.swift").read_text(
        encoding="utf-8"
    )

    assert 'platforms: [.macOS(.v14), .iOS(.v17)]' in package
    assert '.library(name: "Metal4DSTEMKernels"' in package
    assert '.library(name: "MetalDisplayKernels"' in package
    # Resources Python also reads have one copy, inside the Python package.
    shared = re.findall(r'\.copy\("\\\(pythonPackage\)/([^"]+)"\)', package)
    assert sorted(shared) == [
        "display/colormaps.json",
        "display/metal/display.metal",
        "io/hdf5/mps/kernels/qh5idx.metal",
        "io/hdf5/mps/kernels/save_uint16.msl",
        "resident/mps/kernels/count_tables.msl",
        "resident/mps/kernels/hot_pixels.msl",
        "resident/mps/kernels/precision.msl",
        "resident/mps/kernels/runtime_spatial.msl",
        "resident/mps/kernels/streamed_counts.msl",
    ]
    native = Path(__file__).parents[2] / "native"
    for relative in shared:
        resource = files("quantem.gpu") / relative
        assert resource.is_file(), relative
        assert not list(native.rglob(resource.name)), f"second copy of {relative}"


def test_native_4dstem_metal_resources_are_packaged() -> None:
    qh5idx = (KERNELS / "qh5idx.metal").read_text(encoding="utf-8")
    detector = (NATIVE_RESOURCES / "detector.metal").read_text(encoding="utf-8")

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
    qh5idx = (KERNELS / "qh5idx.metal").read_text(encoding="utf-8")
    detector = (NATIVE_RESOURCES / "detector.metal").read_text(encoding="utf-8")

    assert "kernel void h5lz4dc_qh5idx" not in qh5idx
    assert "kernel void h5lz4dc_unshuffle_u8_qh5idx" not in qh5idx
    assert "kernel void h5lz4dc_frame_low8_qh5idx" not in qh5idx
    assert "kernel void shuf_8192_16_batched" not in detector


def test_packed_u16_decoder_keeps_its_four_simdgroup_launch_contract() -> None:
    qh5idx = (KERNELS / "qh5idx.metal").read_text(encoding="utf-8")
    decoder = (
        files("quantem.gpu") / "io" / "hdf5" / "mps" / "decode.py"
    ).read_text(encoding="utf-8")

    assert "group < 128u; group += 4u" in qh5idx
    assert "exactly four 32-lane SIMD groups (128 threads)" in qh5idx
    assert "Metal.MTLSizeMake(128, 1, 1)" in decoder
    assert "requires exactly 4 x 32 = 128 threads" in decoder
