from importlib.resources import files


def test_native_4dstem_metal_resources_are_packaged() -> None:
    root = (
        files("quantem.gpu")
        / "swift"
        / "Sources"
        / "QuantEM4DSTEMMetal"
        / "Resources"
    )
    qh5idx = (root / "qh5idx.metal").read_text(encoding="utf-8")
    detector = (root / "detector.metal").read_text(encoding="utf-8")

    assert "kernel void h5lz4dc_unshuffle_source_u8_qh5idx" in qh5idx
    assert "kernel void h5lz4dc_unshuffle_u16_qh5idx" in qh5idx
    assert "kernel void detector_products_u8" in detector
    assert "kernel void transpose_scan_words" in detector
    assert "kernel void signed_delta_u16_word_major" in detector


def test_native_4dstem_resources_exclude_experimental_entry_points() -> None:
    root = (
        files("quantem.gpu")
        / "swift"
        / "Sources"
        / "QuantEM4DSTEMMetal"
        / "Resources"
    )
    qh5idx = (root / "qh5idx.metal").read_text(encoding="utf-8")
    detector = (root / "detector.metal").read_text(encoding="utf-8")

    assert "kernel void h5lz4dc_qh5idx" not in qh5idx
    assert "kernel void h5lz4dc_unshuffle_u8_qh5idx" not in qh5idx
    assert "kernel void h5lz4dc_frame_low8_qh5idx" not in qh5idx
    assert "kernel void shuf_8192_16_batched" not in detector
