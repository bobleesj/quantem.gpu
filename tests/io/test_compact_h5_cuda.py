"""CUDA compact-HDF5 source and fail-closed contract tests."""

from __future__ import annotations

from importlib.resources import files
from types import SimpleNamespace

import numpy as np
import pytest


def test_cuda_compact_source_uses_nvrtc_and_never_declares_dense_residency() -> None:
    source = (
        files("quantem.gpu")
        .joinpath("io/backends/cuda/compact_h5.py")
        .read_text(encoding="utf-8")
    )

    assert 'backend="nvrtc"' in source
    assert "compact_h5_lz4_decode" in source
    assert "compact_h5_validate_descriptors" in source
    assert "compact_h5_validate_compact_headers" in source
    assert "compact_h5_selected_diffraction" in source
    assert "compact_h5_detector_update" in source
    assert "expected_whole_file_sha256" in source
    assert "_load_compact_h5_cuda_v3" in source
    assert "decoded.view(cp.uint32)" in source
    assert "resident_bytes=index.resident_bytes" in source
    assert "fft_dispatch_count: int = 0" in source
    assert "np.empty(index.logical_source_bytes" not in source
    assert "cp.empty(index.logical_source_bytes" not in source


def test_cuda_compact_loader_fails_with_corrective_message_without_cupy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantem.gpu.io.backends.cuda import compact_h5

    monkeypatch.setattr(compact_h5, "cp", None)
    with pytest.raises(RuntimeError, match="requires CuPy and an NVIDIA CUDA device"):
        compact_h5.load_compact_h5_cuda("not-opened.h5")


def test_cuda_compact_loader_requires_v3_whole_file_seal_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantem.gpu.io.backends.cuda import compact_h5

    monkeypatch.setattr(compact_h5, "cp", object())
    monkeypatch.setattr(
        compact_h5.CompactH5Index,
        "from_file",
        lambda path: SimpleNamespace(
            schema_version=3,
            require_raw_reconstruction=lambda: None,
        ),
    )
    with pytest.raises(ValueError, match="expected_whole_file_sha256"):
        compact_h5.load_compact_h5_cuda("not-opened.h5")


def test_cuda_compact_loader_rejects_v3_whole_file_mismatch_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantem.gpu.io.backends.cuda import compact_h5

    index = SimpleNamespace(
        schema_version=3,
        path="not-opened.h5",
        require_raw_reconstruction=lambda: None,
    )
    monkeypatch.setattr(compact_h5, "cp", object())
    monkeypatch.setattr(compact_h5.CompactH5Index, "from_file", lambda path: index)
    monkeypatch.setattr(compact_h5, "_sha256_file", lambda path: "1" * 64)
    with pytest.raises(ValueError, match="whole-file SHA-256"):
        compact_h5.load_compact_h5_cuda(
            "not-opened.h5", expected_whole_file_sha256="0" * 64
        )


def test_v3_cuda_width_summary_reads_compact_nibbles_without_expansion() -> None:
    from quantem.gpu.io.backends.cuda.compact_h5 import (
        _header_words_per_pixel,
        _update_v3_maximum_widths,
    )

    index = SimpleNamespace(header_encoding=1)
    tile_count = 40
    header_words_per_pixel = _header_words_per_pixel(index, tile_count)
    assert header_words_per_pixel == 7
    headers = np.zeros((2, header_words_per_pixel), dtype="<u4")
    checkpoint_words = 2
    headers[0, checkpoint_words] = np.uint32(1 | (7 << 4))
    headers[0, checkpoint_words + 4] = np.uint32(8)
    headers[1, checkpoint_words + 2] = np.uint32(6 << 12)
    maximum_widths = np.zeros(2, dtype=np.uint8)

    _update_v3_maximum_widths(
        memoryview(headers.tobytes()),
        detector_pixel_count=2,
        tile_count=tile_count,
        header_words_per_pixel=header_words_per_pixel,
        maximum_widths=maximum_widths,
    )

    assert maximum_widths.tolist() == [8, 6]
