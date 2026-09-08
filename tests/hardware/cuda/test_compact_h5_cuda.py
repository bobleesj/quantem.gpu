"""CUDA compact-HDF5 source and fail-closed contract tests."""

from __future__ import annotations

import hashlib
from importlib import import_module
from importlib.resources import files
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("failure", ["cancel", "allocation"])
def test_failed_load_closes_source_mapping(tmp_path, monkeypatch, failure):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    module = import_module("quantem.gpu.io.backends.cuda.packed")
    spec = spec_from_file_location(
        "compact_failure_fixture", Path(__file__).parents[2] / "contracts/io/test_compact_h5.py"
    )
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    source = tmp_path / "packed.h5"
    fixture._write_fixture(source, np.arange(24, dtype=np.uint16).reshape(6, 4))
    mappings = []
    original_mmap = module.mmap.mmap

    def capture_mapping(*args, **kwargs):
        result = original_mmap(*args, **kwargs)
        mappings.append(result)
        return result

    monkeypatch.setattr(module.mmap, "mmap", capture_mapping)
    if failure == "allocation":

        def unavailable(*args, **kwargs):
            raise MemoryError("injected CUDA allocation failure")

        monkeypatch.setattr(cp, "empty", unavailable)
        with pytest.raises(MemoryError, match="injected CUDA allocation"):
            module.load_compact_h5_cuda(source)
    else:
        checks = iter([False, True])
        with pytest.raises(RuntimeError, match="cancelled before shard"):
            module.load_compact_h5_cuda(source, should_cancel=lambda: next(checks))
    assert mappings and all(mapping.closed for mapping in mappings)


def test_batched_decoder_preserves_counts_and_reports_a_damaged_chunk() -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device is available")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime is available")
    from quantem.gpu.io.backends.cuda.packed import _cuda_kernels

    kernels, _ = _cuda_kernels(cp.cuda.Device().id)
    decoded = cp.empty(2, dtype=cp.uint8)
    for payload, offsets, expected_status in [
        ([0x10, 7, 0x10, 11], [0, 2, 4], 0),
        ([0x10, 7, 0], [0, 2, 3], (1 << 32) | 9),
    ]:
        compressed = cp.asarray(payload, dtype=cp.uint8)
        input_offsets = cp.asarray(offsets, dtype=cp.uint32)
        status = cp.zeros(1, dtype=cp.uint64)
        kernels["compact_h5_lz4_decode"](
            (1,),
            (256,),
            (
                compressed,
                input_offsets,
                decoded,
                np.uint32(2),
                np.uint32(1),
                np.uint32(2),
                status,
            ),
        )
        assert int(status.get()[0]) == expected_status
        if expected_status == 0:
            np.testing.assert_array_equal(decoded.get(), [7, 11])


def test_parallel_chunk_integrity_checks_every_stored_byte(tmp_path) -> None:
    from quantem.gpu.io.backends.cuda.packed import _timed_chunk_sha256_file

    path = tmp_path / "compact.h5"
    payload = bytes(range(251)) * 19
    path.write_bytes(payload)
    chunk_bytes = 1024
    expected = tuple(
        hashlib.sha256(payload[offset : offset + chunk_bytes]).hexdigest()
        for offset in range(0, len(payload), chunk_bytes)
    )

    elapsed_ms = _timed_chunk_sha256_file(path, expected, chunk_bytes)

    assert elapsed_ms >= 0


def test_parallel_chunk_integrity_rejects_one_changed_chunk(tmp_path) -> None:
    from quantem.gpu.io.backends.cuda.packed import _timed_chunk_sha256_file

    path = tmp_path / "compact.h5"
    path.write_bytes(b"a" * 2048)

    with pytest.raises(ValueError, match="chunk 1 SHA-256"):
        _timed_chunk_sha256_file(
            path,
            (
                hashlib.sha256(b"a" * 1024).hexdigest(),
                hashlib.sha256(b"b" * 1024).hexdigest(),
            ),
            1024,
        )


def test_cuda_compact_source_uses_nvrtc_and_never_declares_dense_residency() -> None:
    source = (
        files("quantem.gpu")
        .joinpath("io/backends/cuda/packed.py")
        .read_text(encoding="utf-8")
    )

    assert 'backend="nvrtc"' in source
    assert "compact_h5_lz4_decode" in source
    assert "compact_h5_validate_descriptors" in source
    assert "compact_h5_validate_compact_headers" in source
    assert "compact_h5_selected_diffraction" in source
    assert "compact_h5_detector_columns" in source
    assert "compact_h5_detector_update" in source
    assert "expected_whole_file_sha256" in source
    assert "_load_compact_h5_cuda_v3" in source
    assert "decoded.view(cp.uint32)" in source
    assert "resident_bytes=index.resident_bytes" in source
    assert "fft_dispatch_count: int = 0" in source
    assert "np.empty(index.logical_source_bytes" not in source
    assert "cp.empty(index.logical_source_bytes" not in source


def test_cuda_compact_source_reports_logical_and_resident_bytes_separately() -> None:
    from quantem.gpu.io.backends.cuda.packed import CudaCompactH5ResidentSource

    source = CudaCompactH5ResidentSource.__new__(CudaCompactH5ResidentSource)
    source.whole_file_sha256 = "a" * 64
    source.metadata = SimpleNamespace(
        shape=(512, 512, 192, 192),
        schema_version=1,
        source_identity_sha256="b" * 64,
        logical_source_bytes=19_327_352_832,
        resident_bytes=2_453_722_256,
        detector_calibration={"schema": "quantem.gpu.detector-calibration/v1"},
        manifest={
            "schema": "quantem.gpu.packed-detector-h5/v1",
            "scan_bin": 1,
            "detector_bin": 1,
            "crop": None,
        },
    )

    assert source.shape == (512, 512, 192, 192)
    assert source.dtype == np.dtype(np.uint16)
    assert source.nbytes == 19_327_352_832
    assert source.source_provenance["packed_resident_bytes"] == 2_453_722_256
    assert source.source_provenance["whole_file_sha256"] == "a" * 64


def test_cuda_compact_loader_fails_with_corrective_message_without_cupy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantem.gpu.io.backends.cuda import packed as compact_h5

    monkeypatch.setattr(compact_h5, "cp", None)
    with pytest.raises(RuntimeError, match="requires CuPy and an NVIDIA CUDA device"):
        compact_h5.load_compact_h5_cuda("not-opened.h5")


def test_cuda_compact_loader_requires_v3_whole_file_seal_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantem.gpu.io.backends.cuda import packed as compact_h5

    monkeypatch.setattr(compact_h5, "cp", object())
    monkeypatch.setattr(
        compact_h5.CompactH5Index,
        "from_file",
        lambda path: SimpleNamespace(
            schema_version=3,
            header_encoding=1,
            require_raw_reconstruction=lambda: None,
        ),
    )
    with pytest.raises(ValueError, match="expected_whole_file_sha256"):
        compact_h5.load_compact_h5_cuda("not-opened.h5")


def test_cuda_compact_loader_rejects_v3_whole_file_mismatch_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantem.gpu.io.backends.cuda import packed as compact_h5

    index = SimpleNamespace(
        schema_version=3,
        header_encoding=1,
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
    from quantem.gpu.io.backends.cuda.packed import (
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
