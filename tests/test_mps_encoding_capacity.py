"""Portable admission checks for large detector geometries without allocating counts."""

from types import SimpleNamespace

import pytest

from quantem.gpu.resident.mps import counts


@pytest.mark.parametrize("detector, expected", [(192, 16384), (512, 7680), (1024, 1536), (2048, 509)])
def test_encoding_window_respects_uint32_offsets(monkeypatch, detector, expected):
    monkeypatch.setattr(counts, "metal_device", lambda: SimpleNamespace(maxBufferLength=lambda: 1 << 35))
    source = object.__new__(counts.MPSStreamedCounts)
    source.shape = (128, 128, detector, detector)
    accepted = source._encoding_scan_limit(16384)
    assert accepted == expected
    assert source._encoding_scratch_bytes(accepted) < 1 << 32
    if detector != 192:
        assert source._encoding_scratch_bytes(accepted + (512 if accepted >= 512 else 1)) >= 1 << 32


def test_smaller_device_limits_window_before_allocation(monkeypatch):
    monkeypatch.setattr(counts, "metal_device", lambda: SimpleNamespace(maxBufferLength=lambda: 1 << 28))
    source = object.__new__(counts.MPSStreamedCounts)
    source.shape = (128, 128, 192, 192)
    source.is_released = False
    source._encoding_workspace_buffer = None
    assert source._encoding_scan_limit(16384) == 3584
    with (pytest.raises(MemoryError, match="buffer limit"),
          source.encoding_workspace(8192)):
        pytest.fail("An oversized workspace must be refused before allocation.")
