"""Guardrails for chunk-backed MPS product dispatch.

These tests do not execute Metal on Linux. They pin the ownership boundary:
chunk-backed inputs must be wrapped and dispatched through ``quantem.gpu`` rather
than being rejected or routed back through widget UI code.
"""

from __future__ import annotations

import numpy as np
import importlib


class _ChunkSource:
    chunks = [object()]


def test_masked_sum_dispatches_chunk_source_through_gpu_compute(monkeypatch):
    from quantem.gpu import detector
    from quantem.gpu.compute import backends
    from quantem.gpu.compute import mps

    calls: list[object] = []

    class FakeChunkedFrames:
        _is_gpu_frames = True

        def __init__(self, source):
            self.source = source

    class FakeBackend:
        scan_shape = (2, 2)
        n_frames = 4

        def masked_sum(self, det_mask):
            assert det_mask.shape == (3, 3)
            return np.arange(4, dtype=np.float32).reshape(self.scan_shape)

    def fake_compute_backend(data):
        calls.append(data)
        return FakeBackend()

    monkeypatch.setattr(mps, "ChunkedFrames", FakeChunkedFrames)
    monkeypatch.setattr(backends, "compute_backend", fake_compute_backend)

    out = detector.masked_sum(_ChunkSource(), np.ones((3, 3), dtype=bool))

    np.testing.assert_array_equal(out, np.arange(4, dtype=np.float32).reshape(2, 2))
    assert len(calls) == 1
    assert isinstance(calls[0], FakeChunkedFrames)


def test_center_of_mass_dispatches_chunk_source_through_gpu_compute(monkeypatch):
    dpc = importlib.import_module("quantem.gpu.dpc")
    from quantem.gpu.compute import mps

    class FakeVI:
        n = 4
        det = (2, 2)

        def center_of_mass(self, mask):
            assert mask.shape == self.det
            com_col = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
            com_row = np.array([20.0, 21.0, 22.0, 23.0], dtype=np.float32)
            return com_col, com_row

    class FakeChunkedFrames:
        def __init__(self, source):
            self.source = source
            self.vi = FakeVI()

    monkeypatch.setattr(mps, "ChunkedFrames", FakeChunkedFrames)

    mask = np.ones((2, 2), dtype=bool)
    com_row, com_col = dpc.center_of_mass(_ChunkSource(), mask=mask)

    expected = np.array([[-1.5, -0.5], [0.5, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(com_row, expected)
    np.testing.assert_allclose(com_col, expected)
