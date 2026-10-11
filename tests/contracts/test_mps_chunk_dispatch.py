"""Guardrails for chunk-backed MPS product dispatch.

These tests do not execute Metal on Linux. They pin how Metal frame views
forward column gathers and centre-of-mass reductions.
"""

import importlib

import numpy as np


class _ChunkSource:
    chunks = [object()]


def test_mps_float32_columns_forward_direct_output():
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    expected = np.empty((2, 4), dtype=np.float32)
    rows = np.array([1, 2])
    cols = np.array([3, 4])

    class FakeVI:
        def gather_columns_float32(self, got_rows, got_cols, *, out=None):
            np.testing.assert_array_equal(got_rows, rows)
            np.testing.assert_array_equal(got_cols, cols)
            assert out is expected
            return out

    frames = object.__new__(ChunkedFrames)
    frames.vi = FakeVI()

    assert frames.columns_float32_into(rows, cols, expected) is expected


def test_center_of_mass_dispatches_chunk_source_through_gpu_compute(monkeypatch):
    dpc = importlib.import_module("quantem.gpu.dpc")
    from quantem.gpu import detector

    class FakeSession:
        scan_shape = (2, 2)

        def center_of_mass(self, mask):
            assert mask.shape == (2, 2)
            com_row = np.array([20.0, 21.0, 22.0, 23.0], dtype=np.float32)
            com_col = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
            return com_row.reshape(2, 2), com_col.reshape(2, 2)

    monkeypatch.setattr(detector, "prepare", lambda _source: FakeSession())

    mask = np.ones((2, 2), dtype=bool)
    com_row, com_col = dpc.center_of_mass(_ChunkSource(), mask=mask)

    expected = np.array([[-1.5, -0.5], [0.5, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(com_row, expected)
    np.testing.assert_allclose(com_col, expected)
