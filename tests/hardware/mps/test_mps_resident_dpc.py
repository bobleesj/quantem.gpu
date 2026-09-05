"""Functional parity for Python orchestration of the shared DPC Metal kernels."""

from __future__ import annotations

import numpy as np
import pytest

from quantem.gpu.io.backends.mps.resident_dpc import (
    MPSDPCConfiguration,
    MPSDPCProcessor,
)


def test_python_mps_dpc_publishes_resident_phase_and_fft_seams() -> None:
    Metal = pytest.importorskip("Metal")
    row = np.asarray(
        [
            -1.5,
            -0.5,
            0.5,
            1.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
        ],
        dtype=np.float32,
    )
    column = np.asarray(
        [
            -1.5,
            -1.5,
            -1.5,
            -1.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            1.5,
            1.5,
            1.5,
            1.5,
        ],
        dtype=np.float32,
    )
    expected = np.asarray(
        [
            0,
            0.58474338,
            0.58474338,
            0,
            -0.58474338,
            0,
            0,
            -0.58474338,
            -0.58474338,
            0,
            0,
            -0.58474338,
            0,
            0.58474338,
            0.58474338,
            0,
        ],
        dtype=np.float32,
    )
    processor = MPSDPCProcessor()
    result = processor.process(
        row,
        column,
        MPSDPCConfiguration(
            scan_rows=4,
            scan_columns=4,
            rotation_degrees=17,
        ),
    )
    try:
        phase = np.frombuffer(
            result.phase_buffer.contents().as_buffer(expected.nbytes),
            dtype=np.float32,
        ).copy()
        np.testing.assert_allclose(phase, expected, rtol=0, atol=2e-5)
        assert result.gradient_fft_buffer.storageMode() == Metal.MTLStorageModePrivate
        assert result.phase_fft_buffer.storageMode() == Metal.MTLStorageModePrivate
        assert result.metrics.fft_dispatch_count == 13
        assert result.metrics.total_dispatch_count == 16
        assert result.metrics.upload_bytes == 128
        assert result.metrics.readback_bytes == 0
        assert result.metrics.synchronization_count == 1
    finally:
        result.release()


def test_python_mps_dpc_rejects_non_power_of_two_shape() -> None:
    configuration = MPSDPCConfiguration(
        scan_rows=3,
        scan_columns=4,
        rotation_degrees=0,
    )

    with pytest.raises(ValueError, match="power-of-two"):
        _ = configuration.count
