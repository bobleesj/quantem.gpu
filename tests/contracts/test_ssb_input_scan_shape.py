"""Acquisition dimensions remain independent of FFT preparation and padding."""

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.workflow import SSB


@pytest.mark.parametrize("shape, scan_shape, expected", [
    ((64, 48, 4, 4), None, (64, 48)),
    ((64 * 48, 4, 4), (64, 48), (64, 48)),
    ((64 * 64, 4, 4), None, (64, 64)),
])
def test_input_dimensions_do_not_prepare_fourier_grid(shape, scan_shape, expected):
    session = SSB.__new__(SSB)
    session._data = np.empty(shape, dtype=np.uint8)
    session._scan_shape = scan_shape
    # Deliberately no backend: metadata must be available before preparation.
    assert session.input_scan_shape == expected


def test_input_dimensions_reject_ambiguous_flattened_scan():
    session = SSB.__new__(SSB)
    session._data = np.empty((12, 4, 4), dtype=np.uint8)
    session._scan_shape = None
    with pytest.raises(ValueError, match="scan_shape"):
        _ = session.input_scan_shape


def test_column_export_keeps_measured_scan_dimensions():
    from quantem.gpu.direct_ptycho.mps.frames import MpsBfColumnFrames

    session = SSB.__new__(SSB)
    session._data = MpsBfColumnFrames.__new__(MpsBfColumnFrames)
    session._data.scan_shape = (64, 48)
    assert session.input_scan_shape == (64, 48)
