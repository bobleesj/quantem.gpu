"""The beginner SSB path: SSB(data) finds the rotation, reconstruct() fits once, show() names its package."""

import math
import sys

import numpy as np
import pytest


def _rotated_gradient_scan(angle_deg, n=32, detector=24):
    """Counts whose bright-field disk moves along a smooth gradient field rotated by ``angle_deg`` into the detector.

    The centre-of-mass field is then curl-free only after rotating back by that angle, as for a thin phase object.
    """
    rows, cols = np.mgrid[:n, :n].astype(np.float64)
    phase = sum(np.exp(-((rows - r) ** 2 + (cols - c) ** 2) / 18.0) for r, c in ((9, 11), (21, 8), (15, 23), (25, 25)))
    grad_r, grad_c = np.gradient(phase)
    a = math.radians(angle_deg)
    shift_r = 1.5 * (math.cos(a) * grad_r - math.sin(a) * grad_c)
    shift_c = 1.5 * (math.sin(a) * grad_r + math.cos(a) * grad_c)
    det_r, det_c = np.mgrid[:detector, :detector].astype(np.float64)
    center = (detector - 1) / 2
    distance = np.hypot(det_r[None, None] - center - shift_r[..., None, None],
                        det_c[None, None] - center - shift_c[..., None, None])
    disk = np.clip(6.0 - distance + 0.5, 0.0, 1.0)  # a soft-edged disk of radius 6 px
    return (400 * disk + 5).astype(np.uint16)


def _options():
    return dict(backend="mps", voltage_kV=300, semiangle_mrad=30, scan_sampling_A=(0.3, 0.3))


def test_omitted_rotation_is_estimated_like_dpc_and_labelled(capsys):
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB, dpc

    counts = _rotated_gradient_scan(35.0)
    expected = dpc.run(counts).scan_detector_rotation_mod180_deg % 180.0
    with SSB(counts, **_options()) as ssb:
        assert ssb.scan_detector_rotation_deg == pytest.approx(expected, abs=1e-6)
        # the angle rotates the measured field back onto the scan axes, so it undoes the simulated 35 degrees
        assert ssb.scan_detector_rotation_deg == pytest.approx((-35.0) % 180.0, abs=1.0)
        assert ssb.scan_detector_rotation_flipped is False
        assert ssb.scan_detector_rotation_source == "estimated"
        table = ssb.calibration_table()
        assert table.loc["Scan-detector rotation", "Source"] == "Estimated (centre-of-mass curl)"
        assert table.loc["Rotation flipped", "Value used"] == "no"
    assert "estimated from the centre-of-mass curl" in capsys.readouterr().out


def test_a_given_rotation_is_used_as_given_without_an_estimate(capsys):
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    with SSB(_rotated_gradient_scan(35.0), scan_detector_rotation_deg=200.0, **_options()) as ssb:
        assert ssb.scan_detector_rotation_deg == 200.0
        assert ssb.scan_detector_rotation_flipped is False
        assert ssb.calibration_table().loc["Scan-detector rotation", "Source"] == "Explicit override"
    assert "centre-of-mass curl" not in capsys.readouterr().out


def test_a_recorded_rotation_wins_over_the_estimate():
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB, io

    metadata = {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": (0.3, 0.3),
                "scan_detector_rotation_deg": 12.0, "scan_detector_rotation_flipped": True}
    data = io.Dataset4dstemGPU(_rotated_gradient_scan(35.0), metadata)
    with SSB(data, backend="mps") as ssb:
        assert (ssb.scan_detector_rotation_deg, ssb.scan_detector_rotation_flipped) == (12.0, True)
        assert ssb.scan_detector_rotation_source == "metadata"


def test_reconstruct_fits_once_when_no_aberrations_are_known(capsys):
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    with SSB(_rotated_gradient_scan(0.0), scan_detector_rotation_deg=0.0, **_options()) as ssb:
        fits = []
        find = ssb.find_aberrations
        ssb.find_aberrations = lambda **options: fits.append(options) or find(num_trials=20, **options)
        first = ssb.reconstruct()
        second = ssb.reconstruct()
        given = ssb.reconstruct(aberrations={"C10_nm": 1.0})
    assert len(fits) == 1
    assert first.aberrations == second.aberrations
    assert given.aberrations["C10_nm"] == 1.0
    assert capsys.readouterr().out.count("fitting them with find_aberrations() first") == 1


def test_show_names_the_package_it_needs(monkeypatch):
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    monkeypatch.setitem(sys.modules, "quantem.widget", None)
    with SSB(_rotated_gradient_scan(0.0), scan_detector_rotation_deg=0.0, **_options()) as ssb:
        with pytest.raises(ImportError, match="pip install quantem.widget"):
            ssb.show()
