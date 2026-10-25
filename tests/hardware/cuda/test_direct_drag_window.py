"""Drag windows: the Chebyshev statistics along one control against exact preparation.

At a Chebyshev node the window reproduces that node's own exact statistics, so
the images differ only by the solver (float64 Cholesky here, eigenbasis or LU
in PreparedDirect.solve), each rounded once to a complex64 coefficient: a few
float32 ulps of the largest pixel (tolerance 1e-6). Between nodes, the window
adds its truncation (below 4e-7 of the largest Gram entry at N = 12, h = 3 /
max|u|) to the difference between two float32 transfer evaluations, which is
the 3e-7 that test_ssb_prepared_methods.py documents (tolerance 2e-6, as
there). log|w| of a complex64 wave near |w| = 1 is quantized at 2**-23.
"""

import cupy as cp
import numpy as np
import pytest

from quantem.gpu import DirectPtycho
from quantem.gpu.ssb import direct_workflow
from quantem.gpu.ssb.cuda.direct import DragWindow, PreparedDirect

pytestmark = pytest.mark.slow

ABERRATIONS = {"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 40.0, "C21_nm": 150.0, "phi21_deg": -20.0, "C30_nm": 2000.0}
OPTICS = {"aberrations": ABERRATIONS, "sample_tilt_mrad": (1.5, -0.8), "depth_spread_nm": 8.0}
OPTIONS = dict(smoothness_inv_A=1.0, detector_weighting="uniform")


@pytest.fixture(scope="module")
def ptycho():
    """A 128 x 128 scan of a weak, varying specimen on a 64 x 64 detector, about 1300 BF pixels."""
    rng = np.random.default_rng(7)
    row, col = cp.indices((64, 64))
    disk = ((row - 31.5) ** 2 + (col - 31.5) ** 2 < 20.5**2).astype(cp.float32)
    scan_row, scan_col = cp.indices((128, 128), dtype=cp.float32)
    contrast = cp.asarray(rng.normal(size=(6, 2)), cp.float32)
    pattern = 1.0
    for wave_row, wave_col in contrast.get():
        pattern = pattern + 0.02 * cp.cos(0.3 * (wave_row * scan_row + wave_col * scan_col))[:, :, None, None] * cp.cos(
            0.2 * (wave_row * row + wave_col * col)
        )
    counts = cp.random.poisson(400 * disk * pattern).astype(cp.float32)
    session = DirectPtycho(
        counts, backend="cuda", voltage_kV=300.0, semiangle_mrad=25.0, scan_sampling_A=0.4,
        detector_sampling_mrad=1.25, bright_field_center_px=(31.5, 31.5), bright_field_radius_px=20.5,
        scan_detector_rotation_deg=0.0, aberrations=ABERRATIONS,
    )
    yield session
    session.close()


def _differences(wave, expected, complex_object):
    """Largest phase (and log-amplitude) difference over its tolerance."""
    parts = [(cp.angle, 0.0)] + ([(lambda w: cp.log(cp.abs(w)), 4 * 2.0**-23)] if complex_object else [])
    return [(float(cp.abs(part(wave) - part(expected)).max()), float(cp.abs(part(expected)).max()), floor)
            for part, floor in parts]


@pytest.mark.parametrize("control", ["C10_nm", "phi12_deg", "C30_nm"])
def test_window_matches_exact_preparation(ptycho, control):
    """Nodes, window edges and points between them, both methods, and phase from a complex window."""
    engine = ptycho._backend()._get_accelerator()
    for complex_object in (False, True):
        window = DragWindow(engine, OPTICS, control, 3, complex_object=complex_object, **OPTIONS)
        while not window.step():
            pass
        amplitude = 0.05 if complex_object else None
        node = window.nodes[3]
        for value, relative in [(node, 1e-6)] + [(window.center + window.half_width * x, 2e-6) for x in (-1.0, -0.43, 0.71, 1.0)]:
            aberrations = {**ABERRATIONS, control: value}
            exact = PreparedDirect.prepare(engine, {**OPTICS, "aberrations": aberrations}, 3,
                                           complex_object=complex_object, **OPTIONS)
            wave, _ = window.solve(value, 0.03, amplitude, complex_object=complex_object)
            expected, _ = exact.solve(0.03, amplitude, compute_loss=False)
            for difference, scale, floor in _differences(wave, expected, complex_object):
                assert difference <= max(relative * scale, floor)
            if complex_object:
                phase_only = PreparedDirect.prepare(engine, {**OPTICS, "aberrations": aberrations}, 3,
                                                    complex_object=False, **OPTIONS)
                wave, _ = window.solve(value, 0.03, None, complex_object=False)
                expected, _ = phase_only.solve(0.03, None, compute_loss=False)
                [(difference, scale, _)] = _differences(wave, expected, False)
                assert difference <= relative * scale


def test_drag_builds_between_frames_and_rebuilds_outside(ptycho):
    """Exact frames until the window is complete, window frames inside it, a new window outside it."""
    policy = direct_workflow.settings("phase", 0.03, None, None, None)
    coefs, tilt, depth = dict(ABERRATIONS), OPTICS["sample_tilt_mrad"], OPTICS["depth_spread_nm"]
    ptycho._direct_window = None
    first, _ = direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": 12.1}, tilt, depth, 3, policy)
    exact, _ = direct_workflow.solve(ptycho, {**coefs, "C10_nm": 12.1}, tilt, depth, 3, policy, compute_loss=False)
    cp.testing.assert_array_equal(first, exact)
    while not direct_workflow.advance(ptycho):
        pass
    window = ptycho._direct_window[1]
    assert window.ready and window.center == 12.1
    inside = window.center + 0.6 * window.half_width
    wave, _ = direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": inside}, tilt, depth, 3, policy)
    expected, _ = direct_workflow.solve(ptycho, {**coefs, "C10_nm": inside}, tilt, depth, 3, policy, compute_loss=False)
    [(difference, scale, _)] = _differences(wave, expected, False)
    assert difference <= 2e-6 * scale
    outside = window.center + 1.5 * window.half_width
    direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": outside}, tilt, depth, 3, policy)
    assert ptycho._direct_window[1] is not window and ptycho._direct_window[1].center == outside
    direct_workflow.drag(ptycho, "C12_nm", {**coefs, "C10_nm": outside, "C12_nm": 3.2}, tilt, depth, 3, policy)
    assert ptycho._direct_window[1].control == "C12_nm" and not ptycho._direct_window[1].ready
