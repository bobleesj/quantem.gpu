"""Drag windows: the Chebyshev series along one control against exact preparation and SSB preview.

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
from quantem.gpu.ssb.cuda import chebyshev
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
        node = window.series.nodes[3]
        for value, relative in [(node, 1e-6)] + [(window.series.center + window.series.half_width * x, 2e-6) for x in (-1.0, -0.43, 0.71, 1.0)]:
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

    def exact(aberrations):
        return ptycho.preview(aberrations, method="phase", phase_regularization=0.03, upsampling_factor=3,
                              sample_tilt_mrad=tilt, depth_spread_nm=depth, compute_loss=False, output="native")[0]

    ptycho._direct_windows = {}
    first = direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": 12.1}, tilt, depth, 3, policy)
    cp.testing.assert_array_equal(first, exact({**coefs, "C10_nm": 12.1}))
    while not direct_workflow.advance(ptycho):
        pass
    window = ptycho._direct_windows["C10_nm"][1][0].series
    assert window.ready and window.center == 12.1
    inside = window.center + 0.6 * window.half_width
    phase = direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": inside}, tilt, depth, 3, policy)
    expected = exact({**coefs, "C10_nm": inside})
    assert float(cp.abs(phase - expected).max()) <= 2e-6 * float(cp.abs(expected).max())
    outside = window.center + 1.5 * window.half_width
    direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": outside}, tilt, depth, 3, policy)
    # the next tile on the grid, kept beside the first
    tiles = [tile.series for tile in ptycho._direct_windows["C10_nm"][1]]
    assert tiles[0] is not window and tiles[0].center == window.center + 2 * window.half_width and tiles[1] is window
    # moving C12 invalidates the C10 window, whose held inputs include C12
    direct_workflow.drag(ptycho, "C12_nm", {**coefs, "C10_nm": outside, "C12_nm": 3.2}, tilt, depth, 3, policy)
    assert list(ptycho._direct_windows) == ["C12_nm"] and not ptycho._direct_windows["C12_nm"][1][0].series.ready


def test_prepared_windows_serve_the_first_frame_of_a_grab(ptycho):
    """After a release, windows asked for at idle make the next grab start on window frames."""
    policy = direct_workflow.settings("complex", 0.03, 0.05, None, None)
    coefs, tilt, depth = dict(ABERRATIONS), OPTICS["sample_tilt_mrad"], OPTICS["depth_spread_nm"]
    ptycho._direct_windows = {}
    direct_workflow.prepare(ptycho, ("C10_nm", "phi12_deg"), coefs, tilt, depth, 3, policy)
    steps = 0
    while not direct_workflow.advance(ptycho):
        steps += 1
    assert steps == 2 * (chebyshev.ORDER + 1) - 1
    moved = {**coefs, "phi12_deg": coefs["phi12_deg"] + 0.5}
    assert ptycho._direct_windows["phi12_deg"][1][0].series.covers(moved["phi12_deg"])
    phase = direct_workflow.drag(ptycho, "phi12_deg", moved, tilt, depth, 3, policy)
    expected = ptycho.preview(moved, method="complex", phase_regularization=0.03, amplitude_regularization=0.05,
                              upsampling_factor=3, sample_tilt_mrad=tilt, depth_spread_nm=depth,
                              compute_loss=False, output="native")[0]
    assert float(cp.abs(phase - expected).max()) <= 2e-6 * float(cp.abs(expected).max())
    # the C10 window held phi12 at its old value
    assert list(ptycho._direct_windows) == ["phi12_deg"]


# ---


@pytest.mark.parametrize("control", ["C10_nm", "phi12_deg"])
def test_ssb_window_matches_exact_preview(ptycho, control):
    """SSB drags: single-sideband part interpolated, triple overlap exact, against the full 14-term preview.

    The two classes partition the terms, so at a node only the float64 order
    of the BF sum differs (tolerance 1e-6 of the largest phase); between nodes
    the truncation adds below 4e-7 (tolerance 2e-6).
    """
    coefs, tilt, depth = dict(ABERRATIONS), OPTICS["sample_tilt_mrad"], OPTICS["depth_spread_nm"]
    policy = direct_workflow.settings("ssb", None, None, None, None)

    def exact(aberrations):
        return ptycho.preview(aberrations, upsampling_factor=3, sample_tilt_mrad=tilt, depth_spread_nm=depth,
                              compute_loss=False, output="native")[0]

    ptycho._direct_windows = {}
    first = direct_workflow.drag(ptycho, control, coefs, tilt, depth, 3, policy)
    cp.testing.assert_array_equal(first, exact(coefs))
    while not direct_workflow.advance(ptycho):
        pass
    window = ptycho._direct_windows[control][1][0]
    series = window.series
    for value, relative in [(series.nodes[3], 1e-6)] + [(series.center + series.half_width * x, 2e-6) for x in (-1.0, 0.43, 1.0)]:
        phase = direct_workflow.drag(ptycho, control, {**coefs, control: value}, tilt, depth, 3, policy)
        expected = exact({**coefs, control: value})
        assert float(cp.abs(phase - expected).max()) <= relative * float(cp.abs(expected).max())
    assert ptycho._direct_windows[control][1][0] is window
