"""MPS drag windows: the Chebyshev series along one control against exact preparation, as the CUDA windows are tested.

At a Chebyshev node the window reproduces that node's own statistics up to their float-float to float64 rounding and
the series' float64 arithmetic, so the images differ by a few float32 ulps of the largest pixel (tolerance 1e-6).
Between nodes the window adds its truncation (below 1e-10 of a Gram entry's terms) and the interpolated float32
rounding of the node preparations, amplified by at most the Lebesgue constant L = 1 + (2 / pi) ln(N + 1) of
Chebyshev interpolation (3.1 for a magnitude, 3.5 for an azimuth), to the float32 rounding of the exact preparation.
CUDA's tolerance there is 2e-6 of the largest phase; this 39-pixel scene's phase is only 0.02 rad, so its float32 floor
sits higher, and the tolerance is the larger of 2e-6 and (L + 1) times the exact image's own error against the
float64 model (``test_ssb_direct_mps``). On measured 512 x 512 scans the previews agree with the exact image within
about 1e-6 of its largest phase. log|w| of a complex64 wave near |w| = 1 is quantized at 2**-23.
"""

import numpy as np
import pytest

from quantem.gpu.direct_ptycho import direct_workflow

from .test_ssb_direct_mps import POINTS, _reference_prepare, _reference_solve, _scene, _session
from .test_ssb_thick_sample_mps import _require_mps

pytestmark = pytest.mark.slow

OPTIONS = dict(smoothness_inv_A=1.0, detector_weighting="uniform")
# the "default" preset of test_ssb_direct_mps, which its float64 reference solves
STRENGTH = 0.03


@pytest.fixture(scope="module")
def ptycho():
    _require_mps()
    session = _session(_scene(), "fifth")
    yield session
    session.close()


def _optics(ptycho, aberrations):
    return {"aberrations": aberrations, "sample_tilt_mrad": ptycho.sample_tilt_mrad,
            "depth_spread_nm": ptycho.depth_spread_nm}


def _backend(ptycho):
    policy = direct_workflow.settings("phase", STRENGTH, None, None, None)
    return direct_workflow._inputs(ptycho, dict(ptycho.aberrations), ptycho.sample_tilt_mrad, ptycho.depth_spread_nm,
                                   3, policy)[0]


def _parts(complex_object):
    """(image of a wave, its floor): the phase, and for the complex method log|w|."""
    return [(np.angle, 0.0)] + ([(lambda w: np.log(np.abs(w)), 4 * 2.0**-23)] if complex_object else [])


def _check(wave, expected, complex_object, relative, reference=None, order=None):
    """Every part of ``wave`` within ``relative`` of the largest exact value, or within (L + 1) times the exact
    image's error against the float64 ``reference``, L the Lebesgue constant of an order-``order`` series."""
    for index, (part, floor) in enumerate(_parts(complex_object)):
        exact = part(expected)
        tolerance = max(relative * float(np.abs(exact).max()), floor)
        if reference is not None:
            model = reference[0] if index == 0 else np.log(reference[1])
            lebesgue = 1 + 2 / np.pi * np.log(order + 1)
            tolerance = max(tolerance, (lebesgue + 1) * float(np.abs(exact - model).max()))
        difference = float(np.abs(part(wave) - exact).max())
        assert difference <= tolerance, (index, difference, tolerance)


@pytest.mark.parametrize("control", ["C10_nm", "phi12_deg", "C30_nm"])
def test_window_matches_exact_preparation(ptycho, control):
    """Nodes, window edges and a point between nodes, both methods, and phase from a complex window."""
    from quantem.gpu.direct_ptycho.mps.direct import DragWindow, PreparedDirect

    backend = _backend(ptycho)
    data = _scene()
    aberrations0 = dict(POINTS["fifth"][0])
    references = {}
    for complex_object in (False, True):
        window = DragWindow(backend, _optics(ptycho, aberrations0), control, 3, complex_object=complex_object,
                            **OPTIONS)
        while not window.step():
            pass
        series = window.series
        amplitude = STRENGTH if complex_object else None
        for value, relative in [(series.nodes[3], 1e-6)] + [(series.center + series.half_width * x, 2e-6)
                                                            for x in (-1.0, -0.43, 1.0)]:
            aberrations = {**aberrations0, control: value}
            optics = _optics(ptycho, aberrations)
            if relative > 1e-6 and value not in references:
                references[value] = _reference_prepare(data, "fifth", 3, "uniform", aberrations=aberrations)
            exact = PreparedDirect.prepare(backend, optics, 3, complex_object=complex_object, **OPTIONS)
            wave, _ = window.solve(value, STRENGTH, amplitude, complex_object=complex_object)
            expected, _ = exact.solve(STRENGTH, amplitude, compute_loss=False)
            model = _reference_solve(references[value], "complex" if complex_object else "phase", "default") \
                if relative > 1e-6 else None
            _check(wave, expected, complex_object, relative, model, series.order)
            if complex_object:
                phase_only = PreparedDirect.prepare(backend, optics, 3, complex_object=False, **OPTIONS)
                wave, _ = window.solve(value, STRENGTH, None, complex_object=False)
                expected, _ = phase_only.solve(STRENGTH, None, compute_loss=False)
                model = _reference_solve(references[value], "phase", "default") if relative > 1e-6 else None
                _check(wave, expected, False, relative, model, series.order)


def _exact_phase(ptycho, aberrations, method, amplitude=None):
    options = {} if amplitude is None else {"amplitude_damping": amplitude}
    return ptycho.preview(aberrations, method=method, phase_damping=STRENGTH, upsampling=3,
                          sample_tilt_mrad=ptycho.sample_tilt_mrad, depth_spread_nm=ptycho.depth_spread_nm,
                          compute_loss=False, output="numpy", **options)[0]


def test_drag_builds_between_frames_and_rebuilds_outside(ptycho):
    """Exact frames until the window is complete, window frames inside it, a new window outside it."""
    policy = direct_workflow.settings("phase", STRENGTH, None, None, None)
    coefs, tilt, depth = dict(POINTS["fifth"][0]), ptycho.sample_tilt_mrad, ptycho.depth_spread_nm
    ptycho._direct_windows = {}
    moved = coefs["C10_nm"] + 0.1
    first = direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": moved}, tilt, depth, 3, policy)
    np.testing.assert_array_equal(first, _exact_phase(ptycho, {**coefs, "C10_nm": moved}, "phase"))
    while not direct_workflow.advance(ptycho):
        pass
    tile = ptycho._direct_windows["C10_nm"][1][0]
    window = tile.series
    assert window.ready and window.center == moved
    inside = window.center + 0.6 * window.half_width
    phase = direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": inside}, tilt, depth, 3, policy)
    # the frame is the window's preview, not an exact preparation
    np.testing.assert_array_equal(phase, np.angle(tile.solve(inside, STRENGTH, None, complex_object=False)[0]))
    reference = _reference_solve(_reference_prepare(_scene(), "fifth", 3, "uniform",
                                                    aberrations={**coefs, "C10_nm": inside}), "phase", "default")
    _check(np.exp(1j * phase), np.exp(1j * _exact_phase(ptycho, {**coefs, "C10_nm": inside}, "phase")), False, 2e-6,
           reference, window.order)
    outside = window.center + 1.5 * window.half_width
    direct_workflow.drag(ptycho, "C10_nm", {**coefs, "C10_nm": outside}, tilt, depth, 3, policy)
    # the next tile on the grid, kept beside the first
    tiles = [tile.series for tile in ptycho._direct_windows["C10_nm"][1]]
    assert tiles[0] is not window and tiles[0].center == window.center + 2 * window.half_width and tiles[1] is window
    # moving C12 invalidates the C10 window, whose held inputs include C12
    direct_workflow.drag(ptycho, "C12_nm", {**coefs, "C10_nm": outside, "C12_nm": coefs["C12_nm"] + 0.2}, tilt, depth,
                         3, policy)
    assert list(ptycho._direct_windows) == ["C12_nm"] and not ptycho._direct_windows["C12_nm"][1][0].series.ready


def test_prepared_windows_serve_the_first_frame_of_a_grab(ptycho):
    """After a release, windows asked for at idle make the next grab start on window frames."""
    policy = direct_workflow.settings("complex", STRENGTH, STRENGTH, None, None)
    coefs, tilt, depth = dict(POINTS["fifth"][0]), ptycho.sample_tilt_mrad, ptycho.depth_spread_nm
    ptycho._direct_windows = {}
    direct_workflow.prepare(ptycho, ("C10_nm", "phi12_deg"), coefs, tilt, depth, 3, policy)
    steps = 0
    while not direct_workflow.advance(ptycho):
        steps += 1
    assert steps == sum(tiles[0].series.order + 1 for _, tiles, _ in ptycho._direct_windows.values()) - 1
    moved = {**coefs, "phi12_deg": coefs["phi12_deg"] + 0.5}
    tile = ptycho._direct_windows["phi12_deg"][1][0]
    assert tile.series.covers(moved["phi12_deg"])
    phase = direct_workflow.drag(ptycho, "phi12_deg", moved, tilt, depth, 3, policy)
    np.testing.assert_array_equal(
        phase, np.angle(tile.solve(moved["phi12_deg"], STRENGTH, STRENGTH, complex_object=True)[0]))
    reference = _reference_solve(_reference_prepare(_scene(), "fifth", 3, "uniform", aberrations=moved), "complex",
                                 "default")
    _check(np.exp(1j * phase), np.exp(1j * _exact_phase(ptycho, moved, "complex", STRENGTH)), False, 2e-6, reference,
           tile.series.order)
    # the C10 window held phi12 at its old value
    assert list(ptycho._direct_windows) == ["phi12_deg"]


def test_close_drops_drag_windows():
    """A closed session keeps no window, each of which holds its backend and its statistics."""
    _require_mps()
    session = _session(_scene(), "primary")
    session._direct_windows = {"C10_nm": object()}
    session.close()
    assert session._direct_windows == {}
