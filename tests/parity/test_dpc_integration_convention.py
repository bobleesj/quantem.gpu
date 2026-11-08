"""Check physical operator sign and gain independently of image appearance."""
import numpy as np
from quantem.gpu.dpc import integrate


def test_known_gradient_recovers_positive_scalar_with_unit_gain():
    row, col = np.mgrid[:64, :80]
    expected = np.cos(2*np.pi*row/64) + .5*np.cos(4*np.pi*col/80)
    grad_row = -2*np.pi/64*np.sin(2*np.pi*row/64)
    grad_col = -2*np.pi/80*np.sin(4*np.pi*col/80)
    result = integrate(grad_row, grad_col)
    np.testing.assert_allclose(result, expected, atol=5e-7)
    np.testing.assert_allclose(integrate(-grad_row, -grad_col), -expected, atol=5e-7)


def test_constant_and_transverse_fields_do_not_invent_scalar_contrast():
    row, col = np.mgrid[:64, :64]
    np.testing.assert_allclose(integrate(np.ones((64, 64)), np.ones((64, 64))), 0, atol=1e-7)
    # This field has only divergence-free transverse Fourier components.
    transverse_row = np.cos(2*np.pi*col/64)
    transverse_col = np.sin(2*np.pi*row/64)
    np.testing.assert_allclose(integrate(transverse_row, transverse_col), 0, atol=1e-7)


def test_screening_preserves_sign_for_forced_rotation_and_exchanged_axes():
    from quantem.gpu.screening.workflow import _dpc_phase
    row, col = np.mgrid[:32, :64]
    phase = np.cos(2*np.pi*row/32) + np.sin(2*np.pi*col/64)
    grad_row = -2*np.pi/32*np.sin(2*np.pi*row/32)
    grad_col = 2*np.pi/64*np.cos(2*np.pi*col/64)
    for angle in (0, 37, 170, 350):
        radians = np.deg2rad(angle)
        detector_row = np.cos(radians)*grad_row + np.sin(radians)*grad_col
        detector_col = -np.sin(radians)*grad_row + np.cos(radians)*grad_col
        for flipped in (False, True):
            source = (detector_col, detector_row) if flipped else (detector_row, detector_col)
            result = _dpc_phase(*source, angle, flipped)
            np.testing.assert_allclose(result, phase, atol=1e-6)
