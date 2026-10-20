"""Output sampling is an aperture-support choice, independent of fitting."""

import numpy as np
import pytest

from quantem.gpu.optics.physics import ssb_upsampling_factor, wavelength_A_from_kV


@pytest.mark.parametrize("required,expected", [(0.5, 1), (1, 1), (1.1, 2), (2.1, 3), (3.1, 4), (4.1, 8), (8, 8)])
def test_smallest_grid_covers_both_scan_axes(required, expected):
    spacing = wavelength_A_from_kV(300) / (4 * 20e-3)
    step = np.array([0.5, required]) * spacing
    factor = ssb_upsampling_factor(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=step)
    assert factor == expected
    cutoff = 2 * 20e-3 / wavelength_A_from_kV(300)
    assert np.all(factor / (2 * step) >= cutoff - 1e-12)
    smaller = [value for value in (1, 2, 3, 4, 8) if value < factor]
    if smaller:
        assert np.any(max(smaller) / (2 * step) < cutoff)


def test_float32_voltage_gives_the_double_precision_wavelength():
    """A voltage read from float32 metadata gives the same wavelength as the Python number."""
    assert wavelength_A_from_kV(np.float32(250)) == wavelength_A_from_kV(250) == 0.021986349996165905


@pytest.mark.parametrize("step", [0.37, 0, -1, float("nan"), float("inf"), [], [1, 2, 3], None])
def test_missing_or_invalid_calibration_is_not_guessed(step):
    with pytest.raises(ValueError):
        ssb_upsampling_factor(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=step)


def test_numpy_values_and_limit():
    assert ssb_upsampling_factor(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(np.float32(0.37),) * 2) == 2
    with pytest.raises(ValueError, match="more than 8"):
        ssb_upsampling_factor(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(10, 10))


@pytest.mark.parametrize("voltage,angle", [(0, 20), (300, 0), (float("nan"), 20), (300, float("inf"))])
def test_invalid_voltage_or_aperture(voltage, angle):
    with pytest.raises(ValueError):
        ssb_upsampling_factor(voltage_kV=voltage, semiangle_mrad=angle, scan_sampling_A=(0.37, 0.37))


@pytest.mark.parametrize("voltage_kV", [60, 80, 120, 200, 300, 400, 123.4])
def test_interaction_parameter_is_two_pi_m_e_lambda_over_h_squared(voltage_kV):
    """sigma = 2 pi gamma m_e e lambda / h^2 with the same relativistic wavelength, in rad / (V Angstrom)."""
    from quantem.gpu.optics import interaction_parameter
    from quantem.gpu.optics.physics import ELECTRON_CHARGE, ELECTRON_MASS, PLANCK_H, SPEED_OF_LIGHT

    gamma = 1 + ELECTRON_CHARGE * voltage_kV * 1e3 / (ELECTRON_MASS * SPEED_OF_LIGHT**2)
    expected = (2 * np.pi * gamma * ELECTRON_MASS * ELECTRON_CHARGE * wavelength_A_from_kV(voltage_kV) * 1e-10
                / PLANCK_H**2 * 1e-10)
    assert interaction_parameter(voltage_kV) == pytest.approx(expected, rel=1e-12)


def test_interaction_parameter_at_300_kV_and_invalid_voltages():
    from quantem.gpu.optics import interaction_parameter

    assert interaction_parameter(300) == pytest.approx(6.526161e-4, rel=1e-6)
    for voltage_kV in (0, -300, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="voltage_kV"):
            interaction_parameter(voltage_kV)
