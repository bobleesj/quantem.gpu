"""The plain-torch SSB agreement (``direct_ptycho.torch_agreement``): float64 CPU ground truth for the fit's derivatives.

The scene is the weak-phase simulation of ``tests/hardware/thin_object`` (64 x 64 scan, known probe), built on the
CPU; no GPU is used. The kernels and searches are checked against this reference in
``tests/hardware/{cuda,mps}/test_ssb_exact_derivatives.py``.
"""

import math

import numpy as np
import pytest
import torch

from quantem.gpu.direct_ptycho.fifth_order_parameters import Coordinates
from quantem.gpu.direct_ptycho.torch_agreement import TorchAgreement, band_frequencies, fifth_from_low
from tests.hardware.thin_object import SIDE, STEP_A, TRUTH, simulate


@pytest.fixture(scope="module")
def scene():
    from quantem.gpu.optics.physics import wavelength_A_from_kV

    data, _, detector_mrad = simulate(TRUTH)
    wavelength = float(wavelength_A_from_kV(300.0))
    radius = 30.0 / detector_mrad
    grid = np.arange(SIDE)
    rows, cols = np.nonzero((grid[:, None] - SIDE / 2) ** 2 + (grid[None] - SIDE / 2) ** 2 <= radius ** 2)
    counts = np.moveaxis(data[:, :, rows, cols], 2, 0)

    def reference(rotation_deg=0.0):
        return TorchAgreement.from_counts(
            counts, pixel_rows=rows, pixel_cols=cols, center_px=(SIDE / 2, SIDE / 2), rotation_deg=rotation_deg,
            wavelength=wavelength, semiangle_rad=0.03, scan_sampling_A=STEP_A,
            detector_sampling_rad=(detector_mrad * 1e-3,) * 2, chunk=512)

    return reference


@pytest.fixture(scope="module")
def point():
    z = np.random.default_rng(3).uniform(-0.5, 0.5, 28)
    z[5] = 1.2
    return z


def test_chunked_chain_rule_gradient_equals_full_autograd(scene, point):
    reference = scene()
    value, gradient = reference.value_and_gradient(point)
    z = torch.tensor(point, requires_grad=True)
    total = reference.value(z)
    (full,) = torch.autograd.grad(total, z)
    assert value == float(total.detach())
    np.testing.assert_allclose(gradient, full.numpy(), rtol=0, atol=1e-12 * np.abs(gradient).max())


def test_gradient_matches_central_differences(scene, point):
    reference = scene()
    _, gradient = reference.value_and_gradient(point)
    step = 1e-5
    coordinates = [0, 2, 3, 5, 6, 13, 22, 27]       # C10, C12, tilt, depth and higher orders of every degree
    unit = np.eye(28)
    central = np.array([(reference.agreement(point + step * unit[i]) - reference.agreement(point - step * unit[i]))
                        / (2 * step) for i in coordinates])
    assert np.abs(central - gradient[coordinates]).max() < 1e-7 * np.abs(gradient).max()


def test_hessian_is_symmetric_and_matches_gradient_differences(scene, point):
    reference = scene()
    coordinates = [0, 1, 2, 3, 4, 5]
    hessian = reference.hessian(point, coordinates)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12 * np.abs(hessian).max())
    differences = reference.gradient_differences(point, step=1e-5, coordinates=coordinates)
    assert np.abs(hessian - differences).max() < 1e-7 * np.abs(hessian).max()


def test_opposite_branch_image_has_the_same_agreement(scene, point):
    """A 180-degree rotation negates k: even-degree chi terms (C10, C12, C30, C32, C34, C50, ...) and the tilt
    change sign, odd-degree terms (C21, C23, C41, C43, C45) and the depth spread are kept."""
    from quantem.gpu.direct_ptycho.higher_order import ORDERS

    mapped = point.copy()
    mapped[0:5] = -point[0:5]
    j = 6
    for n, m in ORDERS:
        if n < 2:
            continue
        width = 2 if m else 1
        if (n + 1) % 2 == 0:
            mapped[j:j + width] = -point[j:j + width]
        j += width
    first = scene(0.0).agreement(point)
    assert scene(180.0).agreement(mapped) == pytest.approx(first, rel=1e-12)


def test_packed_coefficients_are_the_shared_coordinates(scene, point):
    reference = scene()
    coordinates = Coordinates()
    coordinates.p = type("P", (), {"wavelength": reference.wavelength, "semiangle_rad": reference.semiangle})()
    _, packed = coordinates.encode(point[None], 5)
    np.testing.assert_allclose(reference.packed(torch.tensor(point)).numpy(), packed[0].ravel(), rtol=1e-6,
                               atol=1e-12)


def test_low_order_coordinates_and_subsample(scene):
    reference = scene()
    phi = math.radians(30.0)
    z6 = torch.tensor([-50.0, 20 * math.cos(2 * phi), 20 * math.sin(2 * phi), 1.5, -2.0, 150.0], dtype=torch.float64)
    z = fifth_from_low(z6)
    assert z[0] == -0.5 and z[3] == 0.3 and z[5] == 1.5 and torch.all(z[6:] == 0)
    half = reference.subsample(2)
    assert half.num_bright_field_pixels == (reference.num_bright_field_pixels + 1) // 2
    assert 0 < half.agreement(z) < reference.agreement(z)


def test_band_frequencies_follow_the_half_plane_rule():
    r, c, q_row, q_col, weight = band_frequencies((64, 64), 0.25)
    magnitude = np.hypot(q_row, q_col)
    assert np.all((magnitude > 0.2) & (magnitude < 0.9))
    assert np.all(c <= 32) and np.all(weight[c == 0] == 1) and np.all(weight[c > 0] == 2)


@pytest.mark.slow
def test_float64_search_recovers_the_probe(scene):
    """The exact-gradient search on the reference's own float64 values and gradients finds the simulated probe."""
    from quantem.gpu.direct_ptycho._gradient_refinement import Settings, refine
    from quantem.gpu.direct_ptycho.fifth_order_parameters import public_params

    reference = scene()
    coordinates = Coordinates()
    coordinates.p = type("P", (), {"wavelength": reference.wavelength, "semiangle_rad": reference.semiangle})()
    truth_point = np.zeros(28)
    phi = math.radians(TRUTH["phi12_deg"]) * 2
    truth_point[:6] = [TRUTH["C10_nm"] / 10, TRUTH["C12_nm"] / 10 * math.cos(phi),
                       TRUTH["C12_nm"] / 10 * math.sin(phi), 0, 0, 0.2]
    start = truth_point.copy()
    start[0] += 0.05
    norm = reference.agreement(start)
    strided = {1: reference, 4: reference.subsample(4), 8: reference.subsample(8)}

    def evaluate(points, kind, stride):
        values, gradients = [], []
        for z in np.atleast_2d(points):
            value, gradient = strided[stride].value_and_gradient(z)
            values.append(value / norm)
            gradients.append(gradient / norm)
        return np.array(values), (None if kind == "value" else np.array(gradients))

    result = refine(evaluate, start, np.array([(-3, 3), (-2, 2), (-2, 2), (-5, 5), (-5, 5), (0.2, 6)]
                                              + [(-2, 2)] * 22),
                    settings=Settings(hessian_bf_stride=8, coarse_bf_stride=4))
    assert result["stop"] == "converged"
    aberrations, _, _ = public_params(coordinates, result["coordinates"], 5)
    assert aberrations["C10_nm"] == pytest.approx(TRUTH["C10_nm"], abs=0.2)
    assert aberrations["C12_nm"] == pytest.approx(TRUTH["C12_nm"], abs=0.2)
    assert reference.agreement(result["coordinates"]) > reference.agreement(start)
