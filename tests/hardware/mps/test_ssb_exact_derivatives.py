"""MPS exact derivatives and fits against the float64 torch reference (``tests/hardware/exact_derivative_checks``).

The 64 x 64 weak-phase simulation of ``thin_object``.
"""

import numpy as np
import pytest

from tests.hardware import exact_derivative_checks as checks
from tests.hardware.thin_object import TRUTH, backend_or_skip, session, simulate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def simulation():
    if backend_or_skip() != "mps":
        pytest.skip("MPS only")
    return simulate(TRUTH)


@pytest.fixture(scope="module")
def ssb(simulation):
    data, _, detector_mrad = simulation
    with session(data, detector_mrad, "mps") as opened:
        yield opened


@pytest.mark.parametrize("stride", [1, 4])
def test_fifth_order_gradient_matches_the_reference(ssb, stride):
    checks.check_fifth_order_gradient(ssb, stride)


def test_gradient_pass_needs_its_value_pass(ssb):
    objective = checks.fifth_objective(ssb)
    z = np.zeros(28)
    z[5] = 1.0
    objective.values(z, 1)
    with pytest.raises(RuntimeError, match="same points"):
        objective.gradients(z, 4)


def test_ring_halves_partition_the_band(ssb):
    checks.check_band_halves(ssb)


def test_low_order_gradient_and_hessian_match_the_reference(ssb):
    checks.check_low_order_derivatives(ssb)


def test_opposite_branch_point_has_the_same_agreement(simulation):
    checks.check_opposite_branch(simulation, "mps")


def test_exact_fit_reaches_the_float64_optimum(simulation):
    checks.check_fit(simulation, "mps")


def test_default_wide_fit_is_checked_on_two_band_halves(simulation):
    checks.check_wide_fit(simulation, "mps")


@pytest.mark.parametrize("depth", [150.0, -150.0, 0.0])
def test_low_order_derivatives_match_central_differences(ssb, depth):
    """The low-order gradient against central differences of the value pass, and the Hessian against central
    differences of the gradient, with tilt, at a positive, a negative (the kernels see |z5|) and a zero depth spread;
    the fused pass gives the two-stage pass's derivatives."""
    import math

    low = checks.low_order(ssb)
    phi = math.radians(37.0) * 2
    z = np.array([-42.0, 25 * math.cos(phi), 25 * math.sin(phi), 1.5, -2.0, depth])
    step = np.array([0.5, 0.5, 0.5, 0.05, 0.05, 2.0])
    low.values(z[None])
    _, gradient, hessian = low.derivatives([0])[0]
    rows = np.concatenate([z + np.diag(step), z - np.diag(step)])
    values = low.values(rows, True)
    np.testing.assert_allclose(hessian, hessian.T)
    held = 5 if depth == 0.0 else 6   # |z5| has no derivative at 0
    differences = (values[:6] - values[6:]) / (2 * step)
    assert checks.relative(gradient[:held], differences[:held]) < 2e-3
    fused = low.derivatives(range(12))
    columns = np.array([(fused[i][1] - fused[i + 6][1]) / (2 * step[i]) for i in range(6)])
    assert checks.relative(hessian[:held, :held], columns[:held, :held]) < 2e-3
    low.values(z[None], True)
    _, fused_gradient, fused_hessian = low.derivatives([0])[0]
    assert checks.relative(fused_gradient, gradient) < 1e-5
    assert checks.relative(fused_hessian, hessian) < 1e-5


@pytest.mark.parametrize("anisotropy", [1.0, 1.07])
def test_fit_aperture_equals_the_soft_aperture_everywhere(ssb, anisotropy):
    """The low-order kernels' aperture (exactly 1 inside and 0 outside ``aperture_radii``, the edge expression in the
    ring) equals ``tf_geometry``'s soft aperture at q - k and q + k of every band frequency and BF pixel, also for
    anisotropic detector sampling: bit for bit wherever either is 0 or 1, and in the edge ring within the float32
    rounding of the quotient (``fit_quotient`` against the division of ``tf_geometry``)."""
    import dataclasses
    import math

    from quantem.gpu.direct_ptycho.mps import resident_fit
    from quantem.gpu.direct_ptycho.mps.thick_sample import (
        _THICK_FIT_BATCH_HEADER,
        _band,
    )

    backend = ssb._backend()
    backend.cache_rotation(math.radians(ssb.scan_detector_rotation_deg))
    prepared = dataclasses.replace(backend._prepared, ang_x_rad=backend._prepared.ang_x_rad * anisotropy,
                                   fit_radii=None)
    mx = prepared.mx
    _, qx, qy, _ = _band(prepared, (0.2, 0.9))
    n_band, nk = int(qx.size), int(prepared.kx.size)
    kernel = mx.fast.metal_kernel(
        name="ssb_fit_aperture_check", input_names=["qx", "qy", "kx", "ky", "scalars", "radii"],
        output_names=["fast", "soft"], header=_THICK_FIT_BATCH_HEADER + resident_fit.FIT_HEADER,
        source="""
        uint q = thread_position_in_grid.x, k = thread_position_in_grid.y;
        if (q >= N_BAND) return;
        float wl = scalars[0], semi = scalars[1], ay = scalars[2], ax = scalars[3];
        float2 e = fit_aperture(qx[q], qy[q], kx[k], ky[k], wl, semi, ay, ax, radii[0], radii[1]);
        size_t o = ((size_t)k * N_BAND + q) * 2;
        fast[o] = e.x;
        fast[o + 1] = e.y;
        soft[o] = tf_geometry(qx[q] - kx[k], qy[q] - ky[k], wl, semi, ay, ax).w;
        soft[o + 1] = tf_geometry(qx[q] + kx[k], qy[q] + ky[k], wl, semi, ay, ax).w;
        """)
    scalars = mx.array([prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad, prepared.ang_x_rad],
                       dtype=mx.float32)
    fast, soft = kernel(inputs=[qx, qy, prepared.kx, prepared.ky, scalars, resident_fit.aperture_radii(prepared)],
                        template=[("N_BAND", n_band)], grid=(n_band, nk, 1), threadgroup=(64, 1, 1),
                        output_shapes=[(nk * n_band * 2,)] * 2, output_dtypes=[mx.float32] * 2)
    fast, soft = np.asarray(fast), np.asarray(soft)
    edge = (soft > 0) & (soft < 1)
    assert edge.any() and (soft == 1).any() and (soft == 0).any()
    flat = (soft == 0) | (soft == 1) | (fast == 0) | (fast == 1)
    np.testing.assert_array_equal(fast[flat], soft[flat])
    np.testing.assert_allclose(fast[edge], soft[edge], rtol=0, atol=2 * np.finfo(np.float32).eps)
