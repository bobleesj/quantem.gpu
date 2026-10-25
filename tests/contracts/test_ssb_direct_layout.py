"""The CPU side of the MPS regularized direct methods: frequency layout, detector weights and the float64 solve.

These run without a GPU: the alias packing and weights are NumPy, and the per-frequency Cholesky solve is the
compiled CPU code that every MPS ``method="phase"`` / ``"complex"`` reconstruction uses.
"""

import numpy as np
import pytest

from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.ssb.mps.direct import _alias_groups, _detector_weights, _solve_group


def _scene_groups(factor, smoothness=1.0):
    """Alias groups of a 128 x 128 scan at 0.4 Angstrom, 300 kV, 30 mrad, 7.5 mrad detector sampling."""
    cutoff = (2 * 30e-3 + 7.5e-3) / float(wavelength_A_from_kV(300.0))
    return _alias_groups((128, 128), (0.4, 0.4), factor, cutoff, smoothness)


@pytest.mark.parametrize("factor, histogram", [
    (1, [0, 8064]),
    (2, [0, 0, 0, 56, 8008]),
    (3, [0, 0, 0, 0, 1158, 676, 3870, 2360]),
    (4, [0, 0, 0, 0, 1158, 676, 3870, 2360]),
])
def test_alias_histogram(factor, histogram):
    """Every kept frequency's supported aliases, counted; beyond 3x the extra aliases all lie outside the cutoff."""
    groups = _scene_groups(factor)
    counts = np.zeros(len(histogram), int)
    for group in groups:
        counts[group.count] = group.frequency.shape[0]
    assert counts.tolist() == histogram


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_solved_set_and_mirrors(factor):
    """No DC replica or native Nyquist family is solved, and no alias or mirror is placed twice."""
    size = 128 * factor
    placed = []
    for group in _scene_groups(factor):
        rows, cols = np.divmod(group.positions.ravel(), size)
        assert not np.any((rows % 128 == 0) & (cols % 128 == 0))
        assert not np.any((rows % 128 == 64) | (cols % 128 == 64))
        mirrors = ((size - rows) % size) * size + (size - cols) % size
        placed += [group.positions.ravel(), mirrors]
        if group.count:
            source = group.frequency.astype(np.int64)
            np.testing.assert_array_equal(rows.reshape(-1, group.count) % 128, source[:, :1].repeat(group.count, 1))
            np.testing.assert_array_equal(cols.reshape(-1, group.count) % 128, source[:, 1:].repeat(group.count, 1))
        q2 = (group.alias.astype(np.float64) ** 2).sum(axis=-1)
        np.testing.assert_allclose(group.prior, (1 + q2) ** 2, rtol=1e-6)
    placed = np.concatenate(placed)
    assert np.unique(placed).size == placed.size


def test_flat_prior():
    assert all(np.array_equal(group.prior, np.ones_like(group.prior)) for group in _scene_groups(3, 0.0))


def test_inverse_mean_weights():
    """1 / max(mean, 5 % of the median mean), normalized to mean 1; uniform weights are ones."""
    sums = np.array([0, 1, 2, 4, 8, 16], np.float64) * 100
    weights = _detector_weights(sums, 100, "inverse_mean")
    np.testing.assert_allclose(weights, [4.6489, 0.69734, 0.34867, 0.17433, 0.087167, 0.043584], rtol=2e-4)
    assert weights.mean() == pytest.approx(1, rel=1e-6)
    assert np.array_equal(_detector_weights(sums, 100, "uniform"), np.ones(6, np.float32))
    with pytest.raises(ValueError, match="positive bright-field counts"):
        _detector_weights(np.array([0, 0, 0, 5.0]), 100, "inverse_mean")


def _packed(matrices, rhs):
    """Augmented Hermitian (Gram | rhs) upper triangles in the GPU's float-float layout (hi re, hi im, lo re, lo im)."""
    size, dim = rhs.shape
    width = dim + 1
    augmented = np.zeros((size, width, width), complex)
    augmented[:, :dim, :dim] = matrices
    augmented[:, :dim, dim] = rhs
    rows, cols = np.triu_indices(width)
    values = augmented[:, rows, cols]
    hi = values.astype(np.complex64)
    lo = (values - hi).astype(np.complex64)
    return np.ascontiguousarray(np.stack((hi.real, hi.imag, lo.real, lo.imag), axis=-1).astype(np.float32))


@pytest.mark.parametrize("complex_object", [False, True])
def test_float64_solve_and_scatter(complex_object):
    """Per-frequency (Gram + diag(strength * prior)) x = rhs / dose, coefficients scattered with factor^2 and mirrors."""
    rng = np.random.default_rng(3)
    size, count, side, factor, dose, scale = 40, 3, 24, 3, 2.5, 7.0
    dim = 2 * count if complex_object else count
    basis = rng.normal(size=(size, dim, 2 * dim)) + 1j * rng.normal(size=(size, dim, 2 * dim))
    gram = basis @ basis.conj().transpose(0, 2, 1)
    rhs = rng.normal(size=(size, dim)) + 1j * rng.normal(size=(size, dim))
    prior = 1 + rng.random((size, count))
    # rows 1 .. side/2 - 1: their mirrors lie in the other half, so no placement collides with a mirror
    positions = rng.choice(np.arange(side, side * (side // 2)), size=(size, count), replace=False)
    phase_strength, amplitude_strength = 0.03 * scale, 0.2 * scale
    spectrum = np.zeros((side * side, 2), np.float32)
    losses = np.zeros(size)
    sums = _packed(gram, rhs)
    failed = _solve_group(sums, dim + 1, count, dim, complex_object, prior, amplitude_strength, phase_strength,
                          dose, positions, side, side, np.float32(factor**2), spectrum, True, losses)
    assert failed == 0
    exact = sums[..., 0].astype(np.float64) + sums[..., 2] + 1j * (sums[..., 1].astype(np.float64) + sums[..., 3])
    width = dim + 1
    stored = np.zeros((size, width, width), complex)
    rows, cols = np.triu_indices(width)
    stored[:, rows, cols] = exact
    gram = np.triu(stored[:, :dim, :dim], 1)
    gram = gram + gram.conj().transpose(0, 2, 1) + np.eye(dim) * stored[:, np.arange(dim), np.arange(dim)].real[:, None]
    rhs = stored[:, :dim, dim] / dose
    strengths = np.concatenate((np.full(count, amplitude_strength), np.full(count, phase_strength))) if complex_object \
        else np.full(count, phase_strength)
    penalty = np.tile(prior, 2 if complex_object else 1) * strengths
    x = np.linalg.solve(gram + penalty[:, :, None] * np.eye(dim), rhs[..., None])[..., 0]
    rounded = x.astype(np.complex64).astype(complex)
    expected_loss = (np.einsum("fi,fij,fj->f", rounded.conj(), gram, rounded).real
                     - 2 * np.einsum("fi,fi->f", rounded.conj(), rhs).real)
    np.testing.assert_allclose(losses, expected_loss, rtol=1e-9, atol=1e-9 * np.abs(expected_loss).max())
    packed = spectrum.view(np.complex64).ravel()
    phi = x[:, -count:].astype(np.complex64) * np.float32(factor**2)
    eta = x[:, :count].astype(np.complex64) * np.float32(factor**2) if complex_object else np.zeros_like(phi)
    np.testing.assert_allclose(packed[positions], eta + 1j * phi, rtol=1e-6)
    r, c = np.divmod(positions, side)
    mirrors = ((side - r) % side) * side + (side - c) % side
    np.testing.assert_allclose(packed[mirrors], eta.conj() + 1j * phi.conj(), rtol=1e-6)


def test_solve_reports_indefinite_systems():
    """A system that is not positive definite is reported, not scattered."""
    sums = _packed(-np.eye(2)[None].astype(complex), np.ones((1, 2), complex))
    spectrum = np.zeros((16, 2), np.float32)
    failed = _solve_group(sums, 3, 2, 2, False, np.ones((1, 2)), 0.0, 1e-3, 1.0, np.array([[1, 2]]), 4, 4,
                          np.float32(1), spectrum, False, np.zeros(0))
    assert failed == 1 and not spectrum.any()
