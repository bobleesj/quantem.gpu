"""Prepared regularized systems: agreement with the previous implementation, and one system per method.

The previous preparation formed T^H W T with complex128 batched matrix products
over chunks of 128 frequencies; it is kept here, unchanged, as the reference.
The new kernel differs in two float32 places, neither a summation-order effect
alone. It evaluates the same float32 transfer inside its reduction, where the
compiler contracts multiply-adds differently: identical source compiled in the
two kernel contexts differs by one float32 ulp in about 40 % of transfer
values. And it forms the products and per-lane sums (about 300 terms) in
float32, meeting in float64 only across lanes, because float64 runs at 1/64 of
the float32 rate. Gram, right side, eigenvalues and signal power therefore
agree to a few float32 ulps of their largest value: measured 3.1e-7, 3.7e-7,
3.0e-7 and 7e-9 on the 512 x 512 test field, at most 2.1e-7 here; tolerance
1e-6 (8 ulps), and 1e-7 for the power. Images are float32 and agree to a few
ulps of their largest value (tolerance 2e-6), log|w| to a few ulps at |w| = 1,
where a complex64 wave quantizes it (4 x 2**-23).
"""

import time

import cupy as cp
import numpy as np
import pytest

from quantem.gpu import DirectPtycho
from quantem.gpu.ssb.cuda.direct import PreparedDirect
from quantem.gpu.ssb.cuda.weak_object import transfer_frequencies
from quantem.gpu.ssb.units import higher_order_arrays

pytestmark = pytest.mark.slow

ABERRATIONS = {"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 40.0, "C21_nm": 150.0, "phi21_deg": -20.0, "C30_nm": 2000.0}
OPTICS = {"aberrations": ABERRATIONS, "sample_tilt_mrad": (1.5, -0.8), "depth_spread_nm": 8.0}


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


def test_prepared_systems_match_previous_implementation(ptycho):
    """Gram, right side, eigenvalues, power and both images against the complex128 GEMM reference."""
    engine = ptycho._backend()._get_accelerator()
    for complex_object in (False, True):
        for weighting in ("uniform", "inverse_mean"):
            options = dict(complex_object=complex_object, smoothness_inv_A=1.0, detector_weighting=weighting)
            new = PreparedDirect.prepare(engine, OPTICS, 3, **options)
            old = reference_prepare(engine, OPTICS, 3, **options)
            np.testing.assert_allclose(float(new.signal_power), float(old.signal_power), rtol=1e-7)
            assert [group.rows.shape for group in new.groups] == [group.rows.shape for group in old.groups]
            for group, reference in zip(new.groups, old.groups):
                cp.testing.assert_array_equal(group.rows, reference.rows)
                cp.testing.assert_array_equal(group.cols, reference.cols)
                cp.testing.assert_array_equal(group.prior, reference.prior)
                assert float(cp.abs(group.gram - reference.gram).max()) <= 1e-6 * float(cp.abs(reference.gram).max())
                assert float(cp.abs(group.rhs - reference.rhs).max()) <= 1e-6 * float(cp.abs(reference.rhs).max())
                if not complex_object:
                    eigen, expected = cp.sort(group.eigen, axis=1), cp.sort(reference.eigen, axis=1)
                    assert float(cp.abs(eigen - expected).max()) <= 1e-6 * float(cp.abs(expected).max())
            for strengths in ((0.03, 0.03), (0.01, 0.3)):
                wave, _ = new.solve(*strengths, compute_loss=False)
                expected, _ = old.solve(*strengths, compute_loss=False)
                parts = (cp.angle, lambda w: cp.log(cp.abs(w))) if complex_object else (cp.angle,)
                for part in parts:
                    image, image_expected = part(wave), part(expected)
                    # log|w| of a complex64 wave near |w| = 1 is quantized by the float32 ulp at 1
                    tolerance = max(2e-6 * float(cp.abs(image_expected).max()), 4 * 2.0**-23)
                    assert float(cp.abs(image - image_expected).max()) <= tolerance


def test_switching_methods_keeps_both_prepared(ptycho):
    """phase -> complex -> phase at one optics prepares each method once; the third call reuses the first."""
    first = ptycho.run(method="phase", upsampling_factor=3, compute_loss=False)
    ptycho.run(method="complex", upsampling_factor=3, compute_loss=False)
    assert set(ptycho._prepared_direct[1]) == {"phase", "complex"}
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    third = ptycho.run(method="phase", upsampling_factor=3, compute_loss=False)
    cp.cuda.Device().synchronize()
    assert time.perf_counter() - start < 0.1
    cp.testing.assert_array_equal(third.object_wave, first.object_wave)
    ptycho.run(method="phase", aberrations={**ABERRATIONS, "C10_nm": 13.0}, upsampling_factor=3, compute_loss=False)
    assert set(ptycho._prepared_direct[1]) == {"phase"}


# ---


class _Reference:
    def __init__(self, groups, complex_object, scale, signal_power, factor, shape, dose):
        self.groups, self.complex_object, self.scale, self.signal_power = groups, complex_object, scale, signal_power
        self.factor, self.shape, self.incident_scale = factor, shape, dose

    solve = PreparedDirect.solve


class _Group:
    def __init__(self, rows, cols, gram, rhs, prior, basis, eigen):
        self.rows, self.cols, self.gram, self.rhs, self.prior, self.basis, self.eigen = rows, cols, gram, rhs, prior, basis, eigen


def reference_prepare(engine, optics, factor, *, complex_object, smoothness_inv_A, detector_weighting):
    """The preparation before 2026-10-24, verbatim: chunks of 128 frequencies, complex128 batched GEMM, cusolver eigh."""
    nbf, nrow, ncol = engine._bf_grid()
    if min(nrow, ncol) < 3:
        raise ValueError(
            "Direct inversion needs at least 3 scan positions per side; select a larger region."
        )
    out_row, out_col = factor * nrow, factor * ncol
    cache = engine._cache
    aperture2 = cache["aperture_k_1d"] ** 2
    dose = cp.sum(engine.G_qk[:, 0, 0].real, dtype=cp.float64)
    dose /= nrow * ncol * aperture2.sum()
    if not bool(cp.isfinite(dose) & (dose > 0)):
        raise ValueError(
            "No positive incident intensity estimate; check counts and bright-field calibration."
        )
    if detector_weighting == "uniform":
        weights = cp.ones(nbf, cp.float32)
    else:
        mean = cp.maximum(engine.G_qk[:, 0, 0].real / (nrow * ncol), 0)
        floor = cp.median(mean) * 0.05
        if float(floor) <= 0:
            raise ValueError(
                "Variance weighting needs positive bright-field counts; use uniform weighting."
            )
        weights = 1 / cp.maximum(mean, floor)
        weights /= weights.mean()
    scale = cp.sum(weights * aperture2**2, dtype=cp.float64)
    rows, cols = cp.indices((nrow, ncol // 2 + 1), dtype=cp.int32)
    keep = (cols > 0) | ((rows > 0) & (rows <= (nrow - 1) // 2))
    if nrow % 2 == 0:
        keep &= rows != nrow // 2
    if ncol % 2 == 0:
        keep &= cols != ncol // 2
    rows, cols = rows[keep], cols[keep]
    alias_row, alias_col = cp.indices((factor, factor), dtype=cp.int32)
    output_rows = rows[:, None] + alias_row.ravel()[None, :] * nrow
    output_cols = cols[:, None] + alias_col.ravel()[None, :] * ncol
    qr = (
        cp.where(
            output_rows <= (out_row - 1) // 2, output_rows, output_rows - out_row
        ).astype(cp.float32)
        * cache["qx_1d"][1]
    )
    qc = (
        cp.where(
            output_cols <= (out_col - 1) // 2, output_cols, output_cols - out_col
        ).astype(cp.float32)
        * cache["qy_1d"][1]
    )
    cutoff = (
        2 * cache["semiangle_rad"] + max(cache["ang_y_rad"], cache["ang_x_rad"])
    ) / engine.wavelength
    active = qr * qr + qc * qc < cutoff**2
    counts = active.sum(axis=1)
    higher = higher_order_arrays(optics["aberrations"])
    if higher is None:
        magnitudes = np.zeros(14, np.float32)
        angles = np.zeros(14, np.float32)
        magnitudes[:2] = [
            optics["aberrations"].get("C10_nm", 0),
            optics["aberrations"].get("C12_nm", 0),
        ]
        angles[1] = np.deg2rad(optics["aberrations"].get("phi12_deg", 0))
    else:
        magnitudes, angles = higher
    groups = []
    power = cp.zeros((), cp.float64)
    for count in range(int(counts.max()) + 1):
        selected = cp.flatnonzero(counts == count)
        chunks = []
        for start in range(0, selected.size, 128):
            indices = selected[start : start + 128]
            measured = engine.G_qk[:, rows[indices], cols[indices]].T / dose
            power += cp.sum(
                cp.abs(measured) ** 2 * weights[None, :], dtype=cp.float64
            )
            if not count:
                continue
            group_index, alias_index = cp.nonzero(active[indices])
            group_index = indices[group_index]
            r = output_rows[group_index, alias_index].reshape(-1, count)
            c = output_cols[group_index, alias_index].reshape(-1, count)
            amplitude, phase = transfer_frequencies(
                engine,
                qr[group_index, alias_index].reshape(-1, count),
                qc[group_index, alias_index].reshape(-1, count),
                magnitudes * 10,
                angles,
                tuple(optics["sample_tilt_mrad"]),
                optics["depth_spread_nm"] * 10,
            )
            transfer = (
                cp.concatenate((amplitude, phase), axis=1)
                if complex_object
                else phase
            )
            transfer = transfer.astype(cp.complex128)
            weighted = transfer.conj() * weights[None, None, :]
            gram = weighted @ transfer.swapaxes(1, 2)
            rhs = (weighted @ measured[:, :, None].astype(cp.complex128))[..., 0]
            prior = cp.ones((indices.size, count), cp.float64)
            if smoothness_inv_A:
                frequency2 = (
                    qr[group_index, alias_index] ** 2
                    + qc[group_index, alias_index] ** 2
                ).reshape(-1, count)
                prior = (1 + frequency2 / smoothness_inv_A**2) ** 2
            if complex_object:
                prior = cp.concatenate((prior, prior), axis=1)
                basis = eigen = None
            else:
                inverse = 1 / cp.sqrt(prior.astype(cp.float64))
                eigen, vectors = cp.linalg.eigh(
                    gram * inverse[:, :, None] * inverse[:, None, :] / scale
                )
                projected = (
                    vectors.conj().swapaxes(1, 2)
                    @ (rhs * inverse / scale)[:, :, None]
                )[..., 0]
                basis = inverse[:, :, None] * vectors * projected[:, None, :]
            chunks.append((r, c, gram, rhs, prior, basis, eigen))
        if chunks:
            arrays = [
                None
                if chunks[0][i] is None
                else cp.ascontiguousarray(
                    cp.concatenate([chunk[i] for chunk in chunks])
                )
                for i in range(7)
            ]
            groups.append(_Group(*arrays))
    return _Reference(groups, complex_object, scale, power, factor, (out_row, out_col), float(dose))
