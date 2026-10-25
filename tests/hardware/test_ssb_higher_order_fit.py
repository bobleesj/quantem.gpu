"""find_aberrations(higher_order=...) on the session backend (MPS or CUDA): the agreement kernel, the fit, the guard.

The data are the weak-phase simulation of ``thin_object`` (one slice, a probe with known coma, trefoil and spherical
aberration): the regime where the SSB model is exact, so the fit must return the probe's coefficients and the guard
must trust them. CUDA reads the scan tiled to 128 x 128, which it reconstructs without padding.
"""

import math

import numpy as np
import pytest

from tests.hardware.thin_object import (
    ORDERS,
    TRUTH,
    backend_or_skip,
    chi,
    session,
    simulate,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def simulation():
    backend = backend_or_skip()
    return (backend, *simulate(TRUTH, tile=2 if backend == "cuda" else 1))


def _rows(aberrations, depth_A=0.0, tilt_mrad=(0.0, 0.0)):
    """Agreement rows: (depth, tilt row, tilt col) and the Cartesian Krivanek coefficients C10 ... C30 (A)."""
    row = [depth_A, *tilt_mrad]
    for name, (_, m) in ORDERS.items():
        magnitude = 10.0 * aberrations.get(f"{name}_nm", 0.0)
        angle = math.radians(aberrations.get(f"phi{name[1:]}_deg", 0.0))
        row += [magnitude * math.cos(m * angle), magnitude * math.sin(m * angle)]
    return np.array([row])


def _evidence(ssb):
    """The backend's G(q, k) on the full q plane, k, q and the probe geometry, in float64 on the host."""
    backend = ssb._backend()
    if ssb.backend == "mps":
        backend.cache_rotation(0.0)
        p = backend._prepared
        g = np.asarray(p.g_qk).astype(np.complex128)
        kr, kc = np.asarray(p.kx, np.float64), np.asarray(p.ky, np.float64)
        q_row, q_col = np.asarray(p.q_row, np.float64).ravel(), np.asarray(p.q_col, np.float64).ravel()
        geometry = p.wavelength, p.semiangle_rad, p.ang_y_rad, p.ang_x_rad
    else:
        engine = backend._get_accelerator()
        cache = engine._cache
        g = engine.G_qk.get().astype(np.complex128)
        kr, kc = cache["kx_bf"].get().astype(np.float64), cache["ky_bf"].get().astype(np.float64)
        q_row, q_col = cache["qx_1d"].get().astype(np.float64), cache["qy_1d"].get().astype(np.float64)
        geometry = engine.wavelength, cache["semiangle_rad"], cache["ang_y_rad"], cache["ang_x_rad"]
    native = len(q_col)
    if g.shape[-1] != native:
        mirror = (-np.arange(len(q_row))) % len(q_row)
        cols = (native - np.arange(native // 2 + 1, native)) % native
        g = np.concatenate((g, g[:, mirror][:, :, cols].conj()), axis=-1)
    return g, kr[:, None], kc[:, None], q_row, q_col, geometry


def test_agreement_reduces_to_the_two_term_kernel_and_matches_the_physical_model(simulation):
    """Zero higher orders: the joint fit's kernel. With coma, trefoil and C30: a float64 sum over the full q plane."""
    backend_name, data, _, detector_mrad = simulation
    with session(data, detector_mrad, backend_name) as ssb:
        backend = ssb._backend()
        low = {"C10_nm": -4.0, "C12_nm": 1.5, "phi12_deg": 25.0}
        two_term = backend.sample_agreement(np.array([[-40.0, 15.0, math.radians(25.0), 2.0, -3.0, 120.0]]), (0.2, 0.9))
        np.testing.assert_allclose(backend.higher_order_agreement(_rows(low, 120.0, (2.0, -3.0))), two_term, rtol=3e-6)

        # independent float64 model on every q of the band (not the half plane the kernels sum with weight 2)
        g, kr, kc, q_row, q_col, (wavelength, semiangle, ang_y, ang_x) = _evidence(ssb)
        band = (np.hypot(q_row[:, None], q_col[None]) > 0.2) & (np.hypot(q_row[:, None], q_col[None]) < 0.9)
        qr, qc = np.broadcast_to(q_row[:, None], band.shape)[band], np.broadcast_to(q_col[None], band.shape)[band]

        def probe(row, col):
            r = np.hypot(row, col)
            with np.errstate(divide="ignore", invalid="ignore"):
                edge = np.hypot(row * ang_y, col * ang_x) / np.where(r > 0, r, 1.0)
                aperture = np.clip(np.where(r > 0, (semiangle - r * wavelength) / edge + 0.5, 1.0), 0.0, 1.0)
            return aperture * np.exp(-1j * chi(row, col, wavelength, TRUTH))

        gamma = probe(kr - qr, kc - qc) * probe(kr, kc).conj() - probe(kr, kc) * probe(kr + qr, kc + qc).conj()
        source = g.reshape(len(g), -1)[:, band.ravel()]
        numerator = np.abs((source * gamma.conj()).sum(0)) ** 2
        denominator = (np.abs(gamma) ** 2).sum(0)
        expected = float((numerator[denominator > 0] / denominator[denominator > 0]).sum())
        np.testing.assert_allclose(backend.higher_order_agreement(_rows(TRUTH)), expected, rtol=1e-5)


@pytest.mark.parametrize("tilt", [True, False])
@pytest.mark.parametrize("compute_loss", [True, False])
def test_fit_recovers_coma_trefoil_and_spherical_aberration(simulation, tilt, compute_loss):
    backend_name, data, _, detector_mrad = simulation
    with session(data, detector_mrad, backend_name) as ssb:
        fit = ssb.find_aberrations(search="primary", tilt=tilt, higher_order=("C21", "C23", "C30"), verbose=False,
                                   compute_loss=compute_loss)
        record = fit.optimizer["higher_order"]
        assert record["terms"] == ["C21", "C23", "C30"]
        assert fit.optimizer["host_numerical_search"] is True
        assert record["host_numerical_search"] is True
        assert record["optimizer_backend"] == "numpy"
        assert record["agreement_backend"] == backend_name
        assert record["agreement"] > record["start_agreement"]
        assert fit.sample_agreement == record["agreement"]
        assert (fit.loss is None) == (not compute_loss)
        assert fit.phase_estimator == "phase_of_mean"  # the fit holds its SSB phase
        phase, _ = ssb.preview(fit.aberrations, sample_tilt_mrad=fit.sample_tilt_mrad or (0.0, 0.0),
                              depth_spread_nm=fit.depth_spread_nm or 0.0, compute_loss=False)
        actual = fit.phase.get() if backend_name == "cuda" else np.asarray(fit.phase)
        expected = np.asarray(phase)  # preview returns the display array on the host
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        # a weak-phase object: the model explains its G (noise-free float input: no Poisson floor), so the guard trusts
        assert record["trusted"] and record["guard"]["coherence"] > 0.95, record["guard"]
        assert ssb.aberrations == fit.aberrations          # reconstruct() applies the fitted terms
    found = fit.aberrations
    assert found["C10_nm"] == pytest.approx(TRUTH["C10_nm"], abs=0.1)
    assert found["C12_nm"] == pytest.approx(TRUTH["C12_nm"], abs=0.05)
    assert found["C21_nm"] == pytest.approx(TRUTH["C21_nm"], rel=0.03)
    assert found["phi21_deg"] == pytest.approx(TRUTH["phi21_deg"], abs=2.0)
    assert found["C23_nm"] == pytest.approx(TRUTH["C23_nm"], rel=0.03)
    assert found["phi23_deg"] == pytest.approx(TRUTH["phi23_deg"], abs=2.0)
    assert found["C30_nm"] == pytest.approx(TRUTH["C30_nm"], rel=0.05)


def test_saved_fit_keeps_the_higher_orders_and_their_setting(simulation, tmp_path):
    from quantem.gpu.ssb.persistence import load_result, result_paths

    backend_name, data, _, detector_mrad = simulation
    np.save(tmp_path / "counts.npy", data)          # an array input records its source for provenance
    with session(data, detector_mrad, backend_name, source_path=tmp_path / "counts.npy") as ssb:
        fit = ssb.find_aberrations(search="primary", tilt=False, higher_order=("C30", "C21"), save_dir=tmp_path, verbose=False)
    signature = fit.metadata["signature"]
    assert signature["settings"]["higher_order"] == ["C21", "C30"]
    loaded = load_result(paths=result_paths(tmp_path, "find_aberrations"), signature=signature, backend=backend_name)
    assert loaded is not None and loaded.aberrations == fit.aberrations
    assert set(loaded.aberrations) == {"C10_nm", "C12_nm", "phi12_deg", "C21_nm", "phi21_deg", "C30_nm"}
    assert loaded.optimizer["higher_order"]["terms"] == ["C21", "C30"]
    assert loaded.optimizer["host_numerical_search"] is True
    assert loaded.optimizer["higher_order"]["agreement_backend"] == backend_name
