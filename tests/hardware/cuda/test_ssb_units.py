"""SSB aberrations are in nm at the public API (the engines work in Angstrom), and the detector calibrates itself.

Known-answer checks against abTEM: a thin crystal (one BaTiO3 unit cell, so thickness cannot move the apparent focus)
scanned with a probe of known defocus; the fitted C10 must equal abTEM's C10 (= -defocus) in nm. Before 2026-09-24 the
fit returned the Angstrom number under the nm label (-100.26 for a -10 nm C10).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

cp = pytest.importorskip("cupy")


def _simulate(defocus_A: float) -> tuple[np.ndarray, float]:
    """abTEM 4D-STEM of a one-cell BaTiO3 crystal at 300 kV, 30 mrad: the counts and abTEM's detector sampling (mrad)."""
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    abtem = pytest.importorskip("abtem")
    from ase import Atoms

    abtem.config.set({"device": "gpu"})
    a, cells = 4.0, 9
    box = cells * a
    basis = [("Ba", (0, 0, 0)), ("Ti", (0.5, 0.5, 0.5)), ("O", (0.5, 0.5, 0)), ("O", (0.5, 0, 0.5)), ("O", (0, 0.5, 0.5))]
    atoms = Atoms([s for _ in range(cells * cells) for s, _f in basis],
                  positions=[((i + f[0]) * a, (j + f[1]) * a, f[2] * a + 0.5) for i in range(cells) for j in range(cells) for _s, f in basis],
                  cell=[box, box, a + 1.0], pbc=True)
    potential = abtem.Potential(atoms, sampling=0.08, slice_thickness=a / 2, projection="infinite", parametrization="lobato")
    probe = abtem.Probe(energy=300e3, semiangle_cutoff=30, defocus=defocus_A)
    scan = abtem.GridScan(start=(2 * a, 2 * a), end=(6 * a, 6 * a), gpts=(64, 64), endpoint=False)
    measurement = probe.scan(potential, scan=scan, detectors=abtem.PixelatedDetector(max_angle=45)).compute()
    return np.asarray(measurement.array, dtype=np.float32), float(measurement.angular_sampling[0])


@pytest.mark.parametrize("defocus_A", [100.0, -150.0])
def test_fitted_c10_is_nm(defocus_A):
    from quantem.gpu import SSB

    counts, det_sampling = _simulate(defocus_A)
    ssb = SSB(counts, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0,
              scan_sampling_A=0.25, det_sampling=det_sampling, rotation_angle_deg=0.0)
    ssb.find_aberrations(trials=200, refinement="nelder-mead", verbose=False)
    expected_nm = -defocus_A / 10.0          # abTEM C10 = -defocus, Angstrom -> nm
    # validated 2026-09-24: -10.03 and +15.03 for -10 and +15 nm
    assert abs(ssb.aberrations["C10"] - expected_nm) < 0.02 * abs(expected_nm)


def test_automatic_detector_sampling_matches_the_simulation_and_recovers_the_defocus():
    """Without det_sampling the bright-field disk edge calibrates the detector to abTEM's sampling.

    The detector sampling is semiangle / disk edge radius: 0.5486 mrad per pixel against abTEM's 0.5469 (+0.3 %), and
    the fit then finds C10 = -9.67 nm for the -10 nm truth. Before 2026-10-05 the automatic sampling was twice the
    semiangle over the integer half-maximum radius, 1.0909 mrad per pixel (+99.5 %), and the fit found -4.36 nm.
    """
    from quantem.gpu import SSB

    counts, det_sampling = _simulate(100.0)
    with SSB(counts, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.25,
             rotation_angle_deg=0.0) as ssb:
        result = ssb.find_aberrations(trials=200, refinement="nelder-mead", verbose=False)
        automatic_sampling = ssb._prepare_cuda().angular_sampling[0]
    assert automatic_sampling == pytest.approx(det_sampling, rel=0.004)
    assert result.aberrations["C10"] == pytest.approx(-10.0, rel=0.04)
