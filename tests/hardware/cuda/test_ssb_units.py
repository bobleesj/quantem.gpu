"""SSB aberrations are in nm at the public API (the engines work in Angstrom).

Known-answer check against abTEM: a thin crystal (one BaTiO3 unit cell, so thickness cannot move the apparent focus) scanned
with a probe of known defocus; the fitted C10 must equal abTEM's C10 (= -defocus) in nm. Before 2026-09-24 the fit returned
the Angstrom number under the nm label (-100.26 for a -10 nm C10).
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("defocus_A", [100.0, -150.0])
def test_fitted_c10_is_nm(defocus_A):
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    abtem = pytest.importorskip("abtem")
    from ase import Atoms

    from quantem.gpu import SSB

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
    ssb = SSB.from_array(np.asarray(measurement.array, dtype=np.float32), backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0,
                         scan_sampling_A=0.25, det_sampling=float(measurement.angular_sampling[0]), rotation_angle_deg=0.0)
    ssb.fit(trials=200, refinement="nelder-mead", verbose=False)
    expected_nm = -defocus_A / 10.0          # abTEM C10 = -defocus, Angstrom -> nm
    # validated 2026-09-24: -10.03 and +15.03 for -10 and +15 nm
    assert abs(ssb.aberrations["C10"] - expected_nm) < 0.02 * abs(expected_nm)
