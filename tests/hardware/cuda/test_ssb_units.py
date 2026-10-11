"""SSB aberrations are in nm at the public API (the engines work in Angstrom), and the detector calibrates itself.

Known-answer checks against abTEM: a thin crystal (one BaTiO3 unit cell, so thickness cannot move the apparent focus)
scanned with a probe of known defocus; the fitted C10 must equal abTEM's C10 (= -defocus) in nm. Before 2026-09-24 the
fit returned the Angstrom number under the nm label (-100.26 for a -10 nm C10).
"""

import math
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
              scan_sampling_A=0.25, det_sampling=det_sampling, scan_detector_rotation_deg=0.0)
    ssb.find_aberrations(trials=200, refinement="nelder-mead", verbose=False)
    expected_nm = -defocus_A / 10.0          # abTEM C10 = -defocus, Angstrom -> nm
    # validated 2026-09-24: -10.03 and +15.03 for -10 and +15 nm
    assert abs(ssb.aberrations["C10_nm"] - expected_nm) < 0.02 * abs(expected_nm)


def test_automatic_detector_sampling_matches_the_simulation_and_recovers_the_defocus():
    """Without det_sampling the bright-field disk edge calibrates the detector to abTEM's sampling.

    The detector sampling is semiangle / disk edge radius: 0.5486 mrad per pixel against abTEM's 0.5469 (+0.3 %), and
    the fit then finds C10 = -9.67 nm for the -10 nm truth. Before 2026-10-05 the automatic sampling was twice the
    semiangle over the integer half-maximum radius, 1.0909 mrad per pixel (+99.5 %), and the fit found -4.36 nm.
    """
    from quantem.gpu import SSB

    counts, det_sampling = _simulate(100.0)
    with SSB(counts, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.25,
             scan_detector_rotation_deg=0.0) as ssb:
        result = ssb.find_aberrations(trials=200, refinement="nelder-mead", verbose=False)
        automatic_sampling = ssb._backend().angular_sampling[0]
    assert automatic_sampling == pytest.approx(det_sampling, rel=0.004)
    assert result.aberrations["C10_nm"] == pytest.approx(-10.0, rel=0.04)


def test_search_ranges_accept_lists_for_every_parameter():
    """``search_ranges`` takes a [low, high] list as well as a tuple; the CUDA search read a phi12 list as a fixed value."""
    from tests.hardware.cuda.test_ssb_thick_sample import _synthetic_session

    ssb, _ = _synthetic_session()
    as_lists = ssb.find_aberrations(search_ranges={"C10_nm": [-5.0, 5.0], "phi12_deg": [-10.0, 10.0]}, trials=4,
                                    refinement=None, check_rotation=False, verbose=False)
    as_tuples = ssb.find_aberrations(search_ranges={"C10_nm": (-5.0, 5.0), "phi12_deg": (-10.0, 10.0)}, trials=4,
                                     refinement=None, check_rotation=False, verbose=False)

    assert dict(as_lists.aberrations) == dict(as_tuples.aberrations)
    assert -10.0 <= as_lists.aberrations["phi12_deg"] <= 10.0


def test_numpy_scalar_sampling_is_read_as_a_number():
    """A scan step or detector sampling read from NumPy metadata (np.float32) works like the Python number."""
    from tests.hardware.cuda.test_ssb_thick_sample import _session

    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rows, cols = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rows, cols) < 10] += 200
    aberrations = {"C10_nm": 3.0, "C12_nm": 0.5, "phi12_deg": math.degrees(0.3)}
    with _session(counts, det_mrad=np.float32(3.0), scan_A=np.float32(0.3)) as ssb:
        phase, loss = ssb.preview(aberrations)
    with _session(counts, det_mrad=3.0, scan_A=float(np.float32(0.3))) as reference:
        expected, expected_loss = reference.preview(aberrations)

    np.testing.assert_array_equal(phase, expected)
    assert loss == expected_loss
