"""``run(aberrations=...)`` on the session backend, against a weak-phase simulation: the image each call reconstructs.

A fifth-order mapping and its C10_nm, C12_nm and phi12_deg alone give different images from two new sessions, for the
phase of the mean wave and for the direct phase. On one session the second call overrides the three it names and
keeps the higher orders the first left there, so it reconstructs the first image again and one line says which terms
stayed. Assigning ``known_aberrations`` replaces all of them.
"""

import numpy as np
import pytest

from tests.hardware.thin_object import TRUTH, backend_or_skip, session, simulate

pytestmark = pytest.mark.slow

LOW = {name: TRUTH[name] for name in ("C10_nm", "C12_nm", "phi12_deg")}
FULL = {**LOW, "C21_nm": 40.0, "phi21_deg": 60.0, "C23_nm": 30.0, "phi23_deg": -20.0, "C30_nm": 3000.0,
        "C32_nm": 2000.0, "phi32_deg": 10.0, "C34_nm": 1500.0, "phi34_deg": -30.0, "C41_nm": 5e4, "phi41_deg": 70.0,
        "C43_nm": 4e4, "phi43_deg": -12.0, "C45_nm": 3e4, "phi45_deg": 3.0, "C50_nm": 3e5, "C52_nm": 4e5,
        "phi52_deg": 68.0, "C54_nm": 3e5, "phi54_deg": -3.0, "C56_nm": 3e5, "phi56_deg": -3.0}
KEPT = "12 higher-order terms the session holds from its latest fit, run() or assignment stay applied"


@pytest.fixture(scope="module")
def simulation():
    backend = backend_or_skip()
    # CUDA: 256 x 256, where the direct phase is the wave sum
    return (backend, *simulate(TRUTH, tile=4 if backend == "cuda" else 1))


def _phase(result) -> np.ndarray:
    phase = result.phase
    return np.asarray(phase.get() if hasattr(phase, "get") else phase, np.float64)


def _difference(image: np.ndarray, reference: np.ndarray) -> float:
    """Standard deviation of the difference, as a fraction of the reference image\x27s."""
    return float(np.std(image - reference) / np.std(reference))


@pytest.mark.parametrize("method", ("phase_of_mean", "phase"))
def test_higher_orders_change_the_image_and_an_override_that_keeps_them_says_so(simulation, capsys, method):
    backend, data, _, detector_mrad = simulation
    options = dict(method=method, compute_loss=False)
    with session(data, detector_mrad, backend) as direct:
        full = _phase(direct.run(aberrations=dict(FULL), **options))
    with session(data, detector_mrad, backend) as direct:
        low = _phase(direct.run(aberrations=dict(LOW), **options))
    assert _difference(low, full) > 0.05
    assert "stay applied" not in capsys.readouterr().out
    with session(data, detector_mrad, backend) as direct:
        direct.run(aberrations=dict(FULL), **options)
        kept = direct.run(aberrations=dict(LOW), **options)
        assert KEPT in capsys.readouterr().out
        assert kept.aberrations == FULL
        assert _difference(_phase(kept), full) < 1e-6
        direct.known_aberrations = dict(LOW)
        replaced = direct.run(**options)
        assert replaced.aberrations == LOW
        assert _difference(_phase(replaced), low) < 1e-6
        assert "stay applied" not in capsys.readouterr().out


def test_all_methods_say_once_which_higher_orders_stay(simulation, capsys):
    backend, data, _, detector_mrad = simulation
    if backend != "cuda":
        pytest.skip("method=\x27all\x27 requires CUDA")
    with session(data, detector_mrad, backend) as direct:
        direct.run(aberrations=dict(FULL), compute_loss=False)
        capsys.readouterr()
        results = direct.run(method="all", aberrations=dict(LOW), compute_loss=False)
        assert capsys.readouterr().out.count(KEPT) == 1
        assert all(result.aberrations == FULL for result in results.values())
