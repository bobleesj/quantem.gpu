"""fit() settles the 180-degree ambiguity of the scan-detector rotation.

The centre-of-mass rotation search cannot tell omega from omega + 180 degrees, and SSB fits both equally well: the second
is the negated phase with the opposite defocus. Atom columns carry positive phase, so ``fit()`` keeps the rotation whose
phase has a positive column sign and otherwise reverses the CoM for the session (the angle stays below 180 degrees,
``com_reversed`` flips) and refits with every aberration flipped in sign (C10 -> -C10, C12 -> -C12). Parity: starting from the wrong rotation must land on the same reconstruction as a direct fit at the right one.
"""

import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from tests.hardware.cuda.test_ssb_thick_sample import _simulated_crystal  # noqa: E402

# an Arina master of a real 512 x 512 x 192 x 192 acquisition whose right rotation is 158.9 deg (atoms bright)
SOURCE = Path(os.environ.get("QUANTEM_SSB_ARINA_MASTER", ""))
REAL = dict(backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.264, det_sampling=0.5554)
REAL_ROTATION_DEG = 158.9


def _require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")


def _host(array) -> np.ndarray:
    return np.asarray(cp.asnumpy(array) if isinstance(array, cp.ndarray) else array, np.float64)


def _correlation(first, second) -> float:
    first, second = _host(first), _host(second)
    first, second = first - first.mean(), second - second.mean()
    return float((first * second).sum() / np.sqrt((first * first).sum() * (second * second).sum()))


def _astigmatism(aberrations) -> complex:
    return aberrations["C12"] * np.exp(2j * aberrations["phi12"])


def _simulated(rotation_angle_deg: float):
    from quantem.gpu import SSB

    data, det_sampling = _simulated_crystal((3.0, -4.0))
    return SSB(data, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.25,
                          det_sampling=det_sampling, rotation_angle_deg=rotation_angle_deg)


# ---


def test_simulated_crystal_wrong_rotation_is_turned_and_refit():
    """Known answer (atoms positive, right rotation 0 deg): starting at 180 deg (0 deg, CoM reversed), fit() restores the CoM."""
    _require_cuda()
    direct = _simulated(0.0).find_aberrations(check_rotation=False, verbose=False)
    session = _simulated(180.0)
    checked = session.find_aberrations(verbose=False)
    assert checked.rotation_flipped
    assert checked.rotation_angle_deg == pytest.approx(0.0, abs=1e-9) and not checked.com_reversed
    assert session.rotation_angle_deg == pytest.approx(0.0, abs=1e-9) and not session.com_reversed   # session keeps it
    # validated 2026-09-26: column sign +0.88; phase correlation 0.993 with the direct fit, whose loss is 0.8% higher
    # (the refit from the flipped aberrations lands in a slightly deeper minimum of the same branch)
    assert checked.column_sign > 0.8
    assert _correlation(checked.phase, direct.phase) > 0.99
    assert checked.loss <= direct.loss * (1.0 + 1e-6)


def test_simulated_crystal_right_rotation_is_left_exactly_as_fitted():
    """Starting at the right rotation, the check only measures: same result as check_rotation=False, bit for bit."""
    _require_cuda()
    kept = _simulated(0.0).find_aberrations(verbose=False)
    direct = _simulated(0.0).find_aberrations(check_rotation=False, verbose=False)
    assert not kept.rotation_flipped
    assert kept.column_sign > 0.8
    np.testing.assert_array_equal(_host(kept.phase), _host(direct.phase))
    assert kept.aberrations == direct.aberrations
    assert kept.loss == direct.loss


def test_check_rotation_false_keeps_the_given_rotation():
    """Opting out keeps the rotation as given, dark atoms included, and records no column sign."""
    _require_cuda()
    from quantem.gpu.ssb.results import column_sign

    result = _simulated(180.0).find_aberrations(check_rotation=False, verbose=False)
    assert result.rotation_angle_deg == pytest.approx(0.0, abs=1e-9) and result.com_reversed
    assert not result.rotation_flipped and result.column_sign is None
    assert column_sign(result.phase) < -0.8


def test_real_acquisition_from_the_wrong_rotation_matches_a_direct_fit():
    """Real 512 x 512 data started 180 deg off: same rotation, aberrations and phase as a direct fit at the right one."""
    if not SOURCE.is_file():
        pytest.skip("set QUANTEM_SSB_ARINA_MASTER to a real Arina master file")
    _require_cuda()
    from quantem.gpu import SSB

    direct = SSB.open(str(SOURCE), rotation_angle_deg=REAL_ROTATION_DEG, **REAL).find_aberrations(check_rotation=False, verbose=False)
    direct_phase, direct_loss, direct_aberrations = _host(direct.phase), direct.loss, dict(direct.aberrations)
    del direct
    cp.get_default_memory_pool().free_all_blocks()
    checked = SSB.open(str(SOURCE), rotation_angle_deg=REAL_ROTATION_DEG + 180.0, **REAL).find_aberrations(verbose=False)
    assert checked.rotation_flipped
    assert checked.rotation_angle_deg == pytest.approx(REAL_ROTATION_DEG, abs=1e-9) and not checked.com_reversed
    # validated 2026-09-26: column sign +0.665, phase correlation 0.99995, loss 1.4e-6 relative, C10 0.018 nm apart
    # (two Nelder-Mead polishes from different starts on the same minimum)
    assert checked.column_sign > 0.5
    assert _correlation(checked.phase, direct_phase) > 0.9999
    assert checked.loss == pytest.approx(direct_loss, rel=1e-5)
    assert checked.aberrations["C10"] == pytest.approx(direct_aberrations["C10"], abs=0.05)
    # astigmatism as one physical quantity, C12 * exp(2 i phi12): (-C12, phi12) and (C12, phi12 + 90 deg) are the same
    # aberration, and the refit starts from every aberration flipped, so compare the vector, not the signed C12
    assert abs(_astigmatism(checked.aberrations) - _astigmatism(direct_aberrations)) < 0.05


def test_reversed_com_is_the_conjugate_object_with_every_aberration_flipped():
    """(angle, com_reversed=True) reconstructs the negated phase of (angle, False) with C10 and C12 flipped: the engine
    receives the physical angle + 180 degrees, and nothing else changes."""
    _require_cuda()
    aberrations = {"C10": 1.5, "C12": 0.8, "phi12": 0.4}
    flipped = {"C10": -1.5, "C12": -0.8, "phi12": 0.4}
    measured = _simulated(0.0).reconstruct(aberrations=aberrations)
    reversed_ = _simulated(0.0)
    reversed_.set_rotation(0.0, com_reversed=True)
    conjugate = reversed_.reconstruct(aberrations=flipped)
    assert reversed_.rotation_angle_deg == 0.0 and reversed_.com_reversed and reversed_.physical_rotation_deg == 180.0
    assert conjugate.rotation_angle_deg == 0.0 and conjugate.com_reversed
    assert _correlation(conjugate.phase, -_host(measured.phase)) > 0.9999
    same = _simulated(180.0).reconstruct(aberrations=flipped)     # the old spelling of the same physical rotation
    np.testing.assert_array_equal(_host(same.phase), _host(conjugate.phase))
