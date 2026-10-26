"""Fused MPS thick-sample SSB kernels against the plain-torch reference, on the public gold scan.

``tests/parity/torch_ssb.py`` is the readable model the CUDA engine is tested against. Here the fused Metal paths of
tilt-corrected (thick-sample) SSB are compared with it on real data: the public gold acquisition (``bobleesj/quantem-data``
revision 5925e4b, ``4dstem/gold_512/gold_004_master.h5``, 512 x 512 scan, 7644 bright-field pixels), at the parameters
the default ``reconstruct()`` fits (the tilt-corrected joint fit) and at perturbed ones:

- ``thick_fit_batch``, the per-term objective (``TorchSSB.fit_power``), for one batch of eight parameter sets, and
  the resident fit's objective (``resident_fit.evaluate``) for the same rows;
- ``reconstruct_thick``, the interactive preview, against the reference mean phase (max and RMS, rad);
- the phase-variance loss of the same call.

Both sides read the same float32 inputs (G(q, k), k and q of the prepared session), so every difference is float32
arithmetic. Each implementation has a float32 bound, measured against the float64 reference in the opt-in test below,
and the parity tolerance is the sum of the two bounds (u = 2^-24, K = bright-field pixels):

- Objective and loss: relative error u sqrt(K) each, 2 u sqrt(K) = 1.0e-5 between them. Each value is a float32 sum
  over the K pixels (per q for the objective, per image pixel for the loss), and random rounding leaves a relative
  error of about u sqrt(K) in such a sum (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., 4.2); no
  credit is taken for averaging over the band's 1e5 q. Measured against float64 on gold: at most 1.4e-7
  (objective) and 3.9e-7 (loss) for either implementation.
- Phase: max 2.5 u chi_max and RMS 0.5 u chi_max each, twice that between them. chi_max = (pi / lambda)
  (2 semiangle)^2 (|C10| + |C12|) is the largest aberration phase at a double overlap (|q -/+ k| reaches twice the
  semiangle; 160 rad at the gold fit, so u chi_max = 9.5e-6 rad), and float32 holds every correction term's phase to
  that scale. How far the mean phase moves for that per-term error depends on the bright-field images' noise, so the
  two factors are the float64 measurement: 0.42 and 0.09 on gold, and 2.0 and 0.42 on a lower-dose 512 x 512 scan
  (8 counts per bright-field pixel and position, 70 here) checked by hand, which sets them.

The bounds sit far below any model-term mistake: moving the tilt by (2, -2) mrad changes the gold objective by 1 %.

``QUANTEM_SSB_CPU_ORACLE=1`` adds the float64 reference (TorchSSB in complex128 on the CPU, about two minutes and
16 GB of memory) at the fitted parameters and checks each float32 implementation against its own bound.
"""

import importlib.util
import math
import os
import platform
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

UNIT_ROUNDOFF = 2.0 ** -24
# float32 phase bounds of one implementation, in units of u chi_max (see the module docstring)
PHASE_MAX = 2.5
PHASE_RMS = 0.5


def _require_mps():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("MPS SSB requires Apple silicon")
    pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("needs the Apple GPU (MPS)")


def _gold_harness():
    """``tests/benchmarks/mps_ssb.py``: the gold scan's revision, cached-file lookup and recorded calibration."""
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "mps_ssb.py"
    spec = importlib.util.spec_from_file_location("mps_ssb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine_rows(fit):
    """The fitted point as an engine row (C10, C12 Angstrom; phi12 rad; tilt row, col mrad; thickness Angstrom) and seven
    perturbations of it: defocus, astigmatism, tilt, depth weighting off and doubled, and one far from the fit."""
    point = np.array([fit.aberrations["C10_nm"] * 10.0, fit.aberrations["C12_nm"] * 10.0,
                      math.radians(fit.aberrations["phi12_deg"]), *fit.sample_tilt_mrad, fit.depth_spread_nm * 10.0])
    return np.array([
        point,
        point + [20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        point + [-20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        point + [0.0, 10.0, 0.3, 0.0, 0.0, 0.0],
        point + [0.0, 0.0, 0.0, 2.0, -2.0, 0.0],
        point * [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        point * [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        point + [-100.0, 30.0, -0.5, -3.0, 3.0, 200.0],
    ])


# rows of _engine_rows the phase is compared at: the fit, its tilt moved by (2, -2) mrad, and the far point
PHASE_ROWS = (0, 4, 7)


@pytest.fixture(scope="module")
def gold():
    _require_mps()
    from quantem.gpu import SSB, io
    from tests.parity.torch_ssb import TorchSSB

    harness = _gold_harness()
    master = harness.gold_master()
    if master is None:
        pytest.skip("the gold scan is not in the Hugging Face cache (tests/benchmarks/mps_ssb.py run --download fetches it)")
    data = io.load(str(master), hot_pixel_correction=harness.HOT_PIXELS, verbose=False)
    harness.calibrate(data)
    ssb = SSB(data)
    assert ssb.backend == "mps"
    fit = ssb.reconstruct()
    assert fit.sample_tilt_mrad is not None and fit.depth_spread_nm, "the default fit is tilt-corrected"
    backend = ssb._backend()
    backend.cache_rotation(math.radians(fit.scan_detector_rotation_deg))
    prepared = backend._prepared
    case = {
        "prepared": prepared,
        "chunk_bf": backend._phase_chunk_bf,
        "rows": _engine_rows(fit),
        "reference": TorchSSB.from_mps(prepared, device="mps"),
    }
    yield case
    case.clear()
    import torch

    torch.mps.empty_cache()
    ssb.close()
    data.close()


def _objective_rtol(prepared) -> float:
    """Relative rounding of one float32 sum over the bright-field pixels (see the module docstring)."""
    return UNIT_ROUNDOFF * math.sqrt(prepared.num_bright_field_pixels)


def _chi_max(prepared, row) -> float:
    """Largest aberration phase at a double overlap, rad: |q -/+ k| reaches twice the semiangle."""
    alpha = 2.0 * prepared.semiangle_rad
    return math.pi / prepared.wavelength * alpha * alpha * (abs(row[0]) + abs(row[1]))


def _mps_preview(case, row):
    from quantem.gpu.direct_ptycho.mps.thick_sample import reconstruct_thick

    return reconstruct_thick(case["prepared"], C10=row[0], C12=row[1], phi12=row[2], sample_tilt_mrad=(row[3], row[4]),
                             thickness=row[5], chunk_bf=case["chunk_bf"])


def _phase_errors(phase, reference_phase):
    difference = np.asarray(phase, np.float64) - np.asarray(reference_phase, np.float64)
    return float(np.abs(difference).max()), float(np.sqrt(np.mean(difference ** 2)))


def test_fit_objective_matches_torch_reference(gold):
    """The fused batch objective equals ``fit_power`` for eight parameter sets around the gold fit."""
    from quantem.gpu.direct_ptycho.mps.thick_sample import thick_fit_batch

    rows, reference = gold["rows"], gold["reference"]
    fused = thick_fit_batch(gold["prepared"], rows)
    expected = np.array([reference.fit_power(r[0], r[1], r[2], (r[3], r[4]), r[5]) for r in rows])
    relative = np.abs(fused - expected) / np.abs(expected)
    print(f"objective relative error, max {relative.max():.2e}: {np.array2string(relative, precision=2)}")
    assert relative.max() <= 2.0 * _objective_rtol(gold["prepared"])
    # the objective really sees the tilt: the fit is the best of these rows, by more than the tolerance
    assert np.argmax(expected) == 0 and expected[4] < expected[0] * (1.0 - 1e-3)


def test_resident_fit_objective_matches_torch_reference(gold):
    """The resident fit's objective (angle-sum phases advanced along detector rows) equals ``fit_power`` too."""
    from quantem.gpu.direct_ptycho._resident_search import _Device, _trial
    from quantem.gpu.direct_ptycho.mps import resident_fit

    rows, reference = gold["rows"], gold["reference"]
    device = _Device("mps")
    resident = device.host(resident_fit.evaluate(gold["prepared"], _trial(device, device.pack(rows))))
    expected = np.array([reference.fit_power(r[0], r[1], r[2], (r[3], r[4]), r[5]) for r in rows])
    relative = np.abs(resident - expected) / np.abs(expected)
    print(f"resident objective relative error, max {relative.max():.2e}: {np.array2string(relative, precision=2)}")
    assert relative.max() <= 2.0 * _objective_rtol(gold["prepared"])


@pytest.mark.parametrize("index", PHASE_ROWS)
def test_thick_preview_matches_torch_reference(gold, index):
    """``reconstruct_thick`` phase and loss equal the reference mean phase and phase-variance loss."""
    row = gold["rows"][index]
    phase, loss = _mps_preview(gold, row)
    reference_phase, reference_loss = gold["reference"].reconstruct(row[0], row[1], row[2], (row[3], row[4]), row[5])
    max_error, rms_error = _phase_errors(phase, reference_phase.cpu().numpy())
    scale = UNIT_ROUNDOFF * _chi_max(gold["prepared"], row)
    loss_error = abs(loss - reference_loss) / abs(reference_loss)
    print(f"row {index}: phase max {max_error:.2e} rad ({max_error / scale:.2f} u chi_max), RMS {rms_error:.2e} rad "
          f"({rms_error / scale:.2f} u chi_max), loss relative error {loss_error:.2e}")
    assert max_error <= 2.0 * PHASE_MAX * scale
    assert rms_error <= 2.0 * PHASE_RMS * scale
    assert loss_error <= 2.0 * _objective_rtol(gold["prepared"])


@pytest.mark.skipif(os.environ.get("QUANTEM_SSB_CPU_ORACLE") != "1",
                    reason="float64 CPU reference: set QUANTEM_SSB_CPU_ORACLE=1 (about two minutes, 16 GB)")
def test_each_float32_path_is_within_its_bound_of_float64(gold):
    """The float64 reference at the fit: the fused MPS kernels and the float32 reference each stay within their own
    float32 bound, so neither side of the parity above hides an error behind the other."""
    import torch

    from quantem.gpu.direct_ptycho.mps.thick_sample import thick_fit_batch
    from tests.parity.torch_ssb import TorchSSB

    prepared, row = gold["prepared"], gold["rows"][0]
    exact = TorchSSB.from_mps(prepared, device="cpu", dtype=torch.complex128)
    arguments = (row[0], row[1], row[2], (row[3], row[4]), row[5])
    objective = exact.fit_power(*arguments)
    exact_phase, exact_loss = exact.reconstruct(*arguments)
    del exact
    exact_phase = exact_phase.numpy()
    scale = UNIT_ROUNDOFF * _chi_max(prepared, row)
    rtol = _objective_rtol(prepared)
    phase, loss = _mps_preview(gold, row)
    reference_phase, reference_loss = gold["reference"].reconstruct(*arguments)
    measured = {
        "fused MPS": (float(thick_fit_batch(prepared, row[None])[0]), phase, loss),
        "torch float32": (gold["reference"].fit_power(*arguments), reference_phase.cpu().numpy(), reference_loss),
    }
    for name, (value, image, image_loss) in measured.items():
        objective_error = abs(value - objective) / objective
        max_error, rms_error = _phase_errors(image, exact_phase)
        loss_error = abs(image_loss - exact_loss) / exact_loss
        print(f"{name} vs float64: objective {objective_error:.2e}, phase max {max_error:.2e} rad "
              f"({max_error / scale:.2f} u chi_max), RMS {rms_error:.2e} rad, loss {loss_error:.2e}")
        assert objective_error <= rtol
        assert max_error <= PHASE_MAX * scale
        assert rms_error <= PHASE_RMS * scale
        assert loss_error <= rtol
