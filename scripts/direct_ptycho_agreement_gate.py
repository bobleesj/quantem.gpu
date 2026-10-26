"""Gate the native Direct Ptycho fit on the agreement objective: the Swift port must draw the Python search's trials.

The default fit (``find_aberrations()``, tilt-corrected) and ``find_aberrations(tilt=False)`` maximize the model
agreement with the resident multivariate TPE of ``direct_ptycho._resident_search``, then a trust-region Newton on the
exact gradient and Hessian (``direct_ptycho._exact_newton``). The native ``SSBAgreementSearch`` ports both. This gate runs
them on one synthetic float32 objective, which each side's search evaluates from the same float32 trial rows,
with the same analytic derivatives for the Newton, and requires the 200 search rows to be identical and the
fitted points to agree to 1e-6:

1. Python ``fit_resident`` and ``fit_standard_resident`` (seed 42, 200 trials) on MPS or CUDA record the
   SHA-256 of their search rows and their fits;
2. ``swift test --filter SSBAgreementSearchTests`` checks the native fits against those records (and its own
   pins);
3. with ``SSB_PARITY_FIXTURE`` and ``SSB_PARITY_REFERENCE`` set, ``MetalSSBMeasuredParityTests`` also compares
   the native fits, branch decision and phases with an MPS reference on a measured acquisition.

Usage: PYTHONPATH=src python scripts/direct_ptycho_agreement_gate.py [--backend mps] [--json out.json]
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

# The maximum of ``synthetic_score`` in the Newton coordinates z (C10, C12 cos 2 phi12, C12 sin 2 phi12, tilt row,
# tilt column [mrad], depth spread [A]) and its curvatures, as ``SSBAgreementSearchTests.target`` and ``curvature``.
TARGET = (37.25, 30.0, -12.0, 3.5, -1.25, 151.0)
CURVATURE = (0.25, 1.0, 1.0, 400.0, 400.0, 0.01)


def synthetic_score(trial, xp):
    """The float32 objective of ``SSBAgreementSearchTests.score``, written in the same order."""
    f = lambda value: xp.array(value, dtype=xp.float32)
    c10, c12, c2, s2, t, thr, thc = (trial[:, i] for i in range(7))
    a = c12 * c2 - f(30.0)
    b = c12 * s2 + f(12.0)
    d10 = c10 - f(37.25)
    dr = thr * f(1000.0) - f(3.5)
    dc = thc * f(1000.0) + f(1.25)
    dt = t - f(151.0)
    s = f(1.0e6) - d10 * d10 * f(0.25) - a * a - b * b
    s = s - dr * dr * f(400.0) - dc * dc * f(400.0)
    return s - dt * dt * f(0.01)


def synthetic_exact(z):
    """``synthetic_score`` in z in double precision with its analytic gradient and Hessian, summed in the order of
    ``SSBAgreementSearchTests.exact``."""
    value = 1.0e6
    gradient = np.zeros(6)
    hessian = np.zeros((6, 6))
    for d in range(6):
        x = abs(z[d]) if d == 5 else z[d]
        r = x - TARGET[d]
        value -= CURVATURE[d] * r * r
        gradient[d] = -2 * CURVATURE[d] * r * (-1 if d == 5 and z[d] < 0 else 1)
        hessian[d, d] = -2 * CURVATURE[d]
    return value, gradient, hessian


class SyntheticDerivatives:
    """The Newton objective of ``_exact_newton``: values at rows of z, derivatives at rows of the latest call."""

    def __init__(self):
        self.rows = np.zeros((0, 6))

    def values(self, rows, all_derivatives):
        self.rows = np.asarray(rows, dtype=float)
        return np.array([synthetic_exact(z)[0] for z in self.rows])

    def derivatives(self, indices):
        return [synthetic_exact(self.rows[index]) for index in indices]


def python_reference(backend: str) -> dict:
    """Run the Python resident fits on the synthetic objective and digest their search rows."""
    from quantem.gpu.direct_ptycho._resident_search import _Device, fit_resident, fit_standard_resident

    xp = _Device(backend).xp
    rows: list[np.ndarray] = []

    def evaluate(trial):
        trial = trial.astype(xp.float32)
        values = synthetic_score(trial, xp)
        host = np.asarray(trial) if backend == "mps" else trial.get()
        rows.append(np.asarray(host, dtype=np.float32))
        zero = xp.zeros_like(values)
        return xp.stack([values, zero, zero], axis=1)

    def search(point):
        searched = np.concatenate(rows).astype(np.float32)
        return {"search_rows": len(searched), "search_sha256": hashlib.sha256(searched.tobytes()).hexdigest(),
                "point": point}

    tilt = fit_resident(evaluate, backend=backend, trials=200, seed=42, verbose=False, compare_standard=False,
                        derivatives=SyntheticDerivatives())
    record = {"tilt": search(
        [tilt[key] for key in ("C10", "C12", "phi12", "tilt_row_mrad", "tilt_col_mrad", "depth_spread_A")])}
    rows.clear()
    standard = fit_standard_resident(evaluate, backend=backend, lower=(-400.0, 0.0, -np.pi / 2),
                                     upper=(400.0, 100.0, np.pi / 2), trials=200, seed=42,
                                     derivatives=SyntheticDerivatives())
    record["standard"] = search([standard[key] for key in ("C10", "C12", "phi12")])
    return record


def swift_test(filter_: str, environment: dict) -> int:
    command = ["swift", "test", "-c", "release", "-Xswiftc", "-enable-testing", "--filter", filter_]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, env=environment).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="mps", choices=("mps", "cuda"))
    parser.add_argument("--json", type=Path, help="Also write the Python records here")
    args = parser.parse_args()
    record = python_reference(args.backend)
    print(json.dumps(record, indent=1), flush=True)
    if args.json:
        args.json.write_text(json.dumps(record, indent=1))
    with tempfile.TemporaryDirectory(prefix="direct-ptycho-agreement-gate-") as scratch:
        reference = Path(scratch) / "agreement-reference.json"
        reference.write_text(json.dumps(record))
        environment = {**os.environ, "SSB_AGREEMENT_REFERENCE": str(reference)}
        failed = swift_test("SSBAgreementSearchTests", environment)
        if environment.get("SSB_PARITY_FIXTURE") and environment.get("SSB_PARITY_REFERENCE"):
            failed = failed or swift_test("MetalSSBMeasuredParityTests", environment)
        else:
            print("measured parity: SKIP (set SSB_PARITY_FIXTURE and SSB_PARITY_REFERENCE)", flush=True)
    print("FAIL" if failed else "PASS: the native agreement fit draws the Python trials and fits", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
