"""Gate the native SSB fit on the agreement objective: the Swift port must draw the Python search's trials.

The default fit (``find_aberrations()``, tilt-corrected) and ``find_aberrations(tilt=False)`` maximize the model
agreement with the resident multivariate TPE and trust-region Newton of ``ssb._resident_search``. The native
``SSBAgreementSearch`` is a port of that search. This gate runs both on one synthetic float32 objective, which
each side evaluates from the same float32 trial rows, and requires every trial row (search and Newton, in
order) to be identical and the fitted points to agree to 1e-6:

1. Python ``fit_resident`` and ``fit_standard_resident`` (seed 42, 200 trials) on MPS or CUDA record the
   SHA-256 of their trial rows and their fits;
2. ``swift test --filter SSBAgreementSearchTests`` checks the native search against those records (and its
   own pins);
3. with ``SSB_PARITY_FIXTURE`` and ``SSB_PARITY_REFERENCE`` set, ``MetalSSBMeasuredParityTests`` also compares
   the native fits, branch decision and phases with an MPS reference on a measured acquisition.

Usage: PYTHONPATH=src python scripts/ssb_agreement_gate.py [--backend mps] [--json out.json]
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


def python_reference(backend: str) -> dict:
    """Run the Python resident fits on the synthetic objective and digest their trial rows."""
    from quantem.gpu.ssb._resident_search import _Device, fit_resident, fit_standard_resident

    xp = _Device(backend).xp
    rows: list[np.ndarray] = []

    def evaluate(trial):
        trial = trial.astype(xp.float32)
        values = synthetic_score(trial, xp)
        host = np.asarray(trial) if backend == "mps" else trial.get()
        rows.append(np.asarray(host, dtype=np.float32))
        zero = xp.zeros_like(values)
        return xp.stack([values, zero, zero], axis=1)

    def digest():
        return hashlib.sha256(np.concatenate(rows).astype(np.float32).tobytes()).hexdigest()

    tilt = fit_resident(evaluate, backend=backend, trials=200, seed=42, verbose=False, compare_standard=False)
    record = {"tilt": {
        "point": [tilt[key] for key in ("C10", "C12", "phi12", "tilt_row_mrad", "tilt_col_mrad", "depth_spread_A")],
        "agreement": tilt["fit"], "rows": int(sum(len(r) for r in rows)), "sha256": digest()}}
    rows.clear()
    standard = fit_standard_resident(evaluate, backend=backend, lower=(-400.0, 0.0, -np.pi / 2),
                                     upper=(400.0, 100.0, np.pi / 2), trials=200, seed=42)
    record["standard"] = {"point": [standard[key] for key in ("C10", "C12", "phi12")], "agreement": standard["fit"],
                          "rows": int(sum(len(r) for r in rows)), "sha256": digest()}
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
    with tempfile.TemporaryDirectory(prefix="ssb-agreement-gate-") as scratch:
        reference = Path(scratch) / "agreement-reference.json"
        reference.write_text(json.dumps(record))
        environment = {**os.environ, "SSB_AGREEMENT_REFERENCE": str(reference)}
        failed = swift_test("SSBAgreementSearchTests", environment)
        if environment.get("SSB_PARITY_FIXTURE") and environment.get("SSB_PARITY_REFERENCE"):
            failed = failed or swift_test("MetalSSBMeasuredParityTests", environment)
        else:
            print("measured parity: SKIP (set SSB_PARITY_FIXTURE and SSB_PARITY_REFERENCE)", flush=True)
    print("FAIL" if failed else "PASS: the native agreement search draws the Python trials", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
