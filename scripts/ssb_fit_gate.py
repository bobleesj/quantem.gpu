"""Execute frozen GPU fits with immutable inputs and separate run outputs."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Run the native repeat and current MPS reference under one process lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--batch-mode", choices=("closure", "both"), required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    cases = ["arina-128-full-disk"]
    if args.full:
        cases.append("arina-512-full-disk")
    pin_path = Path("tests/parity/fixtures/ssb_fit_trajectory_128.json")
    pin_bytes = pin_path.read_bytes()
    expected = json.loads(pin_bytes)["case_json_sha256"]
    actual = hashlib.sha256(
        (args.runs / "strict-parity" / cases[0] / "case.json").read_bytes()
    ).hexdigest()
    if actual != expected:
        raise ValueError(f"Fixture SHA256 {actual} differs from pin {expected}; restore the original fixture.")
    output = Path(os.environ.get(
        "QUANTEM_SSB_PARITY_OUTPUT",
        args.runs / ("fit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")),
    ))
    output.mkdir(parents=True, exist_ok=False)
    lock_path = Path(os.environ.get(
        "QUANTEM_SSB_PARITY_LOCK", Path.home() / ".cache/quantem/gpu-parity.lock"
    ))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve()) + os.pathsep + environment.get("PYTHONPATH", "")

    def run(command: list[str]) -> None:
        subprocess.run(command, env=environment, check=True)

    print(f"GPU gate output: {output}", flush=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in cases:
            case_root = args.runs / "strict-parity" / name
            first, repeat = output / name, output / (name + "-repeat")
            for destination in (first, repeat):
                run(["build/ssb-fit-trajectory", str(case_root), str(destination), args.batch_mode])
            reference = output / (name + "-reference.json")
            run([sys.executable, "scripts/ssb_fit_reference_trajectory.py",
                 "--runs", str(args.runs), "--case", name, "--json", str(reference)])
            command = [sys.executable, "scripts/ssb_fit_trajectory_report.py",
                       "--native", str(first / "fit-trajectory.json"),
                       "--repeat", str(repeat / "fit-trajectory.json"),
                       "--reference", str(reference)]
            if name == cases[0]:
                command.extend(["--pin", str(pin_path)])
            run(command)
    if pin_path.read_bytes() != pin_bytes:
        raise RuntimeError("Frozen pin changed during execution; investigate before accepting the run.")
    print("PASS: frozen GPU fit, repeat determinism, and MPS reference recording", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
