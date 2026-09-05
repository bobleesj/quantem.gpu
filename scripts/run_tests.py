"""Run named regression suites or translate retained pre-migration test paths.

Examples
--------
python scripts/run_tests.py contracts parity -q
python scripts/run_tests.py hardware/cuda -q
python scripts/run_tests.py tests/io/test_load.py -q

Hardware opt-ins and pytest options are unchanged. A skipped device gate is
not a hardware pass. Historical commands remain reproducible at their pinned
revision; this runner translates their test-file paths for the current tree.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "all": "tests",
    "contracts": "tests/contracts",
    "parity": "tests/parity",
    "hardware": "tests/hardware",
    "hardware/cuda": "tests/hardware/cuda",
    "hardware/mps": "tests/hardware/mps",
    "e2e": "tests/e2e",
    "infrastructure": "tests/infrastructure",
}


def main(argv=None):
    """Execute pytest without changing scientific gates or tolerance settings."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--list"]:
        for name, path in SUITES.items():
            print(f"{name:18} {path}")
        return 0
    migrations = json.loads((ROOT / "tests/path_migrations.json").read_text())
    translated = []
    for value in arguments or ["all"]:
        path, separator, node = value.partition("::")
        target = SUITES.get(path, migrations.get(path, path))
        translated.append(target + separator + node)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT / "src"), environment.get("PYTHONPATH")) if value
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", *translated], cwd=ROOT, env=environment
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
