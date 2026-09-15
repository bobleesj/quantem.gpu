"""Profile the best validated scan512 + trusted-table detector path."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/run.py"
SPEC = importlib.util.spec_from_file_location("best_path_compose_runner", COMPOSE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load validated composition runner: {COMPOSE_RUNNER}")
compose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compose)
base = compose.base


_environment = base._environment


def profile_environment() -> dict[str, str]:
    environment = _environment()
    # The resident creates the timestamp profiler during source construction.
    environment["QGPU_PAIRED_RUNTIME_PROFILE"] = "1"
    return environment


base._environment = profile_environment

_request = base._request


def profile_request(process, raw, command: dict) -> dict:
    request = dict(command)
    if request.get("op") == "run":
        request["profile"] = True
    return _request(process, raw, request)


base._request = profile_request

_fingerprint = base._fingerprint_code


def fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    _fingerprint(root, executable, manifest)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["code"]["profile_runner_sha256"] = digest
    manifest["code"]["profile_runner"] = str(Path(__file__).resolve().relative_to(root))


base._fingerprint_code = fingerprint_code


if __name__ == "__main__":
    base.main()
