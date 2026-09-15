"""Run the exact indexed seven-source scan512 contiguous-quad A/B/A test."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = ROOT / "experiments/20260913-apple-m5-ans-polar-scan512/run.py"
SPEC = importlib.util.spec_from_file_location("polar_scan512_runner", BASE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load resident-loop runner: {BASE_RUNNER}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

BASE.ARMS = (
    ("A1", "A1", "scan512"),
    ("B", "candidate", "scan512-contiguous-quad"),
    ("A2", "A2", "scan512"),
)
BASE.SOURCE_FILES.update({
    "polar_index": (
        "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/"
        "MetalPairedRuntimeTANSPolarIndex.swift"
    ),
    "polar_plan": (
        "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/"
        "PairedRuntimeTANSPolarPlan.swift"
    ),
})
ORIGINAL_FINGERPRINT = BASE._fingerprint_code


def _fingerprint_code(root: Path, executable: Path, manifest: dict, runner=None) -> None:
    ORIGINAL_FINGERPRINT(root, executable, manifest, runner=Path(__file__).resolve())


BASE._fingerprint_code = _fingerprint_code


if __name__ == "__main__":
    BASE.main()
