from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_webgpu_h5_browser import _exact_run_evidence

FULL_HASH = "a" * 64


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        require_integer_resident=True,
        expected_resident_dtype="uint16",
        require_checksum_parity=True,
        require_full_output_parity=True,
        expected_full_output_sha256=FULL_HASH,
    )


def _profile(dtype: str = "uint16", output_hash: str | None = FULL_HASH) -> dict:
    return {"residentDtype": dtype, "fullOutputSha256": output_hash}


def test_exact_webgpu_gate_accepts_only_matching_width_and_full_output() -> None:
    evidence = _exact_run_evidence(_profile(), {"0": 1}, {"0": 1}, _args())

    assert evidence["passed"] is True
    assert evidence["residentDtype"] == "uint16"
    assert evidence["sampledFrameParity"] is True
    assert evidence["fullOutputSha256"] == FULL_HASH


@pytest.mark.parametrize("dtype", ["uint8", "uint32", "float32", None])
def test_exact_webgpu_gate_rejects_wrong_or_missing_resident_dtype(
    dtype: str | None,
) -> None:
    with pytest.raises(RuntimeError, match="resident dtype"):
        _exact_run_evidence(_profile(dtype=dtype), {"0": 1}, {"0": 1}, _args())


def test_exact_webgpu_gate_rejects_absent_or_failed_sample_parity() -> None:
    with pytest.raises(RuntimeError, match="reference is missing"):
        _exact_run_evidence(_profile(), {"0": 1}, None, _args())
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _exact_run_evidence(_profile(), {"0": 2}, {"0": 1}, _args())


def test_exact_webgpu_gate_rejects_missing_or_changed_full_output_hash() -> None:
    with pytest.raises(RuntimeError, match="did not provide"):
        _exact_run_evidence(_profile(output_hash=None), {"0": 1}, {"0": 1}, _args())
    with pytest.raises(RuntimeError, match="full-output SHA-256 mismatch"):
        _exact_run_evidence(_profile(output_hash="b" * 64), {"0": 1}, {"0": 1}, _args())
