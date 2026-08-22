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
        expected_resident_shape=(512, 512, 192, 192),
        det_bin=1,
        require_checksum_parity=True,
        require_full_output_parity=True,
        expected_full_output_sha256=FULL_HASH,
    )


def _profile(dtype: str = "uint16", output_hash: str | None = FULL_HASH) -> dict:
    return {
        "residentDtype": dtype,
        "outputRows": 512,
        "outputCols": 512,
        "outputDetRows": 192,
        "outputDetCols": 192,
        "detBin": 1,
        "logicalPixelHashSchema": "quantem.gpu.4dstem-logical-pixels/v1",
        "fullOutputSha256": output_hash,
    }


def test_exact_webgpu_gate_accepts_only_matching_width_and_full_output() -> None:
    evidence = _exact_run_evidence(_profile(), {"0": 1}, {"0": 1}, _args())

    assert evidence["passed"] is True
    assert evidence["residentDtype"] == "uint16"
    assert evidence["residentShape"] == [512, 512, 192, 192]
    assert evidence["detectorBin"] == 1
    assert evidence["sampledFrameParity"] is True
    assert evidence["logicalPixelHashSchema"] == (
        "quantem.gpu.4dstem-logical-pixels/v1"
    )
    assert evidence["fullOutputSha256"] == FULL_HASH


@pytest.mark.parametrize("dtype", ["uint8", "uint32", "float32", None])
def test_exact_webgpu_gate_rejects_wrong_or_missing_resident_dtype(
    dtype: str | None,
) -> None:
    with pytest.raises(RuntimeError, match="resident dtype"):
        _exact_run_evidence(_profile(dtype=dtype), {"0": 1}, {"0": 1}, _args())


def test_exact_webgpu_gate_requires_expected_resident_metadata() -> None:
    missing_dtype = _args()
    missing_dtype.expected_resident_dtype = None
    with pytest.raises(RuntimeError, match="expected resident dtype is missing"):
        _exact_run_evidence(_profile(), {"0": 1}, {"0": 1}, missing_dtype)

    missing_shape = _args()
    missing_shape.expected_resident_shape = None
    with pytest.raises(RuntimeError, match="expected resident shape is missing"):
        _exact_run_evidence(_profile(), {"0": 1}, {"0": 1}, missing_shape)

    missing_full_parity = _args()
    missing_full_parity.require_full_output_parity = False
    with pytest.raises(RuntimeError, match="requires full-output parity"):
        _exact_run_evidence(
            _profile(), {"0": 1}, {"0": 1}, missing_full_parity
        )


def test_exact_webgpu_gate_rejects_absent_or_failed_sample_parity() -> None:
    with pytest.raises(RuntimeError, match="reference is missing"):
        _exact_run_evidence(_profile(), {"0": 1}, None, _args())
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _exact_run_evidence(_profile(), {"0": 2}, {"0": 1}, _args())


def test_exact_webgpu_gate_rejects_wrong_shape_or_detector_bin() -> None:
    wrong_shape = _profile()
    wrong_shape["outputDetCols"] = 96
    with pytest.raises(RuntimeError, match="resident shape mismatch"):
        _exact_run_evidence(wrong_shape, {"0": 1}, {"0": 1}, _args())

    wrong_bin = _profile()
    wrong_bin["detBin"] = 2
    with pytest.raises(RuntimeError, match="detector bin mismatch"):
        _exact_run_evidence(wrong_bin, {"0": 1}, {"0": 1}, _args())


def test_exact_webgpu_gate_rejects_missing_or_changed_full_output_hash() -> None:
    missing_reference = _args()
    missing_reference.expected_full_output_sha256 = None
    with pytest.raises(RuntimeError, match="reference is missing"):
        _exact_run_evidence(
            _profile(), {"0": 1}, {"0": 1}, missing_reference
        )
    with pytest.raises(RuntimeError, match="did not provide"):
        _exact_run_evidence(_profile(output_hash=None), {"0": 1}, {"0": 1}, _args())
    with pytest.raises(RuntimeError, match="full-output SHA-256 mismatch"):
        _exact_run_evidence(_profile(output_hash="b" * 64), {"0": 1}, {"0": 1}, _args())


def test_exact_webgpu_gate_requires_canonical_logical_pixel_hash() -> None:
    profile = _profile()
    profile.pop("logicalPixelHashSchema")

    with pytest.raises(RuntimeError, match="logical pixel hash schema mismatch"):
        _exact_run_evidence(profile, {"0": 1}, {"0": 1}, _args())
