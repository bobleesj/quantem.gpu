from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _webgpu_cdp import CdpTarget as CanonicalCdpTarget
from benchmark_webgpu_h5_browser import (
    CdpTarget,
    _exact_run_evidence,
    _runtime_prelude,
    _summary,
)
from benchmark_webgpu_h5_product_browser import CdpTarget as ProductCdpTarget

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
        decode_dtype=None,
        source_scan_shape=None,
        scan_region=None,
        workers=None,
        group_size=None,
        decode_batch=None,
        frame_index_json=None,
        frame_low8="default",
        u32_shared_low8=False,
        single_parse_low8=False,
        frame_serial_low8=False,
        frames_per_wg=None,
        frame_wg=None,
        no_pipeline_staging=False,
        upload="default",
    )


def _profile(dtype: str = "uint16", output_hash: str | None = FULL_HASH) -> dict:
    return {
        "residentDtype": dtype,
        "outputRows": 512,
        "outputCols": 512,
        "outputDetRows": 192,
        "outputDetCols": 192,
        "detBin": 1,
        "fullOutputHashState": "complete",
        "fullOutputHashDomain": "corrected-logical-pixels",
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
    assert evidence["fullOutputHashDomain"] == "corrected-logical-pixels"
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

    profile = _profile()
    profile["fullOutputHashDomain"] = "raw-resident-bytes"
    with pytest.raises(RuntimeError, match="hash domain mismatch"):
        _exact_run_evidence(profile, {"0": 1}, {"0": 1}, _args())


def test_exact_webgpu_runner_forces_the_declared_decode_width() -> None:
    args = _args()

    inferred = _runtime_prelude(args)
    assert 'globalThis.__QT_H5_DECODE_DTYPE="uint16";' in inferred
    assert "globalThis.__QT_H5_FULL_OUTPUT_HASH=true;" in inferred

    args.decode_dtype = "native"
    explicit = _runtime_prelude(args)
    assert 'globalThis.__QT_H5_DECODE_DTYPE="native";' in explicit
    runner_source = (SCRIPTS / "benchmark_webgpu_h5_browser.py").read_text()
    assert "Object.assign(globalThis.__loadprof, evidence);" in runner_source


@pytest.mark.parametrize("state", [None, "ready", "pending"])
def test_exact_webgpu_gate_rejects_an_unfinished_full_output_hash(
    state: str | None,
) -> None:
    profile = _profile()
    profile["fullOutputHashState"] = state

    with pytest.raises(RuntimeError, match="did not complete"):
        _exact_run_evidence(profile, {"0": 1}, {"0": 1}, _args())


def test_exact_webgpu_gate_reports_a_failed_full_output_hash() -> None:
    profile = _profile()
    profile["fullOutputHashState"] = "failed"
    profile["fullOutputHashError"] = "device lost during readback"

    with pytest.raises(RuntimeError, match="device lost during readback"):
        _exact_run_evidence(profile, {"0": 1}, {"0": 1}, _args())


def test_webgpu_summary_reports_nearest_rank_distribution() -> None:
    runs = [
        {
            "profile": {"totalMs": value},
            "wallMs": value + 10,
            "evidenceWallMs": value + 100,
            "parity": True,
        }
        for value in (1, 2, 3, 4, 5, 6, 100)
    ]

    summary = _summary(runs)

    assert summary["percentileMethod"] == "nearest-rank"
    assert summary["totalProfileMsP50"] == 4
    assert summary["totalProfileMsP95"] == 100
    assert summary["totalProfileMsMax"] == 100
    assert summary["wallMsP50"] == 14
    assert summary["wallMsP95"] == 110
    assert summary["evidenceWallMsP50"] == 104
    assert summary["evidenceWallMsP95"] == 200


def test_cdp_call_uses_the_declared_timeout_for_long_running_evidence() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.timeout = 20.0
            self.observed_timeouts: list[float | None] = []

        def send(self, _message: str) -> None:
            return None

        def gettimeout(self) -> float | None:
            return self.timeout

        def settimeout(self, value: float | None) -> None:
            self.timeout = value
            self.observed_timeouts.append(value)

        def recv(self) -> str:
            return '{"id": 1, "result": {"result": {"value": true}}}'

    target = object.__new__(CdpTarget)
    target._next_id = 0
    target._ws = FakeWebSocket()

    result = target.call("Runtime.evaluate", timeout=900)

    assert result == {"result": {"value": True}}
    assert target._ws.observed_timeouts[0] > 899
    assert target._ws.observed_timeouts[-1] == 20


def test_webgpu_browser_runners_share_one_cdp_transport() -> None:
    assert CdpTarget is CanonicalCdpTarget
    assert ProductCdpTarget is CanonicalCdpTarget
