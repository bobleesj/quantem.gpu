"""Strict float32/complex64 SSB parity gate.

Every claim below is one statement about one measured pair, checked against the
independent double-precision oracle in :mod:`ssb_double_reference`. The gate
never rewrites a tolerance to make a backend pass: the bounds live in
:data:`ssb_parity_gate.GATES` with their float32 derivation, and each measured
single-precision floor is reported alongside the value it bounds.

The exact-input artifacts are expensive to build (they decode the real ARINA
acquisition), so they are generated once, outside the repository, by::

    GPU_RUN_LABEL=parity ~/perf-lab/ssb-audit/gpurun \\
      PYTHONPATH=src python tests/parity/ssb_parity_gate.py \\
      --case arina-128-full-disk --case arina-128-inner-disk --export

These tests only read that artifact. Set ``QUANTEM_SSB_PARITY_REPORT`` to reuse
an existing report JSON instead of measuring again, and
``QUANTEM_SSB_PARITY_FULL=1`` to include the full 512x512 acquisition.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssb_parity_case import (  # noqa: E402
    ARINA_MASTER,
    CASES,
    DOCUMENTED_BF_COUNT,
    case_root,
    detector_dead_pixels,
)
from ssb_parity_gate import GATES, _gate  # noqa: E402

FAST_CASES = ("arina-128-full-disk", "arina-128-inner-disk")
RECORDED_CASE = "arina-128-recorded-c10"
FULL_CASE = "arina-512-full-disk"
HARNESS_MARKER = REPO_ROOT / "build" / "ssb-parity-check.json"


def _gpurun() -> Path:
    return Path(os.environ.get("QUANTEM_SSB_PARITY_GPURUN", "~/perf-lab/ssb-audit/gpurun")).expanduser()


def _python() -> str:
    return os.environ.get("QUANTEM_SSB_PARITY_PYTHON", str(Path.home() / "miniforge3/bin/python3.12"))


def _requires_source() -> None:
    if not ARINA_MASTER.is_file():
        pytest.skip(f"real ARINA acquisition is absent: {ARINA_MASTER}")


def _requires_artifact(case_name: str) -> None:
    if not (case_root(CASES[case_name]) / "case.json").is_file():
        pytest.skip(
            f"case {case_name} is not exported yet; run "
            "`scripts/check_ssb_parity.sh --export` (see this module's docstring)"
        )


def _requires_mps() -> None:
    try:
        import mlx.core  # noqa: F401
    except ImportError:  # pragma: no cover - platform dependent
        pytest.skip("MLX is unavailable; the MPS backend cannot be measured")
    if sys.platform != "darwin":
        pytest.skip("the MPS backend requires macOS")


def _requires_metal_harness() -> None:
    if not HARNESS_MARKER.is_file():
        pytest.skip(
            "the native parity harness is not built; run "
            "`scripts/check_ssb_parity.sh --build-metal`"
        )


def _existing_report() -> list[dict] | None:
    path = os.environ.get("QUANTEM_SSB_PARITY_REPORT")
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _measure(cases: tuple[str, ...], *, use_metal: bool, tag: str) -> list[dict]:
    """Run the gate for ``cases`` under the shared GPU lock and parse it."""

    report_path = case_root(CASES[cases[0]]).parent / f"{tag}-report.json"
    command = [
        str(_python()),
        str(REPO_ROOT / "tests/parity/ssb_parity_gate.py"),
        *[value for name in cases for value in ("--case", name)],
        "--json",
        str(report_path),
    ]
    if not use_metal:
        command.append("--no-metal")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["GPU_RUN_LABEL"] = "parity"
    launcher: list[str] = []
    gpurun = _gpurun()
    if gpurun.is_file() and os.access(gpurun, os.X_OK):
        launcher = [str(gpurun)]
    completed = subprocess.run(
        [*launcher, *command],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "the parity gate command failed; this is a finding, not a test to "
            f"relax.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fast_reports() -> list[dict]:
    """Measured reports for the 128x128 real-data cases."""

    _requires_source()
    for name in FAST_CASES:
        _requires_artifact(name)
    cached = _existing_report()
    if cached is not None:
        return cached
    _requires_mps()
    use_metal = HARNESS_MARKER.is_file()
    return _measure(FAST_CASES, use_metal=use_metal, tag="fast")


@pytest.fixture(scope="session")
def full_reports() -> list[dict]:
    """Measured report for the full 512x512 acquisition (opt-in)."""

    if not os.environ.get("QUANTEM_SSB_PARITY_FULL"):
        pytest.skip("set QUANTEM_SSB_PARITY_FULL=1 to measure the 512x512 case")
    _requires_source()
    _requires_artifact(FULL_CASE)
    _requires_mps()
    return _measure((FULL_CASE,), use_metal=HARNESS_MARKER.is_file(), tag="full")


def _entries(reports: list[dict], case_name: str) -> list[dict]:
    for report in reports:
        if report["case"] == case_name:
            return [entry for entry in report["aberrations"] if entry["gated"]]
    pytest.fail(f"case {case_name} is missing from the report")


def _assert_gate(label: str, metric: str, value: float, floor: dict) -> None:
    bound, passed = _gate(metric, float(value), floor)
    assert passed, (
        f"{label}: {metric} = {value:.6g} exceeds {bound:.6g} "
        f"(absolute bound {GATES[metric]['absolute']:.6g}, "
        f"{GATES[metric]['floor_multiple']}x measured float32 floor). "
        "Do not raise the bound; investigate the precision loss."
    )


def test_bf_policy_matches_documented_count() -> None:
    """The pinned disk reproduces the documented 8,937 logical BF pixels."""

    _requires_source()
    case = CASES["arina-128-full-disk"]
    rows, cols = case.bf_rows_cols
    dead = detector_dead_pixels(case.source)
    assert dead.shape[0] == 4, "the ARINA master declares four hardware dead pixels"
    assert (78, 74) in [tuple(value) for value in dead.tolist()]
    assert rows.size == DOCUMENTED_BF_COUNT, (
        "the documented automatic full disk is 8,938 geometric pixels minus "
        f"the one dead pixel inside it; got {rows.size}"
    )
    selected = set(zip(rows.tolist(), cols.tolist()))
    assert not selected & set(map(tuple, dead.tolist())), (
        "a hardware dead pixel is inside the pinned bright-field set"
    )


def test_exported_counts_are_exact_raw_uint16() -> None:
    """The measured inputs are the acquisition's exact raw counts.

    The export records the result of an independent h5py read of the
    acquisition that bypasses every ``quantem.gpu`` loader, hot-pixel
    correction and dtype conversion, and compares it value by value with the
    payload the backends measure. A failure here means the gate is measuring
    something other than the raw detector counts.
    """

    for name in FAST_CASES:
        _requires_artifact(name)
        declaration = json.loads(
            (case_root(CASES[name]) / "case.json").read_text(encoding="utf-8")
        )
        exactness = declaration.get("counts_exactness")
        assert exactness is not None, (
            f"{name} was exported without the independent raw-count "
            "verification; re-export with --force-export"
        )
        assert exactness["exact"], (
            f"{name}: {exactness['mismatching_scan_positions']} scan positions "
            f"differ from the raw acquisition read ({exactness['read']})"
        )
        assert exactness["values_compared"] == (
            declaration["scan_shape"][0] * declaration["scan_shape"][1]
            * declaration["bf_count"]
        )
        assert declaration["counts_dtype"] == "uint16"


def test_gated_aberration_settings_are_well_conditioned(fast_reports: list[dict]) -> None:
    """Every gated setting has a small measured single-precision floor."""

    for report in fast_reports:
        for entry in _entries(fast_reports, report["case"]):
            floor = entry["float32_floor"]
            assert floor["object_relative_l2"] < 1.0e-5, (
                f"{report['case']}#{entry['index']} is not a usable float32 "
                f"parity point: floor {floor['object_relative_l2']:.3g}"
            )
        diagnostics = [item for item in report["aberrations"] if not item["gated"]]
        for entry in diagnostics:
            assert entry["float32_floor"]["object_relative_l2"] > 1.0e-3, (
                "the retained degenerate start setting is expected to stay "
                "ill-conditioned; if it is now well behaved, re-derive the "
                "diagnostic note instead of deleting it silently"
            )


def test_recorded_c10_settings_no_longer_disagree(fast_reports: list[dict]) -> None:
    """The recorded cached-versus-streamed loss defect stays closed.

    ``experiments/20260913-ssb-loss-diagnosis`` recorded the cached objective
    disagreeing with the streamed one by 4.43e-5, 3.46e-5 and 5.79e-5 relative
    at C10 = 0, 55 and 155.96977 (the last exceeded the 5e-5 gate of the time).
    The defect was the half-plane projection dropping the signed-Nyquist
    contribution. Both paths must now agree with each other and with the
    full-plane oracle at float32 level.
    """

    for entry in _entries(fast_reports, RECORDED_CASE):
        floor = entry["float32_floor"]
        assert floor["object_relative_l2"] < 1.0e-5, (
            f"recorded C10={entry['aberrations']['C10']} is not a usable "
            f"float32 parity point: floor {floor['object_relative_l2']:.3g}"
        )
        cached = entry.get("metal_cached_vs_streamed")
        assert cached is not None, "no cached-versus-streamed record was measured"
        _assert_gate(
            "recorded cached vs streamed",
            "loss_relative_error",
            cached["loss_relative_error"],
            floor,
        )
        _assert_gate(
            "recorded cached vs streamed",
            "object_relative_l2",
            cached["object_relative_l2"],
            floor,
        )
        _assert_gate(
            "recorded cached vs streamed",
            "loss_absolute_error",
            cached["loss_abs_error"],
            floor,
        )
        for variant, metrics in (entry.get("metal") or {}).items():
            _assert_gate(
                f"recorded metal {variant}",
                "loss_relative_error",
                metrics["loss_relative_error"],
                floor,
            )
            _assert_gate(
                f"recorded metal {variant}",
                "loss_absolute_error",
                metrics["loss_abs_error"],
                floor,
            )


def test_mps_full_disk_matches_double_reference(fast_reports: list[dict]) -> None:
    """The public MPS path reproduces the independent oracle in float32."""

    for entry in _entries(fast_reports, "arina-128-full-disk"):
        floor = entry["float32_floor"]
        metrics = entry["mps"]
        assert metrics["num_bf"] == DOCUMENTED_BF_COUNT
        _assert_gate("mps object", "object_relative_l2", metrics["object_relative_l2"], floor)
        _assert_gate(
            "mps object",
            "object_max_abs_error_relative",
            metrics["object_max_abs_error_relative"],
            floor,
        )
        _assert_gate(
            "mps object phase",
            "phase_max_error_radians",
            metrics["object_phase_max_error_radians_core"],
            floor,
        )
        _assert_gate(
            "mps objective", "loss_relative_error", metrics["loss_relative_error"], floor
        )
        _assert_gate(
            "mps objective", "loss_absolute_error", metrics["loss_abs_error"], floor
        )
        _assert_gate(
            "mps objective mean phase",
            "phase_max_error_radians",
            metrics["mean_phase_max_error_radians_core"],
            floor,
        )


def test_mps_inner_disk_matches_double_reference(fast_reports: list[dict]) -> None:
    """A fully aperture-active disk matches the oracle without inactive terms."""

    for entry in _entries(fast_reports, "arina-128-inner-disk"):
        floor = entry["float32_floor"]
        metrics = entry["mps"]
        assert entry["reference"]["inactive_bf"] == 0
        _assert_gate("mps object", "object_relative_l2", metrics["object_relative_l2"], floor)
        _assert_gate(
            "mps objective", "loss_relative_error", metrics["loss_relative_error"], floor
        )
        _assert_gate(
            "mps objective", "loss_absolute_error", metrics["loss_abs_error"], floor
        )


def test_native_metal_matches_double_reference(fast_reports: list[dict]) -> None:
    """Every native Metal cache topology reproduces the oracle in float32."""

    _requires_metal_harness()
    for name in FAST_CASES:
        for entry in _entries(fast_reports, name):
            floor = entry["float32_floor"]
            variants = entry.get("metal")
            if not variants:
                pytest.fail(f"case {name} has no native Metal products in the report")
            for variant, metrics in variants.items():
                label = f"metal {variant}"
                _assert_gate(label, "object_relative_l2", metrics["object_relative_l2"], floor)
                _assert_gate(
                    label,
                    "object_max_abs_error_relative",
                    metrics["object_max_abs_error_relative"],
                    floor,
                )
                _assert_gate(
                    label,
                    "phase_max_error_radians",
                    metrics["object_phase_max_error_radians_core"],
                    floor,
                )
                _assert_gate(
                    label, "loss_relative_error", metrics["loss_relative_error"], floor
                )
                _assert_gate(
                    label, "loss_absolute_error", metrics["loss_abs_error"], floor
                )


def test_metal_cache_topology_does_not_change_results(fast_reports: list[dict]) -> None:
    """Cached, hybrid and fully streamed topologies agree at float32 level.

    This is the regression claim for the retired half-plane endpoint defect,
    which broke the cached objective by 1.06e-3 relative against a 5e-5 gate.
    """

    _requires_metal_harness()
    for name in FAST_CASES:
        for entry in _entries(fast_reports, name):
            floor = entry["float32_floor"]
            variants = entry.get("metal") or {}
            assert variants, f"case {name} has no native Metal products"
            cached = variants["cached"]
            for variant in ("hybrid", "streamed"):
                other = variants[variant]
                relative = abs(other["loss"] - cached["loss"]) / abs(cached["loss"])
                assert relative <= GATES["loss_relative_error"]["absolute"], (
                    f"{name}: metal {variant} loss differs from cached by "
                    f"{relative:.3g} relative; the endpoint projection defect "
                    "is back."
                )
                pair = entry.get(f"metal_cached_vs_{variant}")
                assert pair is not None, (
                    f"case {name} has no cached-vs-{variant} topology record"
                )
                _assert_gate(
                    f"metal cached vs {variant}",
                    "object_relative_l2",
                    pair["object_relative_l2"],
                    floor,
                )
                _assert_gate(
                    f"metal cached vs {variant}",
                    "loss_relative_error",
                    pair["loss_relative_error"],
                    floor,
                )
                _assert_gate(
                    f"metal cached vs {variant}",
                    "phase_max_error_radians",
                    pair["object_phase_max_error_radians_core"],
                    floor,
                )


def test_mps_matches_native_metal_directly(fast_reports: list[dict]) -> None:
    """The two production backends agree with each other in float32."""

    _requires_metal_harness()
    for name in FAST_CASES:
        for entry in _entries(fast_reports, name):
            floor = entry["float32_floor"]
            for variant in ("cached", "hybrid", "streamed"):
                pair = entry.get(f"mps_vs_metal_{variant}")
                assert pair is not None, f"case {name} has no mps-vs-metal:{variant}"
                _assert_gate(
                    f"mps vs metal {variant}",
                    "object_relative_l2",
                    pair["object_relative_l2"],
                    floor,
                )
                _assert_gate(
                    f"mps vs metal {variant}",
                    "phase_max_error_radians",
                    pair["object_phase_max_error_radians_core"],
                    floor,
                )
                _assert_gate(
                    f"mps vs metal {variant}",
                    "loss_relative_error",
                    pair["loss_relative_error"],
                    floor,
                )
                _assert_gate(
                    f"mps vs metal {variant}",
                    "loss_absolute_error",
                    pair["loss_abs_error"],
                    floor,
                )


def test_full_512_matches_double_reference(full_reports: list[dict]) -> None:
    """The production 512x512 scale keeps float32 parity."""

    for entry in _entries(full_reports, FULL_CASE):
        floor = entry["float32_floor"]
        metrics = entry["mps"]
        assert metrics["num_bf"] == DOCUMENTED_BF_COUNT
        _assert_gate("mps object", "object_relative_l2", metrics["object_relative_l2"], floor)
        _assert_gate(
            "mps object phase",
            "phase_max_error_radians",
            metrics["object_phase_max_error_radians_core"],
            floor,
        )
        _assert_gate(
            "mps objective", "loss_relative_error", metrics["loss_relative_error"], floor
        )
        _assert_gate(
            "mps objective", "loss_absolute_error", metrics["loss_abs_error"], floor
        )
        for variant, metal in (entry.get("metal") or {}).items():
            _assert_gate(
                f"metal {variant}", "object_relative_l2", metal["object_relative_l2"], floor
            )
            _assert_gate(
                f"metal {variant}", "loss_relative_error", metal["loss_relative_error"], floor
            )

