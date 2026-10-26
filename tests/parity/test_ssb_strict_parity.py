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
      --cpu-oracle --case arina-128-full-disk --case arina-128-inner-disk --export

These tests only read that artifact. Set ``QUANTEM_SSB_PARITY_REPORT`` to reuse
an existing report JSON instead of measuring again, and
``QUANTEM_SSB_PARITY_FULL=1`` to include the full 512x512 acquisition.
Fresh CPU oracle measurements additionally require
``QUANTEM_SSB_CPU_ORACLE=1``. Normal tests may inspect saved evidence but do not
launch the expensive CPU reference automatically.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssb_parity_case import (  # noqa: E402
    ARINA_MASTER,
    CASES,
    DOCUMENTED_BF_COUNT,
    case_root,
    detector_dead_pixels,
    runs_root,
)
from ssb_parity_gate import GATES, _gate  # noqa: E402

FAST_CASES = ("arina-128-full-disk", "arina-128-inner-disk")
FULL_CASE = "arina-512-full-disk"


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
            "`scripts/check_direct_ptycho_parity.sh --cpu-oracle --export` (see this module's docstring)"
        )


def _requires_mps() -> None:
    try:
        import mlx.core  # noqa: F401
    except ImportError:  # pragma: no cover - platform dependent
        pytest.skip("MLX is unavailable; the MPS backend cannot be measured")
    if sys.platform != "darwin":
        pytest.skip("the MPS backend requires macOS")


def _existing_report() -> list[dict] | None:
    path = os.environ.get("QUANTEM_SSB_PARITY_REPORT")
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _measure(cases: tuple[str, ...], *, tag: str) -> list[dict]:
    """Run the gate for ``cases`` under the shared GPU lock and parse it."""

    if os.environ.get("QUANTEM_SSB_CPU_ORACLE") != "1":
        pytest.skip(
            "CPU oracle is opt-in; use frozen GPU parity or explicitly set "
            "QUANTEM_SSB_CPU_ORACLE=1 for diagnosis"
        )
    report_path = case_root(CASES[cases[0]]).parent / f"{tag}-report.json"
    command = [
        str(_python()),
        str(REPO_ROOT / "tests/parity/ssb_parity_gate.py"),
        "--cpu-oracle",
        *[value for name in cases for value in ("--case", name)],
        "--json",
        str(report_path),
    ]
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
    return _measure(FAST_CASES, tag="fast")


def _recorded_full_report() -> list[dict] | None:
    """Return the recorded 512x512 report, if one exists."""

    configured = os.environ.get("QUANTEM_SSB_PARITY_FULL_REPORT")
    path = Path(configured) if configured else runs_root() / "gate-512-run2.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_gate_gives_the_session_each_setting_in_nm_and_degrees() -> None:
    """A case holds ``(C10_A, C12_A, phi12_rad)``, the oracle's units; the public session takes nm and degrees.

    The gate passed the Angstrom numbers under the nm keys, so the MPS session reconstructed at ten times the
    oracle's defocus. The mapping it hands the session must return, through the session's own conversion, to the
    float32 coefficients the oracle's setting gives the kernels. Needs no acquisition.
    """
    import numpy as np

    from quantem.gpu.direct_ptycho.units import aberrations_to_engine
    from ssb_parity_gate import _aberration_dict

    for case in CASES.values():
        for index, (c10_A, c12_A, phi12_rad) in enumerate(case.aberrations_A_rad):
            public = _aberration_dict(case, index)
            assert public == {"C10_nm": c10_A / 10.0, "C12_nm": c12_A / 10.0, "phi12_deg": math.degrees(phi12_rad)}
            engine = aberrations_to_engine(public)
            np.testing.assert_array_equal(np.float32([engine["C10"], engine["C12"], engine["phi12"]]),
                                          np.float32([c10_A, c12_A, phi12_rad]), err_msg=f"{case.name}#{index}")


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


def test_mps_full_disk_matches_double_reference(fast_reports: list[dict]) -> None:
    """The public MPS path reproduces the independent oracle in float32."""

    for entry in _entries(fast_reports, "arina-128-full-disk"):
        floor = entry["float32_floor"]
        metrics = entry["mps"]
        assert metrics["num_bright_field_pixels"] == DOCUMENTED_BF_COUNT
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


def test_full_512_mps_finding_stays_open() -> None:
    """Finding F1 stays visible: MPS misses the strict bound at 512x512.

    The experiment record measures the MPS object at 4.317e-06 relL2 against the
    2.196e-06 ``2x``-floor bound at the frozen optimum, and 4.179e-06 against the
    native Metal object of the same acquisition. This test re-applies the
    *unchanged* bounds to the recorded full-size report and fails the day that
    cell stops failing: either MPS improved or the artifact moved, and either way
    the record has to be re-measured instead of this test being relaxed or
    deleted.
    """

    report = _recorded_full_report()
    if report is None:
        pytest.skip(
            "no recorded 512 report; measure one with "
            "`scripts/check_direct_ptycho_parity.sh --cpu-oracle --full` and point "
            "QUANTEM_SSB_PARITY_FULL_REPORT at it"
        )
    checked = 0
    failures: list[str] = []
    for entry in _entries(report, FULL_CASE):
        floor = entry["float32_floor"]
        metrics = entry.get("mps")
        if metrics is None:
            pytest.fail("the recorded 512 report carries no MPS row to check F1 against")
        assert metrics["num_bright_field_pixels"] == DOCUMENTED_BF_COUNT
        for metric, key in (
            ("object_relative_l2", "object_relative_l2"),
            ("object_max_abs_error_relative", "object_max_abs_error_relative"),
            ("phase_max_error_radians", "object_phase_max_error_radians_core"),
            ("loss_relative_error", "loss_relative_error"),
            ("loss_absolute_error", "loss_abs_error"),
        ):
            bound, passed = _gate(metric, float(metrics[key]), floor)
            checked += 1
            if not passed:
                failures.append(
                    f"#{entry['index']} mps {metric} = {metrics[key]:.6g} > {bound:.6g}"
                )
    assert checked >= 5, "the recorded 512 report is missing gated MPS metrics"
    assert failures, (
        "the MPS 512 finding (F1) no longer reproduces: every gated MPS metric at "
        "512x512 now sits inside its unchanged bound. That is news to record, not "
        "a test to delete - re-measure the case, refresh "
        "experiment 20260916-ssb-strict-float32-parity, and retire F1 there."
    )
