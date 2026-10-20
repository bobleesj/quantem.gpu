"""Timing regression check of the MPS SSB and detector paths on the gold scan, and its compare rule.

The timing test is deselected by plain ``pytest``; run it on a quiet Apple GPU with::

    pytest -m benchmark tests/benchmarks

It runs only when ``-m`` names ``benchmark`` (so ``pytest -m ""`` skips it too)
and skips without an Apple GPU, the cached gold scan or a committed baseline
for this chip (see ``mps_ssb.py`` for recording one).
"""

import importlib.util
import json
from pathlib import Path

import pytest


def _harness():
    path = Path(__file__).with_name("mps_ssb.py")
    spec = importlib.util.spec_from_file_location("mps_ssb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.benchmark
def test_mps_ssb_paths_are_no_slower_than_this_chips_baseline(request):
    if "benchmark" not in request.config.getoption("markexpr", "").replace("not benchmark", ""):
        pytest.skip("timing check; select it explicitly with -m benchmark")
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlx.core")
    if not torch.backends.mps.is_available():
        pytest.skip("needs the Apple GPU (MPS)")
    harness = _harness()
    baseline_path = harness.baseline_path(harness.chip_name())
    if not baseline_path.is_file():
        pytest.skip(f"no committed baseline {baseline_path.name} for this chip")
    master = harness.gold_master()
    if master is None:
        pytest.skip("the gold scan is not in the Hugging Face cache (mps_ssb.py run --download fetches it)")

    report = harness.measure(master)
    rows, failed = harness.regressions(report, json.loads(baseline_path.read_text()))

    table = "\n".join(f"{name}: {old:.2f} -> {'-' if new is None else f'{new:.2f}'} ms {verdict}"
                      for name, old, new, _change, _tolerance, verdict in rows)
    assert report["values"]["detector_update_equals_full_sum"], "the incremental detector sum is no longer exact"
    assert not failed, f"slower than the committed baseline:\n{table}"


def test_compare_flags_only_slowdowns_beyond_tolerance_and_delta():
    harness = _harness()

    def report(**p50):
        return {"metrics": {name: {"p50_ms": value} for name, value in p50.items()}}

    baseline = report(preview_native=30.0, detector_update=1.0, io_load=5000.0, find_aberrations=25000.0)
    current = report(preview_native=37.0, detector_update=1.4, io_load=6500.0, find_aberrations=19000.0)

    rows, failed = harness.regressions(current, baseline, {"find_aberrations": 0.1})

    verdicts = {name: verdict for name, _old, _new, _change, _tolerance, verdict in rows}
    # 23 % slower than 30 ms fails; 40 % of 1 ms is under the 0.5 ms floor; io_load allows 35 %
    assert failed == ["preview_native"]
    assert verdicts["detector_update"] == verdicts["io_load"] == "ok"
    assert verdicts["find_aberrations"].startswith("faster")
    assert harness.regressions(report(preview_native=30.0), baseline)[1] == [
        "detector_update", "io_load", "find_aberrations"]
