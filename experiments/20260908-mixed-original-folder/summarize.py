"""Produce a path-redacted receipt from native mixed-folder test evidence.

Usage: python summarize.py /path/to/validation-root
The original logs and screenshots stay local; this summary does not replace them.
"""
import hashlib
import json
from pathlib import Path
import sys


def _receipt(path: Path) -> dict:
    """Fingerprint a complete retained evidence file."""
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _main() -> None:
    root = Path(sys.argv[1])
    folder = root / "mixed-original-folder-20260908"
    result = json.loads((folder / "drive.json").read_text())
    states = []
    for line in (folder / "gui-benchmark.log").read_text().splitlines():
        if line.startswith("UI_ACCEPTANCE_STATE "):
            states.append(json.loads(line.split(" ", 1)[1]))
    phases = {}
    for record in result["first_presented_records"]:
        phases.setdefault(record["phase"], []).append(float(record["wall_seconds"]))
    navigation = {key: value for key, value in result["navigation_receipts"].items()
                  if key != "commands"}
    output = {"scope": "Native controller hooks, not physical pointer/Finder acceptance",
              "folder_acquisitions": result["steps"][0]["datasets"],
              "problems": result["problems"], "exit": result["exit"],
              "navigation": navigation, "presentation_seconds_by_phase": phases,
              "poll_observed_peak_allocated_bytes": max(s.get("allocated_bytes", 0) for s in states),
              "final": result["final"], "single_selected_arina_interaction": result["analysis"],
              "retained_drive_json": _receipt(folder / "drive.json"),
              "retained_opened_screenshot": _receipt(folder / "opened.png"),
              "source_transfer_concurrent": True, "cold_io_claim": False,
              "numpy_native_load": "not implemented; excluded from supported acquisition count",
              "empad_journeys": {}}
    for material, directory in [
        ("mos2-mose2", "mixed-original-mos2-retest-20260908"),
        ("snse", "mixed-original-snse-20260908"),
        ("pdpt", "mixed-original-pdpt-20260908"),
    ]:
        path = root / directory / "result.json"
        journey = json.loads(path.read_text())
        output["empad_journeys"][material] = {
            "failures": journey["failures"], "exit": journey["exit"],
            "steps": [s for s in journey["steps"] if "dataset_id" not in s],
            "first_presented_seconds": [float(p["wall_seconds"]) for p in journey["first_presented"]],
            "retained_result_json": _receipt(path)}
    failed = root / "mixed-original-mos2-20260908/result.json"
    output["invalid_test_attempt"] = {
        "retained_result_json": _receipt(failed),
        "reason": "A single ARINA file was passed where the test requires a multiple-acquisition folder; rerun used the folder without changing assertions",
        "claimed_as_pass": False}
    output["original_arina_file_sha256"] = [
        line.split()[0] for line in (folder / "arina-source-sha256.txt").read_text().splitlines()]
    comparison_path = root / "mixed-original-compare-20260908/drive.json"
    comparison = json.loads(comparison_path.read_text())
    output["repeat_and_seven_comparison"] = {
        "problems": comparison["problems"], "exit": comparison["exit"],
        "navigation": {k: v for k, v in comparison["navigation_receipts"].items()
                       if k != "commands"},
        "final": comparison["final"], "analysis": comparison["analysis"],
        "compare_ready": [s for s in comparison["steps"] if s["name"] == "compare"],
        "timing_caveat": "Comparison reuses retained residents; ready time is not loading all seven from disk",
        "first_ten_full_presented_seconds": [float(p["wall_seconds"])
                                              for p in comparison["first_presented_records"][:10]],
        "retained_drive_json": _receipt(comparison_path)}
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    _main()
