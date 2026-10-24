"""Collect immutable loading reports and regenerate a private benchmark history.

Example: python scripts/benchmark_io_history.py run/loading.json --out private-history
Baselines remain explicit arguments to the loading runner, never inferred here.
"""

import argparse
import csv
import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _register(paths: list[Path], output: Path) -> dict:
    """Retain content-addressed reports once and serialize concurrent index updates."""
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("IO history contains private evidence; choose an --out outside the public repository.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = output / "index.json"
        index = (json.loads(index_path.read_text()) if index_path.exists() else
                 {"schema": "quantem-gpu-io-history/v1", "runs": []})
        known = {run["sha256"] for run in index["runs"]}
        records = output / "records"
        records.mkdir(exist_ok=True)
        for path in paths:
            blob = path.read_bytes()
            report = json.loads(blob)
            if report.get("schema") != "quantem-gpu-io-loading/v1":
                raise ValueError(f"{path.name} is not a loading report; use loading.json from --load-only.")
            digest = hashlib.sha256(blob).hexdigest()
            if digest in known:
                continue
            archived = records / f"{digest}.json"
            if archived.exists():
                if archived.read_bytes() != blob:
                    raise ValueError("Archived report identity mismatch; inspect the history before retrying.")
            else:
                with archived.open("xb") as stream:
                    stream.write(blob)
            index["runs"].append({
                "sha256": digest, "report": archived.relative_to(output).as_posix(),
                "source_report": str(path.resolve()),
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "recorded_at_utc": report.get("recorded_at_utc"),
                "backend": report["backend"], "device": report["device"],
                "device_identity": report["device_identity"],
                "storage_id": report["storage_id"],
                "source_fingerprint": report["source_fingerprint"],
                "correctness_passed": report.get("passed", False),
                "source_unchanged": report.get("source_unchanged"),
                "execution_status": report.get("status"),
                "error": report.get("error"),
                "performance_status": report.get("performance", {}).get("status", "incomplete"),
                "baseline": report.get("baseline"),
                "baseline_sha256": report.get("baseline_sha256"),
                "boundary": report["boundary"],
                "allocator_policy": report.get("allocator_policy", "not recorded"),
            })
            known.add(digest)
        temporary = index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(index, indent=2) + "\n")
        temporary.replace(index_path)
        _write_summaries(index, output)
    print(f"IO history: {len(index['runs'])} immutable reports in {output}")
    return index


def _write_summaries(index: dict, output: Path) -> None:
    """Regenerate comparable timing rows without choosing or replacing baselines."""
    rows = []
    text = ["# IO benchmark history", "",
            "Reports are immutable. Registration order does not choose a baseline.",
            "Disk/cache and allocator conditions must match before comparing timings.",
            "`no_baseline` is a measurement, not a performance pass. Failed and diagnostic runs remain visible.",
            "", "| Report | Backend | Overall checks | Performance | Source |",
            "|---|---|---|---|---|"]
    for run in index["runs"]:
        report = json.loads((output / run["report"]).read_text())
        text.append(f"| [{run['sha256'][:12]}]({run['report']}) | {run['backend']} | "
                    f"{run['correctness_passed']} | {run['performance_status']} | "
                    f"{run['source_fingerprint'][:12]} |")
        for case in report["cases"]:
            for format_name in ("hdf5", "qem"):
                for mode in ("cold", "warm"):
                    summary = case.get(format_name, {}).get(mode + "_summary")
                    if summary is None:
                        continue
                    rows.append({
                        "report_sha256": run["sha256"], "backend": run["backend"],
                        "recorded_at_utc": run["recorded_at_utc"],
                        "dataset": case["id"], "format": format_name, "cache": mode,
                        **summary, "performance_status": run["performance_status"],
                        "source_fingerprint": run["source_fingerprint"],
                        "device_identity": run["device_identity"], "storage_id": run["storage_id"],
                        "allocator_policy": run["allocator_policy"],
                    })
    temporary = output / "timings.csv.tmp"
    with temporary.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(output / "timings.csv")
    temporary = output / "README.md.tmp"
    temporary.write_text("\n".join(text) + "\n\nAll measured cells: [timings.csv](timings.csv).\n")
    temporary.replace(output / "README.md")


def main() -> None:
    """Register existing reports using the command documented in the module example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _register(args.reports, args.out)


if __name__ == "__main__":
    main()
