"""Compare stable linear-time width grouping with identical resident reductions.

All seven original acquisitions remain resident together. Complete detector
maps must match frozen independent hashes in every arm. GPU timing is not FPS.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def digest(path: Path) -> str:
    """Identify an executable or retained evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "input", "index", "plans", "reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wide-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    executable = args.exe.resolve()
    binary = digest(executable)
    resources = {
        str(path.relative_to(executable.parent)): digest(path)
        for path in executable.parent.glob("*.bundle/Resources/*.metal")
    }
    assert resources, "Missing runtime Metal resources"
    reference = [json.loads(line) for line in args.reference.read_text().splitlines()]
    expected = {
        (row["source_identity"], row["case"], row["step"]): row["sha256_u32_le"]
        for row in reference if row.get("phase") == "detector"
    }
    identities = {key[0] for key in expected}
    assert len(identities) == 7, "The frozen reference must identify seven acquisitions"
    reports = []
    arms = (("control-a", False), ("bucketed", True), ("control-b", False))
    for name, bucketed in arms:
        assert digest(executable) == binary, "Executable changed during comparison"
        assert all(digest(executable.parent / path) == sha for path, sha in resources.items())
        controls = {
            "QGPU_ORIGINAL_PROFILE": "1", "COMPACT_UPDATE_HOST_PROFILE": "1",
            "COMPACT_PLANAR_WIDTH_BUCKETS": str(int(bucketed)),
            "COMPACT_PLANAR_WIDE_ONLY": str(int(args.wide_only)),
        }
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("COMPACT_", "QGPU_"))}
        stdout = args.out / f"{name}.jsonl"
        stderr = args.out / f"{name}.stderr"
        report = {"case": name, "started_unix": time.time(), "accepted": False,
                  "binary_sha256": binary, "metal_resource_sha256": resources,
                  "environment": controls}
        with stdout.open("w") as output, stderr.open("w") as errors:
            result = subprocess.run([
                str(executable), str(args.input), str(args.index), "--series",
                "--repeats", "1", "--detector-trials", "4", "--budget-bytes",
                "17000000000", "--plan-directory", str(args.plans),
            ], env={**environment, **controls}, stdout=output, stderr=errors, timeout=600)
        report.update(exit=result.returncode, finished_unix=time.time(),
                      stdout_sha256=digest(stdout), stderr_sha256=digest(stderr))
        try:
            rows = [json.loads(line) for line in stdout.read_text().splitlines()]
            assert result.returncode == 0 and rows[-1]["phase"] == "complete"
            loads = [row for row in rows if row["phase"] == "series_load"]
            assert {row["source_identity"] for row in loads} == identities
            assert loads[-1]["resident_bytes"] == 14_752_175_312
            assert rows[-1]["released_device_allocated_bytes"] < 64 << 20
            maps = [row for row in rows if row["phase"] == "series_detector"]
            assert len(maps) == 4 * len(expected), "Missing complete image measurements"
            for trial in range(4):
                observed = {(row["source_identity"], row["case"], row["step"]):
                            row["sha256_u32_le"] for row in maps if row["trial"] == trial}
                assert observed == expected, "Complete detector image differs from reference"
            assert any(row["phase"] == "series_boundary_parity"
                       and row["empty_full_unchanged_masks"] for row in rows)
            host = []
            for line in stderr.read_text().splitlines():
                if not line.startswith("{"):
                    continue
                row = json.loads(line)
                if row.get("phase") == "detector_host_stages" and row["source_count"] == 7:
                    host.append(row)
            assert len(host) == 64, "Missing actual pipeline diagnostics"
            assert sum(row["shared_mask_preparation"] for row in host) == 49
            for row in host:
                actual = row["width_bucketed"]
                assert len(actual) == 7
                assert actual == [
                    bucketed and count >= 32 and aggregate == 0
                    for count, aggregate in zip(row["raw_entries"], row["aggregate_entries"])
                ], "Requested width grouping was not executed"
                assert row["width_bucket_threshold"] == [
                    8 if actual and args.wide_only else 0 for actual in row["width_bucketed"]
                ], "Wrong width grouping boundary"
            assert not bucketed or any(any(row["width_bucketed"]) for row in host)
            stable = {(row["trial"], row["case"], row["step"]): row
                      for row in maps if row["trial"] in (1, 2)}
            report.update(
                accepted=True, exact_maps=len(maps), resident_bytes=loads[-1]["resident_bytes"],
                released_device_allocated_bytes=rows[-1]["released_device_allocated_bytes"],
                total_load_seconds=sum(row["seconds"] for row in loads),
                call_median_ms=statistics.median(row["call_ms"] for row in stable.values()),
                gpu_median_ms=statistics.median(row["gpu_ms"] for row in stable.values()),
                width_bucketed_updates=sum(sum(row["width_bucketed"]) for row in host),
            )
        except (AssertionError, KeyError, ValueError, IndexError) as error:
            report["failure"] = str(error)
        reports.append(report)
        (args.out / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        assert report["accepted"], "Gate failed; raw evidence retained"


if __name__ == "__main__":
    main()
