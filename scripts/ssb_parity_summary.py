#!/usr/bin/env python3
"""Print the strict SSB parity numbers-first tables from gate report JSON.

    scripts/check_ssb_parity.sh                 # writes gate-fast.json
    python scripts/ssb_parity_summary.py /path/to/gate-fast.json [more.json ...]

Every row is one compared pair at one aberration setting. The columns are the
metrics the strict gate reports: relative L2 of the object, the largest absolute
object error relative to the object scale, the largest object-phase error in
radians, and the relative error of the objective (loss).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
    ("object_relative_l2", "obj relL2"),
    ("object_max_abs_error_relative", "obj maxrel"),
    ("object_phase_max_error_radians_core", "phase max"),
    ("loss_relative_error", "loss rel"),
    ("loss_abs_error", "loss abs"),
)


def _row(label: str, metrics: dict) -> str:
    cells = " ".join(
        f"{metrics[key]:>11.4e}"
        if metrics.get(key) not in (None, float("nan"))
        else f"{'n/a':>11}"
        for key, _ in COLUMNS
    )
    return f"  {label:<34}{cells}"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    header = "".join(f"{title:>12}" for _, title in COLUMNS)
    for path in argv:
        for report in json.loads(Path(path).read_text(encoding="utf-8")):
            print(
                f"\n=== {report['case']}  ({report['bf_count']} BF / "
                f"{report['active_bf_count']} aperture-active, "
                f"{report['scan_side']}x{report['scan_side']} scan) ==="
            )
            for entry in report["aberrations"]:
                c10 = entry["aberrations"]["C10"]
                tag = "" if entry.get("gated", False) else "  [diagnostic]"
                print(f"\n  #{entry['index']}  C10={c10:.6g} nm{tag}")
                print(f"  {'compared pair':<34}{header}")
                print(
                    _row(
                        "independent float32 floor",
                        {
                            "object_relative_l2": entry["float32_floor"][
                                "object_relative_l2"
                            ],
                            "object_max_abs_error_relative": entry["float32_floor"][
                                "object_max_abs_error_relative"
                            ],
                            "object_phase_max_error_radians_core": entry[
                                "float32_floor"
                            ]["phase_max_error_radians_core"],
                            "loss_relative_error": entry["float32_floor"][
                                "loss_relative_error"
                            ],
                            "loss_abs_error": entry["float32_floor"].get(
                                "loss_absolute_error", float("nan")
                            ),
                        },
                    )
                )
                for key, metrics in sorted(entry.items()):
                    if key == "mps":
                        print(_row("mps vs oracle", metrics))
                    elif key == "metal":
                        for name, variant in sorted(metrics.items()):
                            print(_row(f"metal:{name} vs oracle", variant))
                    elif key.startswith("metal_cached_vs_"):
                        name = key.removeprefix("metal_cached_vs_")
                        print(_row(f"metal cached vs {name}", metrics))
                    elif key.startswith("mps_vs_metal_"):
                        name = key.removeprefix("mps_vs_metal_")
                        print(_row(f"mps vs metal:{name}", metrics))
                print(f"  oracle loss = {entry['reference']['loss']:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
