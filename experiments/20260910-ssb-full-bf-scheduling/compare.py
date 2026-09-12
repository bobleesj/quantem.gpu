"""Compare identical original-file workloads; do not substitute UI frame rates."""

import json
import sys
from pathlib import Path

import numpy as np


def timings(root: Path) -> dict:
    """Summarize repeated warm calls separately for object and loss."""
    data = json.loads((root / "report.json").read_text())
    result = {key: data.get(key) for key in
              ("load_seconds", "prepare_seconds", "sampled_allocated_bytes",
               "fit_seconds", "fit_loss", "refinement_evaluations", "fit_best")}
    for key in ("redraws", "losses"):
        values = np.array([entry["gpu_seconds"] * 1000 for entry in data[key]
                           if entry["repetition"] > 0])
        result[key] = dict(zip(("mean_ms", "p50_ms", "p95_ms"),
                               [float(values.mean()), *np.percentile(values, [50, 95]).tolist()]))
    return result


before, after = map(Path, sys.argv[1:3])
result = {"before": timings(before), "after": timings(after), "parity": []}
for index in range(3):
    reference = np.fromfile(before / f"object-{index}.c64", dtype=np.complex64)
    candidate = np.fromfile(after / f"object-{index}.c64", dtype=np.complex64)
    relative = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
    phase = float(np.max(np.abs(np.angle(candidate * reference.conj()))))
    result["parity"].append({"case": index, "relative_l2": relative,
                             "maximum_phase_error_rad": phase})
    assert relative < 1e-5, result
baseline_loss = json.loads((before / "report.json").read_text())["losses"]
candidate_loss = json.loads((after / "report.json").read_text())["losses"]
assert len(baseline_loss) == len(candidate_loss)
error = max(abs(a["loss"] - b["loss"]) for a, b in zip(baseline_loss, candidate_loss))
result["maximum_loss_absolute_error"] = error
assert error < 1e-6, result
print(json.dumps(result, indent=2))
