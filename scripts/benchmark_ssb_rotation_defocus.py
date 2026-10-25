"""Measure full-field and regional four-case SSB comparisons on a real GPU.

Unlike benchmark-ssb, this measures all four rotation/defocus cases together,
including regional preparation, Auto reconstruction and host phase transfer.
Loading and calibration are reported separately. No browser rendering is timed.
Write --output into a private evidence directory, outside this checkout.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from quantem.gpu import SSB, io
from quantem.gpu.ssb import rotation_defocus
from quantem.gpu.ssb.results import host_array


def main() -> None:
    """Run the comparison benchmark with an explicitly calibrated acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--device", required=True, help="mps or cuda:N")
    parser.add_argument("--voltage-kv", type=float, required=True)
    parser.add_argument("--semiangle-mrad", type=float, required=True)
    parser.add_argument("--scan-sampling-a", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two repeats to separate first and warm observations.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data = io.load(args.source, device=args.device)
    data.calibration.voltage_kV = args.voltage_kv
    data.calibration.semiangle_mrad = args.semiangle_mrad
    data.calibration.scan_sampling_A = args.scan_sampling_a
    data.calibration.fit_detector_sampling()
    preparation_seconds = time.perf_counter() - start
    backend = "cuda" if args.device.startswith("cuda") else "mps"
    rows, cols = data.shape[:2]
    center = rotation_defocus._region_bounds("center", (rows, cols))
    selections = [("full", None), ("center", "center"),
                  ("top-right", "top-right"), ("coordinates", center)]
    source_file = Path(rotation_defocus.__file__)
    report = {"device": args.device, "machine": platform.machine(),
              "source": str(args.source.resolve()), "shape": list(data.shape),
              "dtype": str(data.dtype),
              "calibration": {"voltage_kV": args.voltage_kv,
                              "semiangle_mrad": args.semiangle_mrad,
                              "scan_sampling_A": args.scan_sampling_a},
              "comparison_source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
              "trials_per_case": args.trials, "cases_per_comparison": 4,
              "refinement": "newton", "preparation_seconds": preparation_seconds,
              "timing": "comparison through materialized host phase arrays; no plotting or browser",
              "observations": []}
    center_phases = None
    with SSB(data, backend=backend) as ssb:
        report["calibration"]["detector_sampling_mrad"] = ssb.detector_sampling_mrad
        original_rotation = ssb.scan_detector_rotation_deg
        original_aberrations = dict(ssb.aberrations)
        for repeat in range(args.repeats):
            for label, region in selections:
                start = time.perf_counter()
                comparison = ssb.compare_rotation_defocus(
                    region=region, trials=args.trials, refinement="newton", verbose=False,
                )
                phases = [host_array(result.phase).copy() for result in comparison.reconstructions]
                elapsed = time.perf_counter() - start
                assert ssb.scan_detector_rotation_deg == original_rotation
                assert ssb.aberrations == original_aberrations
                assert all(fit.num_trials == args.trials for fit in comparison.fits)
                if label == "center":
                    center_phases = phases
                if label == "coordinates":
                    for named, explicit in zip(center_phases, phases, strict=True):
                        np.testing.assert_allclose(named, explicit, rtol=1e-6, atol=1e-6)
                observation = {"selection": label, "repeat": repeat, "seconds": elapsed,
                               "bounds": comparison.region,
                               "native_shapes": [list(fit.phase.shape) for fit in comparison.fits],
                               "output_shapes": [list(phase.shape) for phase in phases],
                               "losses": [fit.loss for fit in comparison.fits]}
                report["observations"].append(observation)
                args.output.write_text(json.dumps(report, indent=2))
                print(json.dumps(observation), flush=True)
    summary = {}
    for label, _ in selections:
        times = [row["seconds"] for row in report["observations"] if row["selection"] == label]
        summary[label] = {"first_seconds": times[0], "warm_median_seconds": float(np.median(times[1:])),
                          "warm_p90_seconds": float(np.percentile(times[1:], 90))}
    report["summary"] = summary
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
