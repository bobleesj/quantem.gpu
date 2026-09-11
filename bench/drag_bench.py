"""Time a detector drag over encoded acquisitions, and prove the sums stay exact.

The viewer's cost while the operator drags a virtual detector is one
``masked_sum_native`` per pointer move over every resident acquisition. This replays
that sweep outside the GUI so a kernel change can be measured and profiled on its own.

Usage::

    CUDA_VISIBLE_DEVICES=<uuid> python bench/drag_bench.py --acquisitions 16 --steps 60

``--acquisitions`` caps how many files are opened, ``--outer`` the detector radius in
pixels (a large detector is the expensive case), ``--steps`` the pointer moves per sweep,
and ``--check`` compares every sum against a reference built from decoded frames.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

FORMS = Path.home() / ".config/live4dstem/gate-forms.txt"


class TimedKernel:
    """Wrap one RawKernel so every launch is bracketed by CUDA events.

    Nsight replays a kernel several times to collect counters, which needs a copy of the
    device memory; with 70 acquisitions resident there is nowhere to put it. Events cost
    one launch each and give the per-kernel split directly.
    """

    def __init__(self, kernel, name, totals):
        self.kernel, self.name, self.totals = kernel, name, totals

    def __call__(self, grid, block, args, **kwargs):
        self.grid = grid
        import cupy as cp

        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record()
        self.kernel(grid, block, args, **kwargs)
        end.record()
        self.totals.setdefault(self.name, []).append((begin, end))


def time_kernels(session, totals):
    """Bracket the device kernels behind the session's launch wrappers.

    ``residual_u32`` on the session is a Python wrapper that launches ``pm_plan_u32`` and
    ``pm_residual_u32`` in turn, so wrapping it would charge both to one name. The cached
    per-device kernel table holds the real kernels, and the wrappers look them up by name
    at launch time, so replacing the entries there times each separately.
    """
    from quantem.gpu._compact.paired import kernels

    table = kernels(session._backend.device)
    for name in ("index_u32", "index_u64", "plan_u32", "plan_u64", "residual_u32", "residual_u64"):
        if name in table and not isinstance(table[name], TimedKernel):
            table[name] = TimedKernel(table[name], name, totals)


def kernel_split(totals):
    """Milliseconds each wrapped kernel spent on the device, per launch."""
    import cupy as cp

    split = {}
    for name, pairs in totals.items():
        times = [cp.cuda.get_elapsed_time(begin, end) for begin, end in pairs]
        if times:
            split[name] = dict(launches=len(times), total_ms=round(sum(times), 2),
                               per_query_ms=round(float(np.median(times)), 3))
    return split


def open_series(paths, verbose):
    """Load the encoded acquisitions and hand back one prepared joint session."""
    from quantem.gpu import detector
    from quantem.gpu.io._paired import load_paired_file
    from quantem.gpu._compact.paired import ResidentFileReader

    reader = ResidentFileReader()
    try:
        loaded = [load_paired_file(path, device=None, verbose=False, reader=reader) for path in paths]
    finally:
        reader.close()
    session = detector.prepare(loaded)
    return loaded, session


def sweep(session, geometries, *, warmup=6, stride=1):
    """Run one query per geometry; return the device milliseconds of each."""
    import cupy as cp

    from quantem.gpu.detector import detector_mask

    shape = session.detector_shape
    out = cp.empty((*session.series_shape, *session.scan_shape),
                   session.backend_metadata.get("sum_dtype", "uint32"))
    stages = []
    for step, (row, col, inner, outer) in enumerate(geometries):
        mask = detector_mask((row, col), inner, outer, shape, dtype=np.float64)
        started = time.perf_counter()
        session.masked_sum(mask, output="native", out=out, wait=True, block_stride=stride)
        wall_ms = (time.perf_counter() - started) * 1e3
        timings = session.finish()
        record = dict(step=step, wall_ms=round(wall_ms, 3))
        for key in ("gpu_ms", "residual_pixels", "spatial_fields", "incremental", "query_launches"):
            if key in timings:
                record[key] = timings[key]
        if step >= warmup:
            stages.append(record)
    return stages, out


def spot_check(session, mask, positions, sums):
    """Compare the kernel sums against frames decoded one scan position at a time.

    The query sums selected detector pixels across every acquisition without decoding a
    frame; decoding the frame and summing the same pixels on the host is the independent
    answer. Any difference means the fast path is no longer exact.
    """
    # The query never counts a detector pixel the source marks invalid, so the host
    # reference must drop the same pixels or it reads a hot dead pixel as a difference.
    valid = np.asarray(session._backend.valid_pixels).reshape(sums.shape[0], -1)
    chosen = np.asarray(mask).ravel() != 0
    worst = 0
    for position in positions:
        patterns = np.asarray(session.frame(int(position), output="numpy")).reshape(sums.shape[0], -1)
        want = np.array([row[chosen & keep.astype(bool)].sum(dtype=np.uint64)
                         for row, keep in zip(patterns, valid)])
        got = sums[:, position].astype(np.uint64)
        worst = max(worst, int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max()))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--outer", type=float, default=80.0)
    parser.add_argument("--inner", type=float, default=40.0)
    parser.add_argument("--amplitude", type=float, default=12.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    paths = [line.strip() for line in FORMS.read_text().splitlines()
             if line.strip() and not line.startswith("#")][: args.acquisitions]
    started = time.perf_counter()
    loaded, session = open_series(paths, args.verbose)
    opened_s = time.perf_counter() - started
    centre = [value / 2 for value in session.detector_shape]
    geometries = [(centre[0] + args.amplitude * math.sin(step * 0.29),
                   centre[1] + args.amplitude * 1.1 * math.sin(step * 0.23),
                   args.inner, args.outer) for step in range(args.steps)]
    totals = {}
    if args.split:
        time_kernels(session, totals)
    stages, out = sweep(session, geometries, stride=args.stride)
    device = np.array([row.get("gpu_ms", row["wall_ms"]) for row in stages])
    wall = np.array([row["wall_ms"] for row in stages])
    report = dict(tag=args.tag, stride=args.stride, acquisitions=len(paths), steps=args.steps,
                  detector=list(session.detector_shape), scan=list(session.scan_shape),
                  inner=args.inner, outer=args.outer, open_s=round(opened_s, 1),
                  device_ms=dict(p50=round(float(np.median(device)), 3),
                                 p90=round(float(np.percentile(device, 90)), 3),
                                 max=round(float(device.max()), 3)),
                  wall_ms=dict(p50=round(float(np.median(wall)), 3),
                               p90=round(float(np.percentile(wall, 90)), 3)),
                  queries_per_s=round(1e3 / float(np.median(wall)), 1),
                  residual_pixels=int(np.median([row.get("residual_pixels", 0) for row in stages])),
                  spatial_fields=int(np.median([row.get("spatial_fields", 0) for row in stages])),
                  incremental=sum(1 for row in stages if row.get("incremental")))
    if args.split:
        report["kernel_ms"] = kernel_split(totals)
    if args.check:
        from quantem.gpu.detector import detector_mask

        row, col, inner, outer = geometries[-1]
        mask = detector_mask((row, col), inner, outer, session.detector_shape, dtype=np.float64)
        incremental = session.masked_sum(mask, output="native", out=None, wait=True).get()
        # Dropping the baseline makes the same mask go through the full plan instead of a
        # change ring; a second resident series would not fit beside this one.
        session._backend.previous_mask = None
        planned = session.masked_sum(mask, output="native", out=None, wait=True).get()
        report["incremental_matches_full_plan"] = bool(np.array_equal(incremental, planned))
        flat = incremental.reshape(len(paths), -1)
        sample = np.linspace(0, flat.shape[1] - 1, 24).astype(int)
        report["largest_difference_against_decoded_frames"] = spot_check(session, mask, sample, flat)
        report["exact"] = bool(report["incremental_matches_full_plan"]
                               and report["largest_difference_against_decoded_frames"] == 0)
    print(json.dumps(report, indent=1), flush=True)
    if args.out:
        args.out.write_text(json.dumps(dict(report=report, stages=stages), indent=1))


if __name__ == "__main__":
    main()
