#!/usr/bin/env python
"""Compare exact CUDA detector updates on one and several encoded acquisitions.

Pass HDF5 or QEM files and a frozen streamed.cu with --reference-source.
Loading is measured separately. A/B/A runs reuse the same encoded allocations,
retain every scan position, and check every image against the frozen kernel.
No input paths or acquisition metadata are written to the JSON report.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from benchmark_support import _git_state, _nearest_rank_summary

from quantem.gpu import detector, io
from quantem.gpu.device.cuda_runtime import cuda_module
from quantem.gpu.resident.cuda import counts


def masks(shape):
    """Deterministic BF, displaced DF, ADF, edge and empty/full transitions."""
    rows, cols = np.indices(shape, dtype=np.float64)
    cy, cx = (np.asarray(shape) - 1) / 2
    scale = min(shape) / 192
    result = []
    for name, inner, outer, offset in (
        ("small_bf", 0, 12, (0, 0)),
        ("large_bf", 0, 70, (0, 0)),
        ("df", 0, 22, (25, 40)),
        ("adf", 40, 80, (0, 0)),
        ("edge_adf", 55, 125, (35, -30)),
    ):
        for step, (dy, dx, dr) in enumerate(
            ((0, 0, 0), (1, 2, 0), (2, 3, 1), (-25, 35, 4))
        ):
            radius2 = (rows - cy - (offset[0] + dy) * scale) ** 2
            radius2 += (cols - cx - (offset[1] + dx) * scale) ** 2
            mask = (radius2 >= (inner * scale) ** 2) & (
                radius2 < ((outer + dr) * scale) ** 2
            )
            result.append((f"{name}_{step}", mask))
    result.extend((("empty", np.zeros(shape, bool)), ("full", np.ones(shape, bool))))
    return result


def compile_kernels(source):
    names = tuple(f"sc_{name}" for name in counts._NAMES)
    compiled = cuda_module(source, names, ("--std=c++17",))
    return {name: compiled[f"sc_{name}"] for name in counts._NAMES}


def summary(records):
    result = {}
    for name in dict.fromkeys(row["mask"] for row in records):
        values = [row for row in records if row["mask"] == name]
        result[name] = {
            field: _nearest_rank_summary([row[field] / 1000 for row in values])
            for field in ("gpu_ms", "wall_ms", "host_copy_ms")
        }
    return result


def run_case(sources, variants, trajectory, reps, profile, reference_warps=None):
    sessions, preparation = [], {}
    try:
        for name, kernels in variants:
            started = time.perf_counter()
            session = detector.prepare(sources)
            sessions.append((name, session))
            session._backend.kernels = kernels
            preparation[name] = time.perf_counter() - started
        if reference_warps is not None:
            for name, session in sessions:
                if name.startswith("baseline"):
                    session._backend.residual_warps = min(
                        reference_warps,
                        int(
                            np.ceil(
                                session._backend.max_scans / session._backend.interval
                            )
                        ),
                    )
        first = sessions[0][1]
        points = sorted(
            {0, 1, 127, 128, 511, 512, first.num_frames // 2, first.num_frames - 1}
        )
        points = [point for point in points if point < first.num_frames]
        # Independent integer reductions at block boundaries and distant scans.
        raw = cp.stack(
            [
                cp.concatenate(
                    [source.data.decode_scan_range_device(p, p + 1) for p in points]
                )
                * cp.asarray(source.data.valid_pixels)
                for source in sources
            ]
        )
        expected = [
            cp.sum(raw * cp.asarray(mask), axis=(-2, -1), dtype=cp.uint64)
            for _, mask in trajectory
        ]
        del raw
        # Freeze full images before the comparison. Each variant then gets its
        # own incremental state and output buffer; order is shuffled per mask.
        reference = [first.masked_sum(mask, output="native") for _, mask in trajectory]
        outputs, observations = {}, {name: [] for name, _ in sessions}
        for name, session in sessions:
            session._backend.previous_mask = None
            outputs[name] = cp.empty(
                (*session.series_shape, *session.scan_shape), session._backend.sum_dtype
            )
        order = np.random.default_rng(0)
        for repeat in range(reps + 1):
            for index, (name, mask) in enumerate(trajectory):
                for variant in order.permutation(len(sessions)):
                    phase, session = sessions[variant]
                    output = outputs[phase]
                    if profile and phase == "candidate":
                        cp.cuda.profiler.start()
                    started = time.perf_counter()
                    session.masked_sum(mask, output="native", out=output)
                    wall_ms = (time.perf_counter() - started) * 1000
                    if profile and phase == "candidate":
                        cp.cuda.profiler.stop()
                    timing = session.finish()
                    started = time.perf_counter()
                    host = output.get()
                    host_copy_ms = (time.perf_counter() - started) * 1000
                    if not bool(cp.array_equal(output, reference[index])):
                        raise AssertionError(
                            f"Full image parity failed: {phase}, {name}, repeat {repeat}."
                        )
                    sample = output.reshape(len(sources), -1)[:, points]
                    if not bool(cp.array_equal(sample, expected[index])):
                        raise AssertionError(
                            f"Decoded-count parity failed: {phase}, {name}."
                        )
                    observations[phase].append(
                        dict(
                            mask=name,
                            repeat=repeat,
                            wall_ms=wall_ms,
                            host_copy_ms=host_copy_ms,
                            sha256=hashlib.sha256(host.tobytes()).hexdigest(),
                            **{
                                key: timing[key]
                                for key in (
                                    "gpu_ms",
                                    "residual_pixels",
                                    "spatial_fields",
                                    "incremental",
                                )
                            },
                        )
                    )
            print(
                f"{len(sources)} sources: repeat {repeat} complete; exact parity passed",
                flush=True,
            )
        return {
            "acquisitions": len(sources),
            "metadata": first.backend_metadata,
            "scan_shape": first.scan_shape,
            "detector_shape": first.detector_shape,
            "sampled_scan_indices": points,
            "preparation_seconds": preparation,
            "residual_warps": {
                name: session._backend.residual_warps for name, session in sessions
            },
            "full_image_parity": True,
            "decoded_count_parity": True,
            "ordering": "seed=0; variants interleaved and shuffled at every mask",
            "phases": [
                {
                    "phase": phase,
                    "first": [row for row in rows if row["repeat"] == 0],
                    "warm": summary([row for row in rows if row["repeat"] > 0]),
                    "observations": rows,
                }
                for phase, rows in observations.items()
            ],
        }
    finally:
        for _, session in sessions:
            session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-warps", type=int, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--contention", choices=("quiet", "contended", "unknown"), default="unknown"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Bracket candidate query kernels for Nsight Compute; use --profile-from-start off.",
    )
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.json_out.exists():
        parser.error(
            "--json-out exists; choose a new path to preserve the previous run"
        )
    frozen = args.reference_source.read_text()
    current = counts._SOURCE
    variants = [
        ("baseline", compile_kernels(frozen)),
        ("candidate", compile_kernels(current)),
        ("baseline_repeat", compile_kernels(frozen)),
    ]
    sources, loading = [], []
    report = {
        "git": _git_state(),
        "cupy": cp.__version__,
        "contention": args.contention,
        "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
            "name"
        ].decode(),
        "reference_source_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(current.encode()).hexdigest(),
        "timing_boundary": "native exact query; host copy separate; no rendering or IO",
        "loading": loading,
        "cases": [],
    }
    try:
        for number, path in enumerate(args.sources):
            started = time.perf_counter()
            source = io.load(
                path, backend="cuda", representation="encoded", verbose=False
            )
            sources.append(source)
            if type(source.data) is not counts.StreamedCounts:
                raise TypeError(
                    "This benchmark requires native uint8/uint16 ANS counts, not paired or scaled sources."
                )
            cp.cuda.get_current_stream().synchronize()
            loading.append(
                {
                    "seconds": time.perf_counter() - started,
                    "shape": source.shape,
                    "resident_bytes": source.resident_bytes,
                    "dtype": str(source.dtype),
                    "hot_pixel_correction": source.metadata.get(
                        "hot_pixel_correction", {}
                    ).get("method"),
                    "valid_pixels_sha256": hashlib.sha256(
                        source.data.valid_pixels.tobytes()
                    ).hexdigest(),
                }
            )
            print(f"Loaded {number + 1}/{len(args.sources)}: {loading[-1]}", flush=True)
        trajectory = masks(sources[0].shape[-2:])
        if args.profile:
            trajectory = trajectory[12:14]
        sizes = [len(sources)] if args.profile else sorted({1, len(sources)})
        for size in sizes:
            report["cases"].append(
                run_case(
                    sources[:size],
                    variants,
                    trajectory,
                    args.reps,
                    args.profile,
                    args.reference_warps,
                )
            )
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__}
        raise
    finally:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        for source in sources:
            source.close()


if __name__ == "__main__":
    main()
