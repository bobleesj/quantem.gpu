"""Before/after benchmark of the CUDA data hot paths, with digests that prove identical values.

Run it on one Arina master with each tree under test, back to back on the same
GPU, then compare the reports::

    CUDA_VISIBLE_DEVICES=0 python tests/benchmarks/data_paths.py run --master scan_master.h5 --out before.json
    CUDA_VISIBLE_DEVICES=0 python tests/benchmarks/data_paths.py run --master scan_master.h5 --out after.json
    python tests/benchmarks/data_paths.py compare before.json after.json

Each timing is the median of ``--reps`` runs after one warm-up. The paths:

- ``load``: ``io.load(master)`` into the encoded CUDA resident, with the
  encode and index phase times it reports;
- ``decode``: GPU bitshuffle/LZ4 decode of up to four 16384-frame blocks;
- ``append``: ``StreamedCounts.append`` of a 4096-scan and of a 512-scan block;
- ``encoded``, ``dense``, ``dense immutable``, ``paired``: detector sessions
  on the encoded resident, on a dense CuPy copy of its first 65536 frames (as
  an array that may change, and declared immutable), and on the paired layout: mean pattern, BF/ADF/DF masked sums (each alternating with another
  mask so incremental planners plan in full), centre of mass, exact BF, a
  32 x 32 scan-ROI mean and one frame.

Digests (SHA-256 prefixes of the results and of the retained encoded bytes)
must agree between two reports unless a change was meant to alter values.
This is not a pytest module; it writes only the report named by ``--out``.
"""

import argparse
import hashlib
import json
import math
import statistics
import time

import cupy as cp
import numpy as np

from quantem.gpu import detector, io
from quantem.gpu.formats.hdf5.reads import FrameReader
from quantem.gpu.io.hdf5.cuda.decode import decompress_prepared
from quantem.gpu.resident.cuda.counts import StreamedCounts


def timed(function, reps):
    """Median and minimum wall time in ms of ``function`` after one warm-up, waiting for the GPU."""
    function()
    cp.cuda.Device().synchronize()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        function()
        cp.cuda.Device().synchronize()
        samples.append(1e3 * (time.perf_counter() - started))
    return {"median_ms": statistics.median(samples), "min_ms": min(samples)}


def digest(*arrays) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        host = array.get() if isinstance(array, cp.ndarray) else np.asarray(array)
        hasher.update(np.ascontiguousarray(host).tobytes())
    return hasher.hexdigest()[:16]


def payload_digest(source) -> str:
    return digest(*(array for chunk in source.chunks for array in chunk.arrays))


def bench_load(master, reps):
    walls, encode, index = [], [], []
    for repetition in range(reps + 1):
        cp.get_default_memory_pool().free_all_blocks()
        started = time.perf_counter()
        data = io.load(master, verbose=False)
        cp.cuda.Device().synchronize()
        if repetition:
            walls.append(time.perf_counter() - started)
            encode.append(data.metadata["load_timings"]["encode_seconds"])
            index.append(data.metadata["load_timings"]["index_seconds"])
        if repetition < reps:
            data.close()
    return data, {
        "wall_ms": 1e3 * statistics.median(walls),
        "encode_ms": 1e3 * statistics.median(encode),
        "index_ms": 1e3 * statistics.median(index),
        "digest": payload_digest(data.data),
    }


def bench_decode(master, reps):
    reader = FrameReader(master)
    frames = min(int(reader.source_starts[-1]), 4 * 16384)
    plans = [list(range(first, min(first + 16384, frames))) for first in range(0, frames, 16384)]
    sums = []

    def run():
        sums.clear()
        for plan in plans:
            sums.append(int(decompress_prepared(reader.prepare(np.asarray(plan))).sum(dtype=cp.uint64)))

    result = timed(run, reps)
    reader.close()
    result["digest"] = hashlib.sha256(str(sums).encode()).hexdigest()[:16]
    return result


def bench_append(data, reps):
    block = data.data.decode_scan_range_device(0, 4096)
    result = {}
    for name, raw in (("4096", block), ("512", block[:512].copy())):
        kept = {}

        def run(raw=raw):
            kept["source"] = StreamedCounts(data.shape, raw.dtype, data.data.valid_pixels)
            kept["source"].append(raw)

        result[name] = timed(run, reps)
        result[name]["digest"] = payload_digest(kept["source"])
    return result


def bench_session(data, reps, exact=True, immutable=False):
    session = detector.prepare(data, immutable=immutable)
    mean_dp = session.mean_dp()
    center, radius = detector.fit_probe(mean_dp)
    rows, cols = np.indices(session.detector_shape)
    distance = np.hypot(rows - center[0], cols - center[1])
    masks = {"bf": distance <= radius, "adf": (distance >= radius) & (distance <= 2 * radius), "df": distance >= radius}
    outputs = {}
    result = {"mean": timed(session.mean_dp, reps)}
    result["mean"]["digest"] = digest(mean_dp)
    for name in ("bf", "adf", "df"):
        other = masks["adf" if name == "bf" else "bf"]

        def run(name=name, other=other):
            session.masked_sum(other)
            outputs[name] = session.masked_sum(masks[name])

        result[name] = timed(run, reps)
        result[name]["digest"] = digest(outputs[name])
    result["com"] = timed(lambda: outputs.__setitem__("com", session.center_of_mass()), reps)
    result["com"]["digest"] = digest(*outputs["com"])
    if exact:
        result["exact_bf"] = timed(lambda: outputs.__setitem__("exact", session.masked_sum_exact(masks["bf"])), reps)
        result["exact_bf"]["digest"] = digest(outputs["exact"])
    roi_rows, roi_cols = np.meshgrid(np.arange(100, 132), np.arange(100, 132), indexing="ij")
    roi = (roi_rows * session.scan_shape[1] + roi_cols).ravel()
    result["roi_mean"] = timed(lambda: outputs.__setitem__("roi", session.reduce_frames(roi, "mean")), reps)
    result["roi_mean"]["digest"] = digest(outputs["roi"])
    index = session.num_frames // 2 + 7
    result["frame"] = timed(lambda: outputs.__setitem__("frame", session.frame(index)), reps)
    result["frame"]["digest"] = digest(outputs["frame"])
    session.close()
    return result


def run(arguments):
    report = {"master": arguments.master.rsplit("/", 1)[-1]}
    data, report["load"] = bench_load(arguments.master, arguments.reps)
    report["decode"] = bench_decode(arguments.master, arguments.reps)
    report["append"] = bench_append(data, arguments.reps)
    report["encoded"] = bench_session(data, arguments.reps)
    frames = min(65536, math.prod(data.shape[:2]))
    dense = data.data.decode_scan_range_device(0, frames).reshape(frames // 256, 256, *data.shape[2:])
    report["dense"] = bench_session(dense, arguments.reps, exact=False)
    report["dense immutable"] = bench_session(dense, arguments.reps, exact=False, immutable=True)
    del dense
    data.close()
    cp.get_default_memory_pool().free_all_blocks()
    paired = io.load(arguments.master, representation="paired", verbose=False)
    report["paired"] = bench_session(paired, arguments.reps)
    report["paired"]["layout"] = {"digest": payload_digest(paired.data)}
    paired.close()
    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=1)


def rows(report):
    """Flatten one report into ``name -> (ms, digest)``."""
    flat = {}
    for section, values in report.items():
        if not isinstance(values, dict):
            continue
        if "digest" in values:
            flat[section] = (values.get("wall_ms", values.get("median_ms")), values["digest"])
            continue
        for name, value in values.items():
            flat[f"{section} {name}"] = (value.get("median_ms"), value["digest"])
    return flat


def compare(arguments):
    reports = [rows(json.load(open(path))) for path in arguments.reports]
    print("| path | " + " | ".join(arguments.reports) + " | same values |")
    print("|---|" + "---|" * (len(reports) + 1))
    for key in reports[0]:
        cells = ["-" if key not in report or report[key][0] is None else f"{report[key][0]:.3f}" for report in reports]
        same = len({report[key][1] for report in reports if key in report}) == 1
        print(f"| {key} ms | " + " | ".join(cells) + f" | {'yes' if same else 'NO'} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("run")
    measure.add_argument("--master", required=True)
    measure.add_argument("--reps", type=int, default=7)
    measure.add_argument("--out", required=True)
    measure.set_defaults(handler=run)
    contrast = commands.add_parser("compare")
    contrast.add_argument("reports", nargs="+")
    contrast.set_defaults(handler=compare)
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
