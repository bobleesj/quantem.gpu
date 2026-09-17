"""Streaming ceiling per access mix on this host.

The 1:1 copy control alone cannot say whether a kernel that writes four bytes
per byte read (the row stage) or reads four per byte written (the column
stage) is at its own ceiling.  These controls measure read-only, write-only,
1:1 and 1:4 write-heavy streams at the intermediate's own byte scale.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def main() -> None:
    import mlx.core as mx

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--label", default="2pass-price")
    args = parser.parse_args()

    def bench(name, fn, bytes_moved, repeats=5):
        for _ in range(2):
            mx.eval(fn())
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(fn())
            times.append(time.perf_counter() - t0)
        p50 = float(np.median(times))
        return {
            "stage": name,
            "p50_ms": round(p50 * 1e3, 3),
            "bytes": int(bytes_moved),
            "gb_per_s": round(bytes_moved / p50 / 1e9, 1),
        }

    plane = 512 * 512 * 8
    big = plane * 512  # 2.147 GB, one 512-BF candidate intermediate
    complex_buf = mx.zeros((big // 8,), dtype=mx.complex64)
    float_buf = mx.arange(big // 4, dtype=mx.float32) % 3.0
    small = mx.arange(big // 16, dtype=mx.float32) % 5.0
    mx.eval(complex_buf, float_buf, small)

    record = {
        "probe": "mix-control",
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "load_average": list(os.getloadavg()),
        "mlx": mx.__version__,
        "buffer_bytes": big,
        "mix": [
            bench("copy_1r1w_2.15GB", lambda: mx.array(float_buf), 2 * big),
            bench("write_only_2.15GB", lambda: mx.zeros_like(complex_buf), big),
            bench("read_only_2.15GB", lambda: mx.sum(float_buf), big),
            bench(
                "broadcast_1r4w_2.68GB",
                lambda: mx.contiguous(mx.broadcast_to(small[None, :], (4, small.shape[0]))),
                2 * big,
            ),
        ],
    }
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
