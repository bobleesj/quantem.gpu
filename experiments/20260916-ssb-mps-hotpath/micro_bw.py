"""Measure the MLX/Metal bandwidth ceiling on this Mac for byte accounting."""
from __future__ import annotations
import sys, time
import numpy as np
import mlx.core as mx

mb = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
n = mb * 1024 * 1024 // 4
a = mx.arange(n, dtype=mx.float32) % 7.0
b = mx.arange(n, dtype=mx.float32) % 11.0
mx.eval(a, b)
print(f"arrays {a.nbytes / 1e9:.3f} GB each; device={mx.default_device()}")

def bench(name, fn, bytes_moved, repeats=5):
    mx.eval(fn())
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    p50 = float(np.median(ts))
    print(f"{name:22s} p50 {p50 * 1e3:8.3f} ms  {bytes_moved / p50 / 1e9:7.1f} GB/s")
    return p50

bench("copy (r+w)", lambda: mx.array(a), 2 * a.nbytes)
bench("add (2r+1w)", lambda: a + b, 3 * a.nbytes)
bench("mul-inplace (2r+1w)", lambda: a * b, 3 * a.nbytes)
bench("sum (1r)", lambda: mx.sum(a), a.nbytes)

# isolate write bandwidth: broadcast add into fresh buffer
bench("broadcast (1r+1w)", lambda: a + 1.0, 2 * a.nbytes)

# complex64 stream, closest to the row_ifft intermediate traffic
c = a.astype(mx.complex64)
mx.eval(c)
bench("complex64 copy (r+w)", lambda: mx.array(c), 2 * c.nbytes)
