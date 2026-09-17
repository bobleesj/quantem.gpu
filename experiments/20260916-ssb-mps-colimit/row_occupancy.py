"""Bit-exact occupancy sweep of the 512 row-IFFT stage on the MPS fit path.

Both shipped kernels in the fit path reserve exactly 32,768 B of threadgroup
memory (measured limit for this host).  The row kernel's reservation is

    shared_rows[ROWS_PER_GROUP][FUSED_CANDIDATES][512] float2

with ROWS_PER_GROUP = 4 and FUSED_CANDIDATES = 2 at batch 2, and the launch
config pins threadgroup=(64, 4, 1).  Every element of the row stage's
arithmetic -- the correction block, the three radix-8 passes, the twiddles and
the tiled store -- is a function of (row, col, bf, candidate) only.  It does
not read ROWS_PER_GROUP.  So changing ROWS_PER_GROUP changes only how many rows
share one threadgroup (and therefore how much threadgroup memory one resident
group holds), and must be bit-identical in output.

This probe: builds the row kernel at ROWS_PER_GROUP in {1, 2, 4}, proves the
output planes are bit-identical, then times each variant on a full 512-BF pack.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent.parent / "20260916-ssb-mps-hotpath"))

from profile_mps import FIXTURE, bandwidth_control, open_fixture, prepared_of  # noqa: E402

REPEATS = 5
WARMUP = 2
CHUNK = 512
GQK_COLS = 257


def stat(name, times, bytes_moved=None, extra=None):
    arr = np.asarray(times, dtype=np.float64)
    out = {
        "stage": name,
        "n": int(arr.size),
        "p50_ms": round(float(np.percentile(arr, 50)) * 1e3, 3),
        "p95_ms": round(float(np.percentile(arr, 95)) * 1e3, 3),
        "min_ms": round(float(arr.min()) * 1e3, 3),
        "max_ms": round(float(arr.max()) * 1e3, 3),
    }
    if bytes_moved:
        out["bytes"] = int(bytes_moved)
        out["gb_per_s"] = round(bytes_moved / float(np.median(arr)) / 1e9, 1)
    if extra:
        out.update(extra)
    return out


INPUT_NAMES = [
    "g", "q_row", "q_col", "kx", "ky", "pk",
    "c10", "c12", "cos2phi12", "sin2phi12", "scalars", "twiddle",
]


def _owning_module(engine):
    """The module whose globals the kernel builders actually resolve.

    ``quantem.gpu.ssb.compute.mps.engine`` re-exports the backends builders, so
    patching ``_require_mlx`` on the compute shim silently does nothing and the
    capture comes back empty.  Resolve the defining module instead.
    """
    import sys

    fn = engine._row_ifft512_dynamic_kernel
    mod_name = getattr(fn, "__module__", None)
    if mod_name and mod_name in sys.modules:
        return sys.modules[mod_name]
    return engine


def _capture_source(engine, batch, chunk, storage_bf):
    """Return the exact MSL the shipped row-kernel builder compiles."""
    import mlx.core as mx

    engine = _owning_module(engine)

    real = mx.fast.metal_kernel
    sources: dict[str, str] = {}

    def capturing(**kwargs):
        sources[kwargs.get("name", "")] = kwargs.get("source", "")
        return real(**kwargs)

    class _Fast:
        metal_kernel = staticmethod(capturing)

    class _Proxy:
        fast = _Fast()

    original = engine._require_mlx
    engine._require_mlx = lambda: _Proxy()
    try:
        engine._row_ifft512_dynamic_kernel.cache_clear()
        engine._row_ifft512_dynamic_kernel(batch, chunk, GQK_COLS, batch <= 2, storage_bf)
    finally:
        engine._require_mlx = original
    return next(iter(sources.values()))


def variant_kernel(mx, engine, rows_per_group: int, batch: int = 2, chunk: int = CHUNK):
    """Rebuild the shipped row kernel with ROWS_PER_GROUP overridden."""
    import re

    src = _capture_source(engine, batch, chunk, chunk)
    src, n = re.subn(
        r"constexpr uint ROWS_PER_GROUP = [^;]+;",
        f"constexpr uint ROWS_PER_GROUP = {int(rows_per_group)}u;",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("could not locate ROWS_PER_GROUP in generated source")
    name = f"ssb_row_ifft512_rpg{rows_per_group}_n{batch}_b{chunk}_g{GQK_COLS}_t1_s{chunk}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=list(INPUT_NAMES),
        output_names=["row_ifft"],
        source=src,
        compile_options={"math_mode": "fast"},
    )


def launch(mx, engine, kernel, prepared, rows_per_group: int, c10, c12, cos2, sin2, start, stop):
    from quantem.gpu.ssb.backends.mps import engine as eng

    chunk = int(stop) - int(start)
    batch = 2
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
            float(prepared.wavelength),
            float(prepared.semiangle_rad),
            float(prepared.ang_y_rad),
            float(prepared.ang_x_rad),
        ],
        dtype=mx.float32,
    )
    pk = eng._pk_batch_from_prepared(
        prepared, start=start, stop=stop, c10=c10, c12=c12,
        cos2phi12=cos2, sin2phi12=sin2,
    )
    return kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.q_row,
            prepared.q_col,
            prepared.kx[start:stop],
            prepared.ky[start:stop],
            pk,
            c10,
            c12,
            cos2,
            sin2,
            scalars,
            eng._twiddle_512(mx),
        ],
        template=[],
        grid=(64, 512, chunk),
        threadgroup=(64, rows_per_group, 1),
        output_shapes=[(batch, chunk, 512, 512)],
        output_dtypes=[mx.complex64],
    )[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--row-groups", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine

    rec = {
        "probe": "row-occupancy",
        "label": os.environ.get("GPU_RUN_LABEL", "unset"),
        "host": platform.node(),
        "platform": platform.platform(),
        "mlx": mx.__version__ if hasattr(mx, "__version__") else "?",
        "load_average": list(os.getloadavg()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "row_groups": args.row_groups,
        "chunk_bf": CHUNK,
        "batch": 2,
    }

    opened, _ = open_fixture(FIXTURE)
    backend, prepared, prep_s = prepared_of(opened)
    rec["prepare_seconds"] = round(prep_s, 3)
    rec["g_qk_bytes"] = int(np.prod(prepared.g_qk.shape)) * 8
    rec["g_qk_shape"] = [int(v) for v in prepared.g_qk.shape]
    rec["bandwidth_control_gbs"] = round(bandwidth_control(mx), 1)
    # The kernel declares, at engine.py:2712,
    #   threadgroup float2 shared_rows[ROWS_PER_GROUP][FUSED_CANDIDATES][512];
    # and batch==2 sets FUSED_CANDIDATES==2, so the reservation is r*2*512*8 B.
    # (An earlier revision of this probe omitted the FUSED_CANDIDATES dimension
    # and reported 8x too little; the timings were never affected.)
    rec["threadgroup_bytes"] = {
        str(r): int(2 * 512 * 8 * r) for r in args.row_groups
    }

    c10 = mx.array(np.asarray([7.017120839737006, -50.0], dtype=np.float32))
    c12 = mx.array(np.asarray([0.0, 35.0], dtype=np.float32))
    phi = np.asarray([-0.15393969519675116, -0.2], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))

    kernels = {r: variant_kernel(mx, engine, r) for r in args.row_groups}

    # ---- bit-exactness on a small slice -------------------------------------
    exact = {}
    ref = None
    for r in args.row_groups:
        out = launch(mx, engine, kernels[r], prepared, r, c10, c12, cos2, sin2, 0, 8)
        mx.eval(out)
        arr = np.asarray(out)
        if ref is None:
            ref = arr
            exact[str(r)] = {"max_abs_diff": 0.0, "bit_exact": True, "sha256": _sha(arr)}
        else:
            diff = float(np.max(np.abs(arr - ref)))
            exact[str(r)] = {
                "max_abs_diff": diff,
                "bit_exact": bool(diff == 0.0 and arr.shape == ref.shape),
                "sha256": _sha(arr),
            }
        del out, arr
    rec["bitexact_vs_rpg4"] = exact
    del ref
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    # ---- timing on a full pack ----------------------------------------------
    timings = {}
    for r in args.row_groups:
        for _ in range(WARMUP):
            mx.eval(launch(mx, engine, kernels[r], prepared, r, c10, c12, cos2, sin2, 0, CHUNK))
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            mx.eval(launch(mx, engine, kernels[r], prepared, r, c10, c12, cos2, sin2, 0, CHUNK))
            ts.append(time.perf_counter() - t0)
        # 512 BF x 2 candidates: 512 plane writes + 539 MB shared G read
        moved = 2 * CHUNK * 2097152 + CHUNK * 512 * GQK_COLS * 8
        timings[str(r)] = stat(f"row_stage_rpg{r}", ts, moved,
                               {"rows_per_group": r, "threadgroup_bytes": 2 * 512 * 8 * r})
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    rec["row_stage"] = timings

    peak = mx.get_active_memory() if hasattr(mx, "get_active_memory") else None
    rec["peak_active_bytes"] = int(peak) if peak is not None else None
    rec["load_average_end"] = list(os.getloadavg())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def _sha(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


if __name__ == "__main__":
    main()
