"""Price a different second-stage decomposition on the de-tiled intermediate.

The row-IFFT intermediate is stored tiled (``flat = (col>>3)*4096 + (col&7) +
row*8``), so any alternative transform has to be fed the de-tiled logical
plane.  This probe compares, on identical values:

  engine k_bf=32   the frozen radix-8 column stage, one BF group
  engine k_bf=8    the same arithmetic with a different float32 reduction
                   grouping: the in-path float32 floor
  mlx ifft         MLX's own complex64 FFT along the same (row) axis
  numpy float32    pocketfft complex64, a third decomposition
  numpy float64    reference

and times each.  Nothing is landed; this is a price, not a candidate.
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

PINNED = (7.017120839737006, 0.0, -0.15393969519675116)
SECOND = (-50.0, 35.0, -0.2)
PACK_BF = 32
REPEATS = 5


def timed(fn, repeats: int = REPEATS, warmup: int = 2):
    mx = _mx()
    for _ in range(warmup):
        out = fn()
        mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
        times.append(time.perf_counter() - t0)
    return times


_MX = None


def _mx():
    global _MX
    if _MX is None:
        import mlx.core as mx  # noqa: PLC0415

        _MX = mx
    return _MX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=FIXTURE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--npz-out", type=Path, required=True)
    parser.add_argument("--label", default="2pass-price")
    args = parser.parse_args()

    mx = _mx()
    from quantem.gpu.ssb.backends.mps import engine  # noqa: PLC0415

    record: dict[str, object] = {
        "probe": "alt-decomposition",
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "load_average": list(os.getloadavg()),
        "mlx": mx.__version__,
        "pinned_loss": 0.13769753277301788,
        "pack_bf": PACK_BF,
    }
    opened, open_seconds = open_fixture(args.fixture)
    _backend, prepared, prepare_seconds = prepared_of(opened)
    record |= {
        "open_seconds": round(open_seconds, 3),
        "prepare_seconds": round(prepare_seconds, 3),
        "bandwidth_control_gbs": round(bandwidth_control(mx), 1),
        "stored_bf": int(prepared.g_qk.shape[0]),
    }

    c10 = mx.array(np.asarray([PINNED[0], SECOND[0]], dtype=np.float32))
    c12 = mx.array(np.asarray([PINNED[1], SECOND[1]], dtype=np.float32))
    phi = np.asarray([PINNED[2], SECOND[2]], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))

    tiled = engine._row_ifft512_batch_from_dynamic_geometry(
        prepared, start=0, stop=PACK_BF, c10=c10, c12=c12, cos2phi12=cos2, sin2phi12=sin2
    )
    mx.eval(tiled)
    active = mx.ones((PACK_BF,), dtype=mx.uint8)

    eng_times = timed(
        lambda: engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, tiled, k_bf=PACK_BF, active_bf=active, tiled_input=True
        )
    )
    eng_sum, eng_sumsq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, tiled, k_bf=PACK_BF, active_bf=active, tiled_input=True
    )
    mx.eval(eng_sum, eng_sumsq)
    eng_sum_k8, eng_sumsq_k8 = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, tiled, k_bf=8, active_bf=active, tiled_input=True
    )
    mx.eval(eng_sum_k8, eng_sumsq_k8)

    def detile(x):
        b, f = x.shape[0], x.shape[1]
        return mx.contiguous(
            x.reshape(b, f, 64, 512, 8).transpose(0, 1, 3, 2, 4).reshape(b, f, 512, 512)
        )

    t0 = time.perf_counter()
    logical = detile(tiled)
    mx.eval(logical)
    record["detile_seconds"] = round(time.perf_counter() - t0, 4)

    mlx_times = timed(lambda: mx.fft.ifft(logical, axis=2))

    def mlx_phases():
        plane = mx.fft.ifft(logical, axis=2)
        return mx.arctan2(plane.imag, plane.real)

    phase = mlx_phases()
    mlx_sum = mx.sum(phase, axis=1)
    mlx_sumsq = mx.sum(phase * phase, axis=(1, 2, 3))
    mx.eval(mlx_sum, mlx_sumsq)

    logical_np = np.asarray(logical)
    del tiled, logical, phase
    mx.clear_cache()

    def phases_of(z):
        return np.arctan2(z.imag, z.real)

    t0 = time.perf_counter()
    np32 = np.fft.ifft(logical_np, axis=2)
    np32_sum = np.sum(phases_of(np32), axis=1, dtype=np.float32)
    np32_sumsq = np.asarray(
        [float(np.sum(phases_of(np32)[i] ** 2, dtype=np.float32)) for i in range(np32.shape[0])]
    )
    record["numpy_float32_seconds"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    np64 = np.fft.ifft(logical_np.astype(np.complex128), axis=2)
    ph64 = phases_of(np64)
    ref_sum = np.sum(ph64, axis=1, dtype=np.float64)
    ref_sumsq = np.asarray([float(np.sum(ph64[i] ** 2)) for i in range(ph64.shape[0])])
    record["numpy_float64_seconds"] = round(time.perf_counter() - t0, 4)

    # same float32 phases, three summation orders: the pure ordering floor
    ph32_from64 = ph64.astype(np.float32)
    order_a = np.sum(ph32_from64, axis=1, dtype=np.float32)
    order_b = np.zeros_like(order_a)
    for i in range(0, ph32_from64.shape[1], 8):
        order_b += np.sum(ph32_from64[:, i : i + 8], axis=1, dtype=np.float32)
    order_c = ph32_from64[:, 0] + ph32_from64[:, 1]
    for i in range(2, ph32_from64.shape[1]):
        order_c = order_c + ph32_from64[:, i]

    record["timings"] = {
        "engine_col_k32_ms": round(float(np.median(eng_times)) * 1e3, 3),
        "mlx_ifft_axis2_ms": round(float(np.median(mlx_times)) * 1e3, 3),
    }

    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_out,
        eng_sum=np.asarray(eng_sum),
        eng_sumsq=np.asarray(eng_sumsq),
        eng_sum_k8=np.asarray(eng_sum_k8),
        eng_sumsq_k8=np.asarray(eng_sumsq_k8),
        mlx_sum=np.asarray(mlx_sum),
        mlx_sumsq=np.asarray(mlx_sumsq),
        np32_sum=np32_sum,
        np32_sumsq=np32_sumsq,
        ref_sum=ref_sum.astype(np.float64),
        ref_sumsq=ref_sumsq,
        order_a=order_a,
        order_b=order_b.astype(np.float32),
        order_c=order_c.astype(np.float32),
    )

    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
