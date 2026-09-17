"""Decide device_memory-bound vs compute-bound for the row stage without aliasing.

The earlier intermediate pricing used a store-folded variant that redirected
every plane write into four addresses.  That removes the device_memory write traffic but
turns the store pipe into an aliasing hazard, so its 1.71 ms "saving" cannot be
read as a device_memory share.  This probe keeps the store instruction count, the
addressing pattern and the whole ALU identical, and halves only the number of
bytes each store writes, by emitting the plane as float32 (real part) instead of
complex64.  Both planes have the same 512x512x512x2 elements in the same tiled
order, so nothing aliases.

If the stage is device_memory-bound the float32 store should recover about
1,073,741,824 B / ceiling = 8.1 ms; if the stage is compute-bound it should
recover almost nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent.parent / "20260916-ssb-mps-hotpath"))

from profile_mps import FIXTURE, bandwidth_control, open_fixture, prepared_of  # noqa: E402
from row_occupancy import CHUNK, GQK_COLS, INPUT_NAMES, _capture_source, stat  # noqa: E402

WARMUP = 2
PINNED = (7.017120839737006, 0.0, -0.15393969519675116)
SECOND = (-50.0, 35.0, -0.2)


def float32_store_variant(mx, engine):
    """Ship the same tiled plane, but as float32 real values."""
    src = _capture_source(engine, 2, CHUNK, CHUNK)
    real_pat = re.compile(r"(\s*)row_ifft\[([^\]]+)\]\.real = ([^;]+);")
    imag_pat = re.compile(r"\s*row_ifft\[[^\]]+\]\.imag = [^;]+;\n")
    src, n_real = real_pat.subn(r"\1row_ifft[\2] = \3;", src)
    src, n_imag = imag_pat.subn("\n", src)
    if n_real == 0 or n_imag == 0:
        raise RuntimeError(
            f"store rewrite failed (real={n_real}, imag={n_imag})")
    leftover_pat = re.compile(r"row_ifft\[[^\]]+\]\.(real|imag)")
    leftover = leftover_pat.findall(src)
    if leftover:
        raise RuntimeError(f"leftover complex store lines: {leftover[:3]}")
    kernel = mx.fast.metal_kernel(
        name="ssb_row_ifft512_f32store",
        input_names=list(INPUT_NAMES),
        output_names=["row_ifft"],
        source=src,
        compile_options={"math_mode": "fast"},
    )
    return kernel, {"real_stores_rewritten": n_real, "imag_stores_dropped": n_imag}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine

    rec = {
        "probe": "row-store-width",
        "label": os.environ.get("GPU_RUN_LABEL", "unset"),
        "host": platform.node(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    opened, _ = open_fixture(FIXTURE)
    backend, prepared, _ = prepared_of(opened)
    ceiling = bandwidth_control(mx)
    rec["bandwidth_control_gbs"] = round(ceiling, 1)

    c10 = mx.array(np.asarray([PINNED[0], SECOND[0]], dtype=np.float32))
    c12 = mx.array(np.asarray([PINNED[1], SECOND[1]], dtype=np.float32))
    phi = np.asarray([PINNED[2], SECOND[2]], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))

    scalars = mx.array(
        [float(prepared.factor), float(prepared.dc_value.real),
         float(prepared.dc_value.imag), float(prepared.wavelength),
         float(prepared.semiangle_rad), float(prepared.ang_y_rad),
         float(prepared.ang_x_rad)], dtype=mx.float32)

    pk = engine._pk_batch_from_prepared(
        prepared, start=0, stop=CHUNK, c10=c10, c12=c12,
        cos2phi12=cos2, sin2phi12=sin2)
    twiddle = engine._twiddle_512(mx)

    def make_launcher(out_dtype, out_shape):
        def call(kernel):
            return kernel(
                inputs=[prepared.g_qk[:CHUNK], prepared.q_row, prepared.q_col,
                        prepared.kx[:CHUNK], prepared.ky[:CHUNK], pk,
                        c10, c12, cos2, sin2, scalars, twiddle],
                template=[],
                grid=(64, 512, CHUNK),
                threadgroup=(64, 4, 1),
                output_shapes=[out_shape],
                output_dtypes=[out_dtype],
            )[0]
        return call

    # shipped kernel, shipped layout
    shipped = engine._row_ifft512_dynamic_kernel(2, CHUNK, GQK_COLS, True, CHUNK)
    call_c64 = make_launcher(mx.complex64, (2, CHUNK, 512, 512))

    f32, meta = float32_store_variant(mx, engine)
    rec["variant"] = meta
    call_f32 = make_launcher(mx.float32, (2, CHUNK, 512, 512))

    c64_bytes = 2 * CHUNK * 512 * 512 * 8 + CHUNK * 512 * GQK_COLS * 8
    f32_bytes = 2 * CHUNK * 512 * 512 * 4 + CHUNK * 512 * GQK_COLS * 8

    samples = {"complex64": [], "float32": []}
    order = ["complex64", "float32", "float32", "complex64"]
    for name in order:
        call = call_c64 if name == "complex64" else call_f32
        kernel = shipped if name == "complex64" else f32
        for _ in range(WARMUP):
            mx.eval(call(kernel))
        t0 = time.perf_counter()
        mx.eval(call(kernel))
        samples[name].append(time.perf_counter() - t0)

    rec["complex64_store"] = stat("row_c64", samples["complex64"], c64_bytes)
    rec["float32_store"] = stat("row_f32", samples["float32"], f32_bytes)
    saved_ms = (float(np.median(samples["complex64"]))
                - float(np.median(samples["float32"]))) * 1e3
    predicted = 1073741824 / ceiling / 1e6
    rec["verdict"] = {
        "bytes_removed": 1073741824,
        "saved_ms": round(saved_ms, 3),
        "predicted_if_device_memory_bound_ms": round(predicted, 3),
        "fraction_of_prediction": round(saved_ms / predicted, 3),
        "read": ("saved_ms close to the prediction means the row stage is "
                 "device_memory-bound on its write traffic; near zero means it is not."),
    }
    rec["order"] = order

    peak = mx.get_active_memory() if hasattr(mx, "get_active_memory") else None
    rec["peak_active_bytes"] = int(peak) if peak is not None else None
    rec["load_average_end"] = list(os.getloadavg())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
