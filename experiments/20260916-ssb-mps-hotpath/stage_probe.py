"""Attribute the MLX/MPS SSB hot path to memory, geometry math, and sync.

Opens the calibrated 512 fixture once, then measures, in one locked GPU
session: the machine streaming ceiling, the exact-pair objective, the two
fused 512 stages in isolation, kernel-source variants that remove single
work components (timing-only ablations), and the scalar phase/loss path with
raw bit hashes for cross-engine A/B.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("SSB_SRC", REL / "src"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import (  # noqa: E402
    FIXTURE,
    SyncCounter,
    bandwidth_control,
    open_fixture,
    prepared_of,
    sha256_array,
)

PARAMS = [
    (7.017120839737006, 0.0, -0.15393969519675116),
    (-50.0, 35.0, -0.2),
    (0.0, 0.0, 0.0),
    (12.5, -7.25, 1.75),
]
WARMUP = 2
REPEATS = 5


def stat(times):
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n": int(arr.size),
        "p50_ms": round(float(np.percentile(arr, 50)) * 1e3, 3),
        "p95_ms": round(float(np.percentile(arr, 95)) * 1e3, 3),
        "min_ms": round(float(arr.min()) * 1e3, 3),
        "max_ms": round(float(arr.max()) * 1e3, 3),
        "seconds": [round(float(v), 6) for v in arr],
    }


def timed(fn, repeats=REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return stat(times)


def capture(mx, builder, *args, **kwargs):
    """Return the (name, source, header) a cached kernel builder would compile."""
    captured = {}
    real = mx.fast.metal_kernel

    def spy(*a, **kw):
        captured["name"] = kw.get("name", a[0] if a else "?")
        captured["source"] = kw.get("source")
        captured["header"] = kw.get("header")
        return real(*a, **kw)

    mx.fast.metal_kernel = spy
    try:
        builder(*args, **kwargs)
    finally:
        mx.fast.metal_kernel = real
    return captured["name"], captured["source"], captured["header"]


def rebuild(mx, name, source, header, replacements):
    for old, new in replacements:
        if old not in source:
            raise AssertionError(f"pattern missing for {name}: {old[:60]}")
    for old, new in replacements:
        source = source.replace(old, new)
    kwargs = dict(
        name=name,
        input_names=["row_ifft", "active_bf", "twiddle"],
        output_names=["sum_out", "sumsq_tile"],
        source=source,
        compile_options={"math_mode": "fast"},
    )
    if header is not None:
        kwargs["header"] = header
    return mx.fast.metal_kernel(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="probe")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    import mlx.core as mx

    from quantem.gpu.ssb.backends.mps import engine

    record: dict = {"schema": "quantem.ssb.mps.stageprobe.v1", "label": args.label}
    record["src_tree"] = str(SRC)
    opened, open_seconds = open_fixture(FIXTURE)
    backend, prepared, prepare_seconds = prepared_of(opened)
    record["open_seconds"] = round(open_seconds, 3)
    record["prepare_seconds"] = round(prepare_seconds, 3)
    record["bw_control_start_gbs"] = round(bandwidth_control(mx), 1)
    stored_bf = int(prepared.g_qk.shape[0])
    record["stored_bf"] = stored_bf
    record["g_qk_shape"] = list(prepared.g_qk.shape)
    record["g_qk_bytes"] = int(prepared.g_qk.nbytes)
    record["stored_g_bytes"] = int(np.prod(prepared.g_qk.shape) * 4)
    record["logical_bf"] = int(prepared.num_bf)

    c10 = np.asarray([PARAMS[0][0]], np.float32)
    c12 = np.asarray([PARAMS[0][1]], np.float32)
    phi = np.asarray([PARAMS[0][2]], np.float32)

    # --- exact pair objective (one full-BF candidate) -----------------------
    for label, n in (("pair_batch1", 1), ("pair_batch2", 2)):
        c10n = np.repeat(c10, n)
        c12n = np.repeat(c12, n)
        phin = np.repeat(phi, n)
        with SyncCounter() as counter:
            mx.reset_peak_memory()
            s = timed(
                lambda: engine._reconstruct_prepared_batch_exact_loss(
                    prepared, C10=c10n, C12=c12n, phi12=phin, chunk_bf=512
                )
            )
            peak = int(mx.get_peak_memory())
        s["label"] = label
        s["batch"] = n
        s["evals"] = counter.eval_calls // (WARMUP + REPEATS)
        s["peak_active_bytes"] = peak
        s["candidates"] = n
        record[label] = s

    # --- isolated fused 512 stages (one 512-wide logical boundary) ---------
    k_bf = engine._default_phase_col_k_bf((512, 512))
    record["phase_col_k_bf"] = int(k_bf)
    c10s = mx.array([PARAMS[0][0]], dtype=mx.float32)
    c12s = mx.array([PARAMS[0][1]], dtype=mx.float32)
    cos2s = mx.array([np.cos(2.0 * PARAMS[0][2])], dtype=mx.float32)
    sin2s = mx.array([np.sin(2.0 * PARAMS[0][2])], dtype=mx.float32)
    active = mx.abs(
        engine._pk_batch_from_prepared(
            prepared, start=0, stop=stored_bf, c10=c10s, c12=c12s,
            cos2phi12=cos2s, sin2phi12=sin2s,
        )[0]
    ) > 0.0
    active = active.astype(mx.uint8)
    mx.eval(active)
    record["active_bf"] = int(mx.sum(active.astype(mx.int32)).item())

    start, stop = 0, 512
    tiled_row = engine._row_ifft512_batch_from_dynamic_geometry(
        prepared, start=start, stop=stop, c10=c10s, c12=c12s,
        cos2phi12=cos2s, sin2phi12=sin2s, storage_bf=stop - start,
    )
    mx.eval(tiled_row)
    flat_row = engine._row_ifft512_from_dynamic_geometry(
        prepared, start=start, stop=stop, c10=c10s, c12=c12s,
        cos2phi12=cos2s, sin2phi12=sin2s,
    )
    mx.eval(flat_row)

    record["row_ifft_tiled"] = timed(
        lambda: mx.eval(
            engine._row_ifft512_batch_from_dynamic_geometry(
                prepared, start=start, stop=stop, c10=c10s, c12=c12s,
                cos2phi12=cos2s, sin2phi12=sin2s, storage_bf=stop - start,
            )
        )
    )
    record["row_ifft_flat"] = timed(
        lambda: mx.eval(
            engine._row_ifft512_from_dynamic_geometry(
                prepared, start=start, stop=stop, c10=c10s, c12=c12s,
                cos2phi12=cos2s, sin2phi12=sin2s,
            )
        )
    )
    cols_tiled = lambda: mx.eval(
        engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, tiled_row, k_bf=k_bf, active_bf=active[start:stop],
            tiled_input=True,
        )
    )
    cols_flat = lambda: mx.eval(
        engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, flat_row[None, ...], k_bf=k_bf, active_bf=active[start:stop],
            tiled_input=False,
        )
    )
    record["phase_cols_tiled"] = timed(cols_tiled)
    record["phase_cols_flat"] = timed(cols_flat)
    record["phase_cols_pack_tiled"] = timed(
        lambda: mx.eval(
            engine._phase_cols512_pack_loss_batch_from_row_ifft(
                mx, tiled_row, bf_ranges=((0, 512),), active_bf=active[start:stop],
            )
        )
    )

    # --- kernel-source ablations (timing only, mathematically incomplete) --
    if not args.skip_ablations:
        engine._phase_cols512_radix8_batch_kernel.cache_clear()
        name, source, header = capture(
            mx,
            engine._phase_cols512_radix8_batch_kernel,
            1,
            512,
            int(k_bf),
            True,
            512,
            0,
        )
        record["phase_cols_kernel_name"] = name
        atan2_off = [(f"metal::atan2(r{i}.y,r{i}.x)", f"(r{i}.x + r{i}.y)")
                     for i in range(8)]
        barrier_off = [("threadgroup_barrier(mem_flags::mem_threadgroup);", "")]
        variants = {
            "phase_cols_no_atan2": atan2_off,
            "phase_cols_no_barrier": barrier_off,
            "phase_cols_no_atan2_no_barrier": atan2_off + barrier_off,
        }
        for label, replacement in variants.items():
            kernel = rebuild(mx, f"probe_{label}", source, header, replacement)
            def run(kernel=kernel):
                out, _tile = kernel(
                    inputs=[tiled_row, active[start:stop], engine._twiddle_512(mx)],
                    template=[],
                    grid=(64, 512, 1),
                    threadgroup=(64, 8, 1),
                    output_shapes=[(1, 1, 512, 512), (1, 1, 512, 64)],
                    output_dtypes=[mx.float32, mx.float32],
                )
                mx.eval(out)
            record[label] = timed(run)

        engine._phase_cols512_radix8_batch_kernel.cache_clear()
        engine._row_ifft512_dynamic_kernel.cache_clear()
        rname, rsource, _rheader = capture(
            mx,
            engine._row_ifft512_dynamic_kernel,
            1,
            stop - start,
            int(prepared.g_qk.shape[-1]),
            True,
            stop - start,
        )
        record["row_ifft_kernel_name"] = rname
        row_variants = {
            "row_ifft_no_sqrt": [
                ("metal::sqrt(r2)", "r2"),
                ("metal::sqrt(denom_num2)", "denom_num2"),
            ],
            "row_ifft_no_sincos": [
                ("metal::fast::sincos(chi_m, cos_chi_m)", "cos_chi_m = chi_m"),
                ("metal::fast::sincos(chi_p, cos_chi_p)", "cos_chi_p = chi_p"),
            ],
            "row_ifft_no_geometry": [
                ("metal::sqrt(r2)", "r2"),
                ("metal::sqrt(denom_num2)", "denom_num2"),
                ("metal::fast::sincos(chi_m, cos_chi_m)", "cos_chi_m = chi_m"),
                ("metal::fast::sincos(chi_p, cos_chi_p)", "cos_chi_p = chi_p"),
                ("1.0f / r2", "r2"),
                ("1.0f / r", "r"),
            ],
        }
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
        pk = engine._pk_batch_from_prepared(
            prepared, start=start, stop=stop, c10=c10s, c12=c12s,
            cos2phi12=cos2s, sin2phi12=sin2s,
        )
        for label, replacement in row_variants.items():
            for old, new in replacement:
                if old not in rsource:
                    raise AssertionError(f"pattern missing: {old}")
            variant_source = rsource
            for old, new in replacement:
                variant_source = variant_source.replace(old, new)
            kernel = mx.fast.metal_kernel(
                name=f"probe_{label}",
                input_names=[
                    "g", "q_row", "q_col", "kx", "ky", "pk", "c10", "c12",
                    "cos2phi12", "sin2phi12", "scalars", "twiddle",
                ],
                output_names=["row_ifft"],
                source=variant_source,
                compile_options={"math_mode": "fast"},
            )
            inputs = [
                prepared.g_qk[start:stop], prepared.q_row, prepared.q_col,
                prepared.kx[start:stop], prepared.ky[start:stop], pk, c10s,
                c12s, cos2s, sin2s, scalars, engine._twiddle_512(mx),
            ]
            def run(kernel=kernel, inputs=inputs):
                out = kernel(
                    inputs=inputs,
                    template=[],
                    grid=(64, 512, stop - start),
                    threadgroup=(64, 4, 1),
                    output_shapes=[(1, stop - start, 512, 512)],
                    output_dtypes=[mx.complex64],
                )[0]
                mx.eval(out)
            record[label] = timed(run)
        engine._row_ifft512_dynamic_kernel.cache_clear()
        engine._phase_cols512_radix8_batch_kernel.cache_clear()

    # --- simd register-transpose column stage (bit-exactness gate) --------
    real_use_simd = engine._use_simd_radix8_col_stage_512
    try:
        engine._use_simd_radix8_col_stage_512 = lambda: True
        engine._phase_cols512_radix8_batch_kernel.cache_clear()
        simd_sum, simd_sumsq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, tiled_row, k_bf=k_bf, active_bf=active[start:stop], tiled_input=True
        )
        base_sum, base_sumsq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, tiled_row, k_bf=k_bf, active_bf=active[start:stop], tiled_input=True
        )
        a, b = np.asarray(simd_sum), np.asarray(base_sum)
        q, r = np.asarray(simd_sumsq), np.asarray(base_sumsq)
        record["simd_cols_bit_exact"] = bool(
            np.array_equal(a, b) and np.array_equal(q, r)
        )
        record["simd_cols_max_abs_diff"] = float(
            max(np.abs(a - b).max(), np.abs(q - r).max())
        )
        record["phase_cols_tiled_simd"] = timed(cols_tiled)
        engine._use_simd_radix8_col_stage_512 = real_use_simd
        engine._phase_cols512_radix8_batch_kernel.cache_clear()
    finally:
        engine._use_simd_radix8_col_stage_512 = real_use_simd

    # --- scalar 512 phase/loss path with raw bit hashes -------------------
    scalar = {"hashes": [], "timings": {}}
    for index, (params_c10, params_c12, params_phi) in enumerate(PARAMS):
        _obj, loss, phase = engine._reconstruct_prepared(
            prepared,
            C10=params_c10,
            C12=params_c12,
            phi12=params_phi,
            chunk_bf=512,
            compute_loss=True,
            compute_object=False,
            return_phase=True,
        )
        scalar["hashes"].append(
            {
                "index": index,
                "loss_repr": repr(loss),
                "loss_hex": float(loss).hex(),
                "phase_sha256": sha256_array(phase),
            }
        )
    with SyncCounter() as counter:
        mx.reset_peak_memory()
        scalar["timings"]["chunk512"] = timed(
            lambda: engine._reconstruct_prepared(
                prepared, C10=PARAMS[0][0], C12=PARAMS[0][1], phi12=PARAMS[0][2],
                chunk_bf=512, compute_loss=True, compute_object=False,
                return_phase=False,
            )
        )
        scalar["peak_active_bytes"] = int(mx.get_peak_memory())
    scalar["timings"]["chunk512"]["evals"] = counter.eval_calls // (WARMUP + REPEATS)
    # Diagnostic only: other chunk sizes change the float32 reduction grouping.
    for chunk in (256, 1024):
        scalar["timings"][f"chunk{chunk}"] = timed(
            lambda chunk=chunk: engine._reconstruct_prepared(
                prepared, C10=PARAMS[0][0], C12=PARAMS[0][1], phi12=PARAMS[0][2],
                chunk_bf=chunk, compute_loss=True, compute_object=False,
                return_phase=False,
            ),
            repeats=3,
        )
    record["scalar_loss"] = scalar

    # --- object redraw -----------------------------------------------------
    with SyncCounter() as counter:
        mx.reset_peak_memory()
        record["redraw"] = timed(
            lambda: engine._object_fourier_sum_dynamic(
                prepared, C10=PARAMS[0][0], C12=PARAMS[0][1], phi12=PARAMS[0][2],
                chunk_bf=engine._default_object_redraw_chunk_bf(),
            )
        )
        record["redraw"]["peak_active_bytes"] = int(mx.get_peak_memory())
    record["redraw"]["evals"] = counter.eval_calls // (WARMUP + REPEATS)

    record["bw_control_end_gbs"] = round(bandwidth_control(mx), 1)
    line = json.dumps(record, sort_keys=True, default=str)
    print(line, flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
