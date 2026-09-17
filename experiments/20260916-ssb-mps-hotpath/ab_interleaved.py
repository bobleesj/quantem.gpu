"""Interleaved (paired) A/B of the candidate MLX/MPS changes.

Session-to-session drift on this shared machine is ~3%, which hides every
change below that. Each variant here is therefore measured by alternating
A/B/A/B inside one process so the paired difference cancels the drift.
"""
from __future__ import annotations

import argparse
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
]


def paired(times_a, times_b):
    a = np.asarray(times_a, dtype=np.float64)
    b = np.asarray(times_b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    delta = a - b
    return {
        "a_p50_ms": round(float(np.median(a)) * 1e3, 3),
        "b_p50_ms": round(float(np.median(b)) * 1e3, 3),
        "a_min_ms": round(float(a.min()) * 1e3, 3),
        "b_min_ms": round(float(b.min()) * 1e3, 3),
        "ratio_b_over_a": round(float(np.median(b) / np.median(a)), 4),
        "paired_delta_p50_ms": round(float(np.median(delta)) * 1e3, 3),
        "paired_delta_mean_ms": round(float(delta.mean()) * 1e3, 3),
        "a_seconds": [round(float(v), 6) for v in a],
        "b_seconds": [round(float(v), 6) for v in b],
        "n": int(n),
    }


def alternate(fn_a, fn_b, reps):
    times_a, times_b = [], []
    fn_a()
    fn_b()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn_a()
        times_a.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fn_b()
        times_b.append(time.perf_counter() - t0)
    return times_a, times_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="interleaved")
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import mlx.core as mx

    from quantem.gpu.ssb.backends.mps import engine

    record = {"schema": "quantem.ssb.mps.interleaved.v1", "label": args.label,
              "src_tree": str(SRC), "reps": args.reps}
    opened, _ = open_fixture(FIXTURE)
    backend, prepared, prepare_seconds = prepared_of(opened)
    record["prepare_seconds"] = round(prepare_seconds, 3)
    record["bw_control_start_gbs"] = round(bandwidth_control(mx), 1)

    c10v, c12v, phiv = PARAMS[0]
    c10s = mx.array([c10v], dtype=mx.float32)
    c12s = mx.array([c12v], dtype=mx.float32)
    cos2s = mx.array([np.cos(2.0 * phiv)], dtype=mx.float32)
    sin2s = mx.array([np.sin(2.0 * phiv)], dtype=mx.float32)
    k_bf = engine._default_phase_col_k_bf((512, 512))
    stored_bf = int(prepared.g_qk.shape[0])
    active = (mx.abs(engine._pk_batch_from_prepared(
        prepared, start=0, stop=stored_bf, c10=c10s, c12=c12s,
        cos2phi12=cos2s, sin2phi12=sin2s)[0]) > 0.0).astype(mx.uint8)
    mx.eval(active)
    start, stop = 0, 512

    # --- A/B 1: the fused row+column stage pair, flat vs tiled storage -----
    def produce_flat():
        return engine._row_ifft512_from_dynamic_geometry(
            prepared, start=start, stop=stop, c10=c10s, c12=c12s,
            cos2phi12=cos2s, sin2phi12=sin2s, tiled_output=False)

    def produce_tiled():
        return engine._row_ifft512_batch_from_dynamic_geometry(
            prepared, start=start, stop=stop, c10=c10s, c12=c12s,
            cos2phi12=cos2s, sin2phi12=sin2s, storage_bf=stop - start)

    flat_row = produce_flat()
    tiled_row = produce_tiled()
    mx.eval(flat_row, tiled_row)

    def stage_flat():
        row = produce_flat()
        s, q = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, row[None, ...], k_bf=k_bf, active_bf=active[start:stop],
            tiled_input=False)
        mx.eval(s, q)
        return s, q

    def stage_tiled():
        row = produce_tiled()
        s, q = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, row, k_bf=k_bf, active_bf=active[start:stop], tiled_input=True)
        mx.eval(s, q)
        return s, q

    stage_flat()
    tiled_out = stage_tiled()
    flat_out = stage_flat()
    record["stage_pair_bit_exact"] = bool(
        np.array_equal(np.asarray(flat_out[0]), np.asarray(tiled_out[0]))
        and np.array_equal(np.asarray(flat_out[1]), np.asarray(tiled_out[1])))
    a, b = alternate(stage_flat, stage_tiled, args.reps)
    record["stage_pair_flat_vs_tiled"] = paired(a, b)

    # --- A/B 2: whole scalar phase/loss path, tiled vs forced flat --------
    real_geom = engine._row_ifft512_from_dynamic_geometry
    real_cols = engine._phase_cols512_scalar_loss_batch_from_row_ifft

    def geom_flat(*a_, **kw):
        kw["tiled_output"] = False
        return real_geom(*a_, **kw)

    def cols_flat(mx_, row_ifft, **kw):
        kw["tiled_input"] = False
        return real_cols(mx_, row_ifft, **kw)

    def scalar_native():
        return engine._reconstruct_prepared(
            prepared, C10=c10v, C12=c12v, phi12=phiv, chunk_bf=512,
            compute_loss=True, compute_object=False, return_phase=False)

    def scalar_flat():
        engine._row_ifft512_from_dynamic_geometry = geom_flat
        engine._phase_cols512_scalar_loss_batch_from_row_ifft = cols_flat
        try:
            return engine._reconstruct_prepared(
                prepared, C10=c10v, C12=c12v, phi12=phiv, chunk_bf=512,
                compute_loss=True, compute_object=False, return_phase=False)
        finally:
            engine._row_ifft512_from_dynamic_geometry = real_geom
            engine._phase_cols512_scalar_loss_batch_from_row_ifft = real_cols

    with SyncCounter() as counter:
        a, b = alternate(scalar_native, scalar_flat, 7)
    record["scalar_loss_tiled_vs_flat"] = paired(a, b)
    record["scalar_loss_evals_per_call"] = counter.eval_calls // 14
    for index, (pc10, pc12, pphi) in enumerate(PARAMS):
        _o, loss_n, phase_n = engine._reconstruct_prepared(
            prepared, C10=pc10, C12=pc12, phi12=pphi, chunk_bf=512,
            compute_loss=True, compute_object=False, return_phase=True)
        engine._row_ifft512_from_dynamic_geometry = geom_flat
        engine._phase_cols512_scalar_loss_batch_from_row_ifft = cols_flat
        try:
            _o, loss_f, phase_f = engine._reconstruct_prepared(
                prepared, C10=pc10, C12=pc12, phi12=pphi, chunk_bf=512,
                compute_loss=True, compute_object=False, return_phase=True)
        finally:
            engine._row_ifft512_from_dynamic_geometry = real_geom
            engine._phase_cols512_scalar_loss_batch_from_row_ifft = real_cols
        record.setdefault("scalar_parity", []).append({
            "index": index,
            "loss_tiled_hex": float(loss_n).hex(),
            "loss_flat_hex": float(loss_f).hex(),
            "loss_equal": float(loss_n) == float(loss_f),
            "phase_equal": bool(np.array_equal(phase_n, phase_f)),
            "phase_sha256_tiled": sha256_array(phase_n),
            "phase_sha256_flat": sha256_array(phase_f),
        })

    # --- A/B 3: column stage with the simd register transpose -------------
    real_simd = engine._use_simd_radix8_col_stage_512

    def cols_simd():
        engine._use_simd_radix8_col_stage_512 = lambda: True
        engine._phase_cols512_radix8_batch_kernel.cache_clear()
        try:
            s, q = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
                mx, tiled_row, k_bf=k_bf, active_bf=active[start:stop],
                tiled_input=True)
            mx.eval(s, q)
            return s, q
        finally:
            engine._use_simd_radix8_col_stage_512 = real_simd
            engine._phase_cols512_radix8_batch_kernel.cache_clear()

    def cols_base():
        s, q = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, tiled_row, k_bf=k_bf, active_bf=active[start:stop],
            tiled_input=True)
        mx.eval(s, q)
        return s, q

    cols_base()
    cols_simd()
    base_sum, base_sq = cols_base()
    simd_sum, simd_sq = cols_simd()
    record["simd_cols_bit_exact"] = bool(
        np.array_equal(np.asarray(base_sum), np.asarray(simd_sum))
        and np.array_equal(np.asarray(base_sq), np.asarray(simd_sq)))
    a, b = alternate(cols_base, cols_simd, args.reps)
    record["cols_tiled_base_vs_simd"] = paired(a, b)

    # --- A/B 4: whole exact pair objective with the simd column stage -----
    def pair_with_simd(simd):
        engine._use_simd_radix8_col_stage_512 = (lambda: True) if simd else real_simd
        engine._phase_cols512_radix8_batch_kernel.cache_clear()
        try:
            return engine._reconstruct_prepared_batch_exact_loss(
                prepared, C10=np.asarray([c10v], np.float32),
                C12=np.asarray([c12v], np.float32),
                phi12=np.asarray([phiv], np.float32), chunk_bf=512)
        finally:
            engine._use_simd_radix8_col_stage_512 = real_simd
            engine._phase_cols512_radix8_batch_kernel.cache_clear()

    loss_base = pair_with_simd(False)
    loss_simd = pair_with_simd(True)
    record["pair_simd_loss_equal"] = bool(np.array_equal(loss_base, loss_simd))
    record["pair_simd_loss_hex"] = [float(loss_base[0]).hex(), float(loss_simd[0]).hex()]
    a, b = alternate(lambda: pair_with_simd(False), lambda: pair_with_simd(True), 5)
    record["pair_base_vs_simd"] = paired(a, b)

    record["bw_control_end_gbs"] = round(bandwidth_control(mx), 1)
    line = json.dumps(record, sort_keys=True, default=str)
    print(line, flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
