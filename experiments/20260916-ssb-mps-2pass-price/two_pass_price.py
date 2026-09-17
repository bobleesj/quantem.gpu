"""Price the 4-passes-to-2 (fused row+column) attempt on the 8937-BF fit path.

Nothing here is landed and nothing here is bit-exact by design.  The probe
measures, in one locked GPU session on the calibrated 512 fixture:

  1. the same-session streaming ceiling (copy control) and host load,
  2. the frozen exact-pair objective (batch 2, chunk 512) with peak memory,
  3. the row stage and the column stage in isolation, so the intermediate
     round trip is measured rather than modelled,
  4. timing-only ablations that fold the row stage's plane store into four
     planes and pin the column stage's read to one plane (the ALU floor a
     fused kernel could not go below; both produce wrong values by design),
  5. a differently-decomposed second stage: MLX's own ``mx.fft.ifft`` on the
     identical row-IFFT intermediate, against the engine's radix-8 column
     stage, with MLX's 2-D ``ifft2`` timed as a whole-plane reference point.

The deviation work is written to a compact NPZ so the float64 reference and
the deviation table can be computed off the GPU lock.
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

from profile_mps import (  # noqa: E402
    FIXTURE,
    bandwidth_control,
    open_fixture,
    prepared_of,
    sha256_array,
)

PINNED = (7.017120839737006, 0.0, -0.15393969519675116)
SECOND = (-50.0, 35.0, -0.2)
DEV_PACK_BF = 32
REPEATS = 5
WARMUP = 2
_MX = None


def _mx():
    global _MX
    if _MX is None:
        import mlx.core as mx  # noqa: PLC0415

        _MX = mx
    return _MX


def stat(name: str, times: list[float], bytes_moved: int | None = None) -> dict:
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
    return out


def _sync(out) -> None:
    """Synchronize whatever an engine stage returns (numpy or MLX arrays)."""
    mx = _mx()
    if isinstance(out, (list, tuple)):
        mx.eval(*out)
    elif isinstance(out, mx.array):
        mx.eval(out)


def bench(fn, repeats: int = REPEATS, warmup: int = WARMUP):
    for _ in range(warmup):
        _sync(fn())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _sync(fn())
        times.append(time.perf_counter() - t0)
    return times


def candidate_arrays(params):
    mx = _mx()
    c10 = mx.array(np.asarray([p[0] for p in params], dtype=np.float32))
    c12 = mx.array(np.asarray([p[1] for p in params], dtype=np.float32))
    phi = np.asarray([p[2] for p in params], dtype=np.float64)
    cos2 = mx.array(np.cos(2.0 * phi).astype(np.float32))
    sin2 = mx.array(np.sin(2.0 * phi).astype(np.float32))
    return c10, c12, cos2, sin2


class _Capture:
    """Capture the exact MSL source an engine kernel builder compiles."""

    def __init__(self, engine):
        self.engine = engine
        self.sources: dict[str, str] = {}

    def __enter__(self):
        mx = _mx()
        real = mx.fast.metal_kernel
        sources = self.sources

        def capturing(**kwargs):
            sources[kwargs.get("name", "")] = kwargs.get("source", "")
            return real(**kwargs)

        class _Fast:
            metal_kernel = staticmethod(capturing)

        class _Proxy:
            fast = _Fast()

        self._original = self.engine._require_mlx
        self.engine._require_mlx = lambda: _Proxy()
        return self

    def __exit__(self, *exc):
        self.engine._require_mlx = self._original
        return False


def build_row_variant(engine, prepared, batch, chunk, storage_bf):
    """Rebuild the exact row kernel, folding the plane store into four planes.

    The store instructions still execute and still consume L2 write bandwidth;
    only the device_memory write traffic of the 2,147 MB intermediate disappears, so the
    result is a conservative upper bound on a no-intermediate kernel's time.
    """
    mx = _mx()
    engine._row_ifft512_dynamic_kernel.cache_clear()
    with _Capture(engine) as cap:
        engine._row_ifft512_dynamic_kernel(
            batch, chunk, int(prepared.g_qk.shape[-1]), batch <= 2, storage_bf
        )
    source = next(iter(cap.sources.values()))
    needle = "(size_t)output_batch * (size_t)CHUNK + (size_t)bf)"
    if needle not in source:
        raise RuntimeError("row store pattern changed; variant not built")
    source = source.replace(needle, "(size_t)output_batch * 4u + (size_t)(bf & 3u))")
    return mx.fast.metal_kernel(
        name=f"ssb_row_ifft512_probe_n{batch}_b{chunk}_foldstore",
        input_names=[
            "g",
            "q_row",
            "q_col",
            "kx",
            "ky",
            "pk",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
            "twiddle",
        ],
        output_names=["row_ifft"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def build_col_variant(engine, batch, num_bf, k_bf, storage_num_bf):
    """Rebuild the exact column kernel with its plane read pinned to BF 0."""
    mx = _mx()
    engine._phase_cols512_radix8_batch_kernel.cache_clear()
    with _Capture(engine) as cap:
        engine._phase_cols512_radix8_batch_kernel(
            batch, num_bf, k_bf, True, storage_num_bf, 0
        )
    source = next(iter(cap.sources.values()))
    needle = "+ (size_t)BF_OFFSET + (size_t)bf)"
    if needle not in source:
        raise RuntimeError("column load pattern changed; variant not built")
    source = source.replace(needle, "+ 0u)")
    return mx.fast.metal_kernel(
        name=f"ssb_phase_cols512_probe_n{batch}_bf{num_bf}_k{k_bf}_pin",
        input_names=["row_ifft", "active_bf", "twiddle"],
        output_names=["sum_out", "sumsq_tile"],
        source=source,
        header=engine._twiddle_512_metal_header(),
        compile_options={"math_mode": "fast"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=FIXTURE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="2pass-price")
    args = parser.parse_args()

    mx = _mx()
    # Canonical implementation: the compute module only re-exports it, and
    # it does not re-export the internal pack-policy helpers used below.
    from quantem.gpu.ssb.backends.mps import engine  # noqa: PLC0415

    record: dict[str, object] = {
        "probe": "two-pass-price",
        "label": args.label,
        "host": platform.node(),
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "load_average": list(os.getloadavg()),
        "pinned_loss": 0.13769753277301788,
        "pinned_optimum": {"C10": PINNED[0], "C12": PINNED[1], "phi12": PINNED[2]},
    }

    opened, open_seconds = open_fixture(args.fixture)
    backend, prepared, prepare_seconds = prepared_of(opened)
    prepared._probe_engine = engine  # type: ignore[attr-defined]
    plane_bytes = 512 * 512 * 8
    half_plane_bytes = 512 * 257 * 8
    record |= {
        "open_seconds": round(open_seconds, 3),
        "prepare_seconds": round(prepare_seconds, 3),
        "scan_shape": list(prepared.scan_shape),
        "stored_bf": int(prepared.g_qk.shape[0]),
        "g_qk_bytes": int(prepared.g_qk.nbytes),
        "g_qk_shape": list(prepared.g_qk.shape),
        "bandwidth_control_gbs": round(bandwidth_control(mx), 1),
    }

    c10, c12, cos2, sin2 = candidate_arrays([PINNED, SECOND])

    # ---- baseline: the frozen exact-pair objective -------------------------
    mx.reset_peak_memory()

    def pair_call():
        return engine._reconstruct_prepared_batch_exact_loss(
            prepared,
            C10=np.asarray([PINNED[0], SECOND[0]], dtype=np.float32),
            C12=np.asarray([PINNED[1], SECOND[1]], dtype=np.float32),
            phi12=np.asarray([PINNED[2], SECOND[2]], dtype=np.float32),
            chunk_bf=512,
        )

    times = bench(pair_call)
    losses = np.asarray(pair_call()).copy()
    modelled = 18 * (2 * 512 * plane_bytes * 2 + 512 * half_plane_bytes)
    record["pair_objective"] = stat("pair_objective", times, modelled)
    record["pair_objective"]["loss"] = [float(v) for v in losses]
    record["pair_objective"]["peak_active_bytes"] = int(mx.get_peak_memory())
    record["pair_objective"]["modelled_bytes_note"] = (
        "18 packs x (2 candidates x 512 planes x write+read) + 18 shared "
        "half-plane G reads of 512 planes, matching the frozen pack structure."
    )
    record["pair_objective"]["per_candidate_bytes_per_pack"] = int(
        (2 * 512 * plane_bytes * 2 + 512 * half_plane_bytes) / 2
    )

    # ---- stages in isolation ----------------------------------------------
    row_bytes = 512 * half_plane_bytes + 2 * 512 * plane_bytes
    row_times = bench(lambda: engine._row_ifft512_batch_from_dynamic_geometry(
        prepared, start=0, stop=512, c10=c10, c12=c12, cos2phi12=cos2, sin2phi12=sin2
    ))
    row_ifft = engine._row_ifft512_batch_from_dynamic_geometry(
        prepared, start=0, stop=512, c10=c10, c12=c12, cos2phi12=cos2, sin2phi12=sin2
    )
    mx.eval(row_ifft)
    record["row_stage_isolated"] = stat("row_stage_512", row_times, row_bytes)

    active_all = mx.ones((512,), dtype=mx.uint8)

    def col_stage():
        return engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, row_ifft, k_bf=32, active_bf=active_all, tiled_input=True
        )

    record["column_stage_isolated"] = stat(
        "column_stage_512", bench(col_stage), 2 * 512 * plane_bytes
    )

    # ---- timing-only ablations: the ALU floor of a fused kernel ------------
    ablations: dict[str, object] = {}
    try:
        storage_bf = engine._exact_pair_row_allocation_bf_512(512)
        pk_full = engine._pk_batch_from_prepared(
            prepared, start=0, stop=512, c10=c10, c12=c12, cos2phi12=cos2, sin2phi12=sin2
        )
        mx.eval(pk_full)
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
        twiddle = engine._twiddle_512(mx)
        folded = build_row_variant(engine, prepared, 2, 512, storage_bf)
        del row_ifft
        mx.clear_cache()

        def row_folded():
            return folded(
                inputs=[
                    prepared.g_qk[0:512],
                    prepared.q_row,
                    prepared.q_col,
                    prepared.kx[0:512],
                    prepared.ky[0:512],
                    pk_full,
                    c10,
                    c12,
                    cos2,
                    sin2,
                    scalars,
                    twiddle,
                ],
                template=[],
                grid=(64, 512, 512),
                threadgroup=(64, 4, 1),
                output_shapes=[(2, storage_bf, 512, 512)],
                output_dtypes=[mx.complex64],
            )

        record["row_stage_folded_store"] = stat(
            "row_stage_folded_store", bench(row_folded), 512 * half_plane_bytes
        )
        ablations["row_folded_store"] = True
    except Exception as exc:  # noqa: BLE001 - probe, report the reason
        ablations["row_folded_store"] = f"{type(exc).__name__}: {exc}"[:300]

    try:
        pin_row = engine._row_ifft512_batch_from_dynamic_geometry(
            prepared, start=0, stop=512, c10=c10, c12=c12, cos2phi12=cos2, sin2phi12=sin2
        )
        mx.eval(pin_row)
        pinned = build_col_variant(engine, 2, 512, 32, 512)

        def col_pinned():
            return pinned(
                inputs=[pin_row, active_all, twiddle],
                template=[],
                grid=(64, 512, 2 * 16),
                threadgroup=(64, 8, 1),
                output_shapes=[(2, 16, 512, 512), (2, 16, 512, 64)],
                output_dtypes=[mx.float32, mx.float32],
            )

        record["column_stage_pinned_plane"] = stat(
            "column_stage_pinned", bench(col_pinned)
        )
        ablations["col_pinned_plane"] = True
        del pin_row
        mx.clear_cache()
    except Exception as exc:  # noqa: BLE001 - probe, report the reason
        ablations["col_pinned_plane"] = f"{type(exc).__name__}: {exc}"[:300]

    record["ablations"] = ablations

    # ---- a differently decomposed second stage ----------------------------
    dev = engine._row_ifft512_batch_from_dynamic_geometry(
        prepared,
        start=0,
        stop=DEV_PACK_BF,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    )
    mx.eval(dev)
    active_dev = mx.ones((DEV_PACK_BF,), dtype=mx.uint8)

    eng_times = bench(
        lambda: engine._phase_cols512_scalar_loss_batch_from_row_ifft(
            mx, dev, k_bf=DEV_PACK_BF, active_bf=active_dev, tiled_input=True
        )
    )
    eng_sum, eng_sumsq = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, dev, k_bf=DEV_PACK_BF, active_bf=active_dev, tiled_input=True
    )
    mx.eval(eng_sum, eng_sumsq)
    eng_sum32, eng_sumsq32 = engine._phase_cols512_scalar_loss_batch_from_row_ifft(
        mx, dev, k_bf=32, active_bf=active_dev, tiled_input=True
    )
    mx.eval(eng_sum32, eng_sumsq32)

    alt_times = bench(lambda: mx.fft.ifft(dev, axis=2))
    alt_plane = mx.fft.ifft(dev, axis=2)
    alt_phase = mx.arctan2(alt_plane.imag, alt_plane.real)
    alt_sum = mx.sum(alt_phase, axis=1)
    alt_sumsq = mx.sum(alt_phase * alt_phase, axis=(1, 2, 3))
    mx.eval(alt_sum, alt_sumsq)
    conj_sum = mx.sum(mx.arctan2(-alt_plane.imag, alt_plane.real), axis=1)
    mx.eval(conj_sum)

    try:
        ifft2_times = bench(lambda: mx.fft.ifft2(dev))
        record["mlx_ifft2_full_plane"] = stat("mlx_ifft2_full", ifft2_times)
    except Exception as exc:  # noqa: BLE001 - reference point only
        record["mlx_ifft2_full_plane"] = f"{type(exc).__name__}: {exc}"[:200]

    record["deviation_pack"] = {
        "bf_terms": DEV_PACK_BF,
        "note": (
            "engine column stage with one BF group (k_bf=bf_terms) versus "
            "k_bf=32 (pure float32 regrouping control) versus MLX's own "
            "mx.fft.ifft along the same axis on the identical intermediate."
        ),
        "engine_column_stage": stat("engine_col_k64", eng_times),
        "mlx_ifft_axis": stat("mlx_ifft_axis", alt_times),
        "intermediate_bytes": int(2 * DEV_PACK_BF * plane_bytes),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "alt_decomp_pack.npz",
        row_ifft=np.asarray(dev),
        eng_sum=np.asarray(eng_sum),
        eng_sumsq=np.asarray(eng_sumsq),
        eng_sum32=np.asarray(eng_sum32),
        eng_sumsq32=np.asarray(eng_sumsq32),
        alt_sum=np.asarray(alt_sum),
        alt_sumsq=np.asarray(alt_sumsq),
        conj_sum=np.asarray(conj_sum),
    )
    record["deviation_pack"]["row_ifft_sha256"] = sha256_array(np.asarray(dev))
    record["deviation_pack"]["npz"] = str(args.out_dir / "alt_decomp_pack.npz")

    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
