"""Off-GPU deviation table for the two-pass price pack.

Reads the NPZ written by ``two_pass_price.py`` and reports, on identical
inputs, how far a differently decomposed second stage moves the per-pixel
phase sum, the phase sum of squares and the pack-level loss:

  engine k_bf=bf_terms   the frozen radix-8 column stage, one BF group
  engine k_bf=32         the same arithmetic with a different float32
                         reduction grouping: the within-path floor
  mlx ifft               MLX's own FFT on the identical intermediate
  mlx conj(ifft)         sign-convention control for the same
  float64                numpy float64 reference of the same intermediate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pack_loss(sum_field, total_sumsq, n_bf: int) -> float:
    """The engine's loss shape: sumsq/norm - mean((sum/n_bf)^2)."""
    norm = float(n_bf) * float(sum_field.shape[-1] * sum_field.shape[-2])
    mean_phase = sum_field / float(n_bf)
    return float(total_sumsq) / norm - float(np.mean(mean_phase * mean_phase))


def compare(name: str, got_sum, got_sumsq, ref_sum, ref_sumsq, n_bf: int) -> dict:
    diff = np.asarray(got_sum, dtype=np.float64) - np.asarray(ref_sum, dtype=np.float64)
    ref = np.asarray(ref_sum, dtype=np.float64)
    got_loss = [pack_loss(np.asarray(got_sum)[i], np.asarray(got_sumsq)[i], n_bf)
                for i in range(np.asarray(got_sum).shape[0])]
    ref_loss = [pack_loss(np.asarray(ref_sum)[i], np.asarray(ref_sumsq)[i], n_bf)
                for i in range(np.asarray(ref_sum).shape[0])]
    loss_abs = [abs(a - b) for a, b in zip(got_loss, ref_loss)]
    loss_rel = [abs(a - b) / abs(b) if b else float("inf") for a, b in zip(got_loss, ref_loss)]
    return {
        "variant": name,
        "sum_field_max_abs": float(np.max(np.abs(diff))),
        "sum_field_rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-30)),
        "pack_loss": [round(v, 12) for v in got_loss],
        "pack_loss_max_abs_vs_engine": float(np.max(loss_abs)),
        "pack_loss_max_rel_vs_engine": float(np.max(loss_rel)),
        "sumsq_max_rel": float(
            np.max(np.abs(np.asarray(got_sumsq, dtype=np.float64) - np.asarray(ref_sumsq, dtype=np.float64))
                   / np.maximum(np.abs(np.asarray(ref_sumsq, dtype=np.float64)), 1e-30))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--label", default="2pass-price")
    parser.add_argument("--gate-rtol", type=float, default=1e-5)
    args = parser.parse_args()

    data = np.load(args.npz)
    row_ifft = data["row_ifft"]
    n_bf = int(row_ifft.shape[1])

    ref_plane = np.fft.ifft(row_ifft.astype(np.complex128), axis=2)
    ref_phase = np.arctan2(ref_plane.imag, ref_plane.real)
    ref_sum = ref_phase.sum(axis=1)
    ref_sumsq = np.asarray([float(np.sum(p * p)) for p in ref_phase])

    engine = compare(
        "engine_k_bf_full", data["eng_sum"], data["eng_sumsq"], ref_sum, ref_sumsq, n_bf
    )
    regroup = compare(
        "engine_k_bf_32", data["eng_sum32"], data["eng_sumsq32"], ref_sum, ref_sumsq, n_bf
    )
    alt = compare("mlx_ifft", data["alt_sum"], data["alt_sumsq"], ref_sum, ref_sumsq, n_bf)
    # mlx_ifft vs engine_k_bf_full: the actual re-decomposition price
    redecomp = compare(
        "mlx_ifft_vs_engine", data["alt_sum"], data["alt_sumsq"],
        data["eng_sum"], data["eng_sumsq"], n_bf,
    )
    regroup_vs_engine = compare(
        "engine_k32_vs_engine", data["eng_sum32"], data["eng_sumsq32"],
        data["eng_sum"], data["eng_sumsq"], n_bf,
    )
    conj = compare("mlx_conj_ifft_sum_only", data["conj_sum"], data["alt_sumsq"],
                   data["eng_sum"], data["eng_sumsq"], n_bf)

    record = {
        "label": args.label,
        "bf_terms": n_bf,
        "npz": str(args.npz),
        "gate_rtol": args.gate_rtol,
        "pinned_loss": 0.13769753277301788,
        "variants_vs_float64_reference": [engine, regroup, alt],
        "redecomposition_vs_engine": {
            "mlx_ifft": redecomp,
            "engine_regroup_k32": regroup_vs_engine,
            "mlx_conj_ifft": conj,
        },
        "float32_floor": {
            "engine_vs_float64_pack_loss_rel": engine["pack_loss_max_rel_vs_engine"],
            "engine_regroup_pack_loss_rel": regroup_vs_engine["pack_loss_max_rel_vs_engine"],
            "note": (
                "the engine's own path already moves by this much when only the "
                "float32 reduction grouping changes; compare the re-decomposition "
                "figure against it and against the repository gate rtol."
            ),
        },
        "exceeds_gate": bool(
            redecomp["pack_loss_max_rel_vs_engine"] > args.gate_rtol
        ),
    }
    line = json.dumps(record, sort_keys=True)
    print(json.dumps(json.loads(line), indent=2, sort_keys=True), flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
