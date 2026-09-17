"""Final deviation table for the two-pass price, on de-tiled logical planes.

Supersedes the earlier ``analyze_two_pass.py`` pass, which compared the
*tiled* store layout against a plain FFT axis and therefore measured a
permutation, not a decomposition.  ``alt_decomp_sums.npz`` was produced on
the de-tiled plane.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pack_loss(sum_field, total_sumsq, n_bf: int, n_pix: int = 512 * 512) -> float:
    mean_phase = np.asarray(sum_field, dtype=np.float64) / float(n_bf)
    return float(np.asarray(total_sumsq, dtype=np.float64)) / (n_bf * n_pix) - float(
        np.mean(mean_phase * mean_phase)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--label", default="2pass-price")
    parser.add_argument("--gate-rtol", type=float, default=1e-5)
    args = parser.parse_args()

    d = np.load(args.npz)
    n_bf = int(d["mlx_sum"].shape[0] and 32)
    variants = {
        "engine_k32_in_path": (d["eng_sum"], d["eng_sumsq"]),
        "engine_k8_regroup": (d["eng_sum_k8"], d["eng_sumsq_k8"]),
        "mlx_ifft_complex64": (d["mlx_sum"], d["mlx_sumsq"]),
        "numpy_ifft_float32": (d["np32_sum"], d["np32_sumsq"]),
        "numpy_float64_reference": (d["ref_sum"], d["ref_sumsq"]),
    }
    base_s = np.asarray(d["eng_sum"], dtype=np.float64)
    base_l = [pack_loss(d["eng_sum"][i], d["eng_sumsq"][i], n_bf) for i in range(2)]
    rows = {}
    for name, (s, sq) in variants.items():
        s = np.asarray(s, dtype=np.float64)
        diff = s - base_s
        losses = [pack_loss(s[i], sq[i], n_bf) for i in range(2)]
        rows[name] = {
            "sum_field_max_abs_vs_engine": float(np.max(np.abs(diff))),
            "sum_field_rel_l2_vs_engine": float(
                np.linalg.norm(diff) / np.linalg.norm(base_s)
            ),
            "pack_loss": [round(v, 12) for v in losses],
            "pack_loss_max_rel_vs_engine": float(
                max(abs(losses[i] - base_l[i]) / abs(base_l[i]) for i in range(2))
            ),
        }

    orders = {
        "order_sequential": d["order_c"],
        "order_blocked8": d["order_b"],
        "order_numpy_axis": d["order_a"],
    }
    ref = np.asarray(d["ref_sum"], dtype=np.float64)
    spreads = {
        name: float(np.max(np.abs(np.asarray(arr, dtype=np.float64) - ref)))
        for name, arr in orders.items()
    }
    names = list(orders)
    pairwise = [
        float(
            np.max(
                np.abs(
                    np.asarray(orders[names[i]], dtype=np.float64)
                    - np.asarray(orders[names[j]], dtype=np.float64)
                )
            )
        )
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]

    record = {
        "label": args.label,
        "npz": str(args.npz),
        "bf_terms_in_pack": n_bf,
        "gate_rtol": args.gate_rtol,
        "pinned_loss_full_fit": 0.13769753277301788,
        "pack_loss_note": (
            "pack loss over the 32 BF terms of this pack only, computed with the "
            "engine's own shape sumsq/(nf*512*512) - mean((sum/nf)^2) for every "
            "variant, so the comparison is like-for-like."
        ),
        "variants": rows,
        "float32_ordering_floor": {
            "max_abs_spread_vs_float64_same_float32_phases": spreads,
            "max_abs_order_to_order": max(pairwise),
            "note": "identical float32 phase values, only the summation order changes",
        },
        "verdict": {
            "engine_float32_vs_float64": rows["numpy_float64_reference"][
                "pack_loss_max_rel_vs_engine"
            ],
            "in_path_regroup": rows["engine_k8_regroup"]["pack_loss_max_rel_vs_engine"],
            "different_decomposition_mlx": rows["mlx_ifft_complex64"][
                "pack_loss_max_rel_vs_engine"
            ],
            "different_decomposition_numpy": rows["numpy_ifft_float32"][
                "pack_loss_max_rel_vs_engine"
            ],
            "gate_rtol": args.gate_rtol,
        },
        "exceeds_gate": bool(
            rows["mlx_ifft_complex64"]["pack_loss_max_rel_vs_engine"] > args.gate_rtol
        ),
    }
    line = json.dumps(record, sort_keys=True)
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
