"""Measure the row-IFFT intermediate: size, traffic share, and any redundancy.

The 512 exact objective writes one (batch, stored BF, 512, 512) complex64
row-IFFT plane per stored BF term and reads it back in the column stage.  That
round trip is the dominant traffic in the fit, so the only way past the
measured streaming ceiling is to touch fewer bytes.  This probe therefore asks
whether the intermediate carries exploitable redundancy.

Layout matters: the batched/scalar 512 kernels store the plane tiled as
``flat = (col>>3)*4096 + (col&7) + row*8`` so the column stage can read eight
rows of one column out of a single 64-byte granule.  The probe de-tiles before
testing symmetry, and proves the de-tiling is correct by checking that the
de-tiled copy equals the flat reference bit-exactly, on real 8937-BF data.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
sys.path.insert(0, str(Path(__file__).parent))

from profile_mps import FIXTURE, bandwidth_control, open_fixture, prepared_of  # noqa: E402


def detile(flat: np.ndarray) -> np.ndarray:
    """Invert the 512-column tiled store into a logical (row, col) plane."""
    tiles = flat.reshape(64, 512, 8)
    return np.ascontiguousarray(tiles.transpose(1, 0, 2).reshape(512, 512))


def symmetry_report(plane: np.ndarray) -> dict:
    """Test the candidate redundancies of one de-tiled (512, 512) plane."""
    n = plane.shape[0]
    idx = np.arange(n)
    scale = float(np.max(np.abs(plane)))
    return {
        "max_abs": scale,
        "joint_mirror_max_abs_err": float(
            np.max(np.abs(plane - np.conj(plane[(n - idx) % n][:, (n - idx) % n])))
        ),
        "row_mirror_max_abs_err": float(
            np.max(np.abs(plane - np.conj(plane[:, (n - idx) % n])))
        ),
        "col_mirror_max_abs_err": float(
            np.max(np.abs(plane - np.conj(plane[(n - idx) % n, :])))
        ),
        "max_abs_imag_over_max_abs_real": float(
            np.max(np.abs(plane.imag)) / max(np.max(np.abs(plane.real)), 1e-30)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--terms", type=int, default=1)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--label", default="intermediate")
    args = ap.parse_args()

    import mlx.core as mx
    from quantem.gpu.ssb.backends.mps import engine as E

    record: dict = {
        "schema": "quantem.ssb.mps.hotpath.v1",
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label,
        "fixture": args.fixture,
        "precision": {"real": "float32", "complex": "complex64"},
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "engine_sha256_prefix": None,
    }
    import hashlib

    record["engine_sha256_prefix"] = hashlib.sha256(
        Path(E.__file__).read_bytes()
    ).hexdigest()[:16]
    record["engine_file"] = str(E.__file__)

    opened, open_seconds = open_fixture(args.fixture)
    backend, prepared, prepare_seconds = prepared_of(opened)
    record["open_seconds"] = open_seconds
    record["prepare_seconds"] = prepare_seconds
    record["scan_shape"] = list(prepared.scan_shape)
    record["logical_bf"] = int(prepared.num_bf)
    record["stored_bf"] = int(prepared.g_qk.shape[0])
    record["g_qk_bytes"] = int(prepared.g_qk.nbytes)

    c10 = mx.array([7.017120839737006], dtype=mx.float32)
    c12 = mx.array([0.0], dtype=mx.float32)
    sin_np = np.sin(2.0 * np.asarray([-0.15393969519675116]))
    cos_np = np.cos(2.0 * np.asarray([-0.15393969519675116]))
    cos2 = mx.array(cos_np.astype(np.float32))
    sin2 = mx.array(sin_np.astype(np.float32))

    common = dict(
        start=0,
        stop=int(args.terms),
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    )
    scalar_fn = E._row_ifft512_from_dynamic_geometry
    accepts_tiled = "tiled_output" in scalar_fn.__code__.co_varnames

    t0 = time.perf_counter()
    flat = scalar_fn(prepared, **common, tiled_output=False) if accepts_tiled else scalar_fn(prepared, **common)
    mx.eval(flat)
    record["scalar_flat_seconds"] = time.perf_counter() - t0
    flat_np = np.asarray(flat)

    if accepts_tiled:
        t0 = time.perf_counter()
        tiled = scalar_fn(prepared, **common, tiled_output=True)
        mx.eval(tiled)
        record["scalar_tiled_seconds"] = time.perf_counter() - t0
        tiled_np = np.asarray(tiled)
        detiled = np.stack([detile(tiled_np[i]) for i in range(tiled_np.shape[0])])
        record["tiled_vs_flat_bit_exact"] = bool(np.array_equal(detiled, flat_np))
        record["tiled_vs_flat_max_abs_err"] = float(np.max(np.abs(detiled - flat_np)))
        record["detile_layout"] = "flat = (col>>3)*4096 + (col&7) + row*8"

    import hashlib as _hashlib

    record["flat_plane_sha256"] = _hashlib.sha256(
        np.ascontiguousarray(flat_np).tobytes()
    ).hexdigest()
    record["engines_compared"] = "frozen-32ba29b" if not accepts_tiled else "head-f357069"

    plane = np.ascontiguousarray(flat_np[0])
    record["n_planes"] = int(flat_np.shape[0])
    record["plane_bytes"] = int(plane.nbytes)
    record["planes"] = [symmetry_report(plane)]

    plane_bytes = 512 * 512 * 8
    g_row_bytes = 512 * 257 * 8
    record["traffic_per_pair_pack"] = {
        "note": "512 BF terms, two candidates (batch 2)",
        "g_read_bytes": int(512 * g_row_bytes),
        "row_ifft_write_bytes": int(2 * 512 * plane_bytes),
        "row_ifft_read_bytes": int(2 * 512 * plane_bytes),
        "intermediate_share": float(2 * 512 * plane_bytes / (2 * 512 * plane_bytes + 512 * g_row_bytes)),
    }
    record["bandwidth_control_gbs"] = bandwidth_control(mx)
    line = json.dumps(record, sort_keys=True)
    print(line)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
