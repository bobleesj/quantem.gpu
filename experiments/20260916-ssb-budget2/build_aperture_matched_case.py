#!/usr/bin/env python3
"""Build the aperture-matched 512x512 ARINA case for the trial-budget test.

Why this case exists
--------------------
``strict-parity/arina-512-full-disk`` carries the *historical* documented
detector sampling (1.0909090909090908 mrad) with the documented disk
(centre 94.88451385498047 / 96.35952758789062, radius 53.35992814757164 px,
8937 selected pixels).  With that sampling only 2464 of the 8937 selected
pixels have a non-zero aperture weight, so the phase-variance loss compacts to
2464 terms and one objective evaluation costs ~78 ms.

The production app does not run that selection: after calibration the engine
matches the aperture to the bright-field disk, which sets the detector sampling
to 0.5622196476170719 mrad so the disk edge lands on the aperture edge and all
8937 terms are active.  That is the selection behind the 8,937-active-BF fit
(``metal-runs/final-fit/report.json``, ~283 ms per objective evaluation).

This script builds that second selection *without touching the shared case
module*: it copies the owner case's verified ``bf_columns.u16`` by symlink
(read-only) and regenerates the geometry with the case module's own
``_declaration`` / ``shared_geometry`` / ``metal_geometry`` functions at the
matched detector sampling.  Only the angular geometry changes; the raw uint16
counts, the selected pixel set, the scan region and the rotation are identical
by construction.

Writes only under ``budget2-runs/``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "parity"))

import ssb_parity_case as case_module  # noqa: E402

OWNER_RUNS = Path("/path/to/local/perf-lab/ssb-audit/parity-runs")
MY_RUNS = Path("/path/to/local/perf-lab/ssb-audit/budget2-runs")
OWNER_NAME = "arina-512-full-disk"
CASE_NAME = "arina-512-aperture-matched-8937"
#: The engine's aperture-matched detector sampling for the documented disk
#: (asserted verbatim by tests/metal/ssb_workflow_check.swift as the value
#: ``matchApertureToBrightfieldDisk()`` produces).
MATCHED_DET_SAMPLING_MRAD = 0.5622196476170719


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    case_module.runs_root = lambda: MY_RUNS  # type: ignore[assignment]
    owner = case_module.CASES[OWNER_NAME]
    owner_root = OWNER_RUNS / "strict-parity" / OWNER_NAME
    owner_payload = owner_root / "source" / "bf_columns.u16"
    if not owner_payload.is_file():
        raise SystemExit(f"owner payload missing: {owner_payload}")
    owner_declaration = json.loads((owner_root / "case.json").read_text(encoding="utf-8"))
    owner_calibration = json.loads(
        (owner_root / "source" / "snapshots" / "cal.json").read_text(encoding="utf-8")
    )

    case = dataclasses.replace(
        owner,
        name=CASE_NAME,
        description=(
            "Full 512x512 real ARINA acquisition, full documented BF disk, "
            "exact uint16 counts, *aperture-matched* detector sampling "
            "(0.5622196476170719 mrad) so that all 8937 selected BF pixels are "
            "aperture-active: the production selection behind the 8937-active "
            "fit. Bytes are the owner case's verified payload, unchanged."
        ),
        det_sampling_mrad=MATCHED_DET_SAMPLING_MRAD,
        artifact_name=None,
    )

    root = case_module.case_root(case)
    source = case_module._source_dir(case)
    (source / "snapshots").mkdir(parents=True, exist_ok=True)
    payload = source / "bf_columns.u16"
    if payload.exists() or payload.is_symlink():
        payload.unlink()
    payload.symlink_to(owner_payload)

    rows, _cols = case.bf_rows_cols
    side = case.scan_side
    counts = np.memmap(payload, dtype=np.uint16, mode="r", shape=(rows.size, side, side))
    column_sums = np.asarray(counts, dtype=np.int64).reshape(rows.size, -1).sum(axis=1)
    counts_max = int(counts.max())
    counts_sum = int(np.asarray(counts, dtype=np.int64).sum())
    del counts

    declaration = case_module._declaration(case)
    (source / "snapshots" / "cal.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema": "quantem-ssb-parity-case/1",
        "calibration": "snapshots/cal.json",
        "case": case.name,
        "description": case.description,
        "source": {
            "path": str(case.source),
            "scan_region": list(case.scan_region),
            "bf_columns": {
                "kind": "bf_columns",
                "order": "bf,scan",
                "path": "bf_columns.u16",
                "encoding": "u16",
                "dtype": "uint16",
                "shape": [int(rows.size), side * side],
                "scan_shape": [side, side],
                "detector_shape": [192, 192],
                "bytes": int(payload.stat().st_size),
                "bytes_per_bf": side * side * 2,
                "bits_per_value": 16,
                "max_value": counts_max,
                "detector_bin": 1,
                "symlinked_from": str(owner_payload),
            },
        },
    }
    (source / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    scientific = {
        "name": case.name,
        "description": case.description,
        "source": str(case.source),
        "scan_region": list(case.scan_region),
        "scan_shape": [side, side],
        "detector_shape": [192, 192],
        "bf_center": list(case.bf_center),
        "bf_radius_px": float(case.bf_radius),
        "bf_count": int(rows.size),
        "documented_bf_count": case_module.DOCUMENTED_BF_COUNT,
        "voltage_kV": case.voltage_kV,
        "semiangle_mrad": case.semiangle_mrad,
        "scan_sampling_A": case.scan_sampling_A,
        "rotation_angle_deg": case.rotation_angle_deg,
        "aberrations": [list(values) for values in case.aberrations],
        "counts_dtype": "uint16",
        "counts_max": counts_max,
        "counts_sum": counts_sum,
        "counts_column_sums_exact": [int(value) for value in column_sums],
        "brightfield_center": list(case.bf_center),
        "brightfield_radius_px": float(case.bf_radius),
        "brightfield_policy": "documented_full_disk_minus_hardware_dead_pixels",
    }
    scientific.update(case_module.shared_geometry(case, source))
    # The bytes are the owner's, and the owner already verified them against a
    # direct h5py read; carry that record over and prove byte identity by hash.
    scientific["counts_exactness"] = dict(owner_declaration.get("counts_exactness", {}))
    scientific["counts_provenance"] = {
        "payload_sha256": sha256_file(owner_payload),
        "payload_bytes": int(owner_payload.stat().st_size),
        "symlink": str(payload),
        "owner_case": OWNER_NAME,
        "owner_verified_against_hdf5": bool(
            dict(owner_declaration.get("counts_exactness", {})).get("exact", False)
        ),
    }
    scientific["metal_geometry"] = case_module.metal_geometry(scientific)
    scientific["metal_geometry_unrotated"] = {
        "brightfieldKX": scientific["brightfield_kx_unrotated"],
        "brightfieldKY": scientific["brightfield_ky_unrotated"],
        "brightfieldAlphaSquared": scientific["brightfield_alpha_squared_unrotated"],
        "brightfieldAperture": scientific["brightfield_aperture_unrotated"],
        "brightfieldCos2Phi": scientific["brightfield_cos2phi_unrotated"],
        "brightfieldSin2Phi": scientific["brightfield_sin2phi_unrotated"],
    }
    (root / "case.json").write_text(
        json.dumps(scientific, indent=2, sort_keys=True), encoding="utf-8"
    )

    aperture = np.asarray(scientific["brightfield_aperture"], dtype=np.float64)
    checks = {
        "bf_count_is_8937": scientific["bf_count"] == 8937,
        "active_bf_count_is_8937": scientific["active_bf_count"] == 8937,
        "all_aperture_weights_positive": bool(np.all(aperture > 0.0)),
        "counts_max_matches_owner": counts_max == int(owner_declaration["counts_max"]),
        "counts_sum_matches_owner": counts_sum == int(owner_declaration["counts_sum"]),
        "pixel_set_matches_owner": (
            declaration["bf_rows"] == owner_calibration["bf_rows"]
            and declaration["bf_cols"] == owner_calibration["bf_cols"]
        ),
        "det_sampling_is_matched_value": (
            abs(float(scientific["det_sampling_mrad"]) - MATCHED_DET_SAMPLING_MRAD) == 0.0
        ),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit("aperture-matched case failed its own construction checks")
    print(f"case: {root}")
    print(f"payload sha256: {scientific['counts_provenance']['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
