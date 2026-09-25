#!/usr/bin/env python3
"""Build the 512x512 Metal case for the *production* acquisition (-17.0x tilt).

The MPS fit evidence and the recorded 62 s production fit both run on
an industrial partner logic acquisition (512 x 512 scan, 17 mrad tilt series, the -17.0x position). The Metal trial-budget
harness needs a case directory (``case.json`` + ``source/bf_columns.u16``), so
this script builds one whose bytes are that acquisition's exact raw counts:

* the payload is the MPS fixture's export (``fixture-512-a``),
  which was written by the production loader from the same file, referenced by
  symlink so no copy exists;
* this script proves the pixel set is the documented disk, then verifies every
  stored value byte-exactly against a direct h5py read of the acquisition
  (``verify_counts_against_hdf5``: no quantem loader, no hot-pixel correction);
* the geometry is regenerated with the parity case module's own functions at
  the aperture-matched sampling (0.5622196476170719 mrad).

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

MY_RUNS = Path("/path/to/local/perf-lab/ssb-audit/budget2-runs")
FIXTURE = Path("/path/to/local/perf-lab/ssb-audit/mps-runs/fixture-512-a")
PRODUCTION_MASTER = Path(
    "/path/to/local/data/Live4DSTEM Testing/ARINA/"
    "arina-fixture-a_master.h5"
    ""
)
CASE_NAME = "arina-512-17p0x-production-8937"
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
    reference = case_module.CASES["arina-512-full-disk"]
    case = dataclasses.replace(
        reference,
        name=CASE_NAME,
        description=(
            "Full 512x512 real ARINA acquisition at the production tilt "
            "(-17.0x/0.0y), full documented BF disk, exact uint16 counts, "
            "aperture-matched detector sampling (0.5622196476170719 mrad): the "
            "acquisition, selection and cost basis of the recorded 8937-active "
            "production fit. Payload is the verified MPS fixture export."
        ),
        scan_region=(0, 512, 0, 512),
        source=PRODUCTION_MASTER,
        det_sampling_mrad=MATCHED_DET_SAMPLING_MRAD,
        artifact_name=None,
    )
    root = case_module.case_root(case)
    source = case_module._source_dir(case)
    (source / "snapshots").mkdir(parents=True, exist_ok=True)
    owner_payload = FIXTURE / "source" / "bf_columns.u16"
    if not owner_payload.is_file():
        raise SystemExit(f"fixture payload missing: {owner_payload}")
    payload = source / "bf_columns.u16"
    if payload.exists() or payload.is_symlink():
        payload.unlink()
    payload.symlink_to(owner_payload)

    rows, cols = case.bf_rows_cols
    side = case.scan_side
    counts = np.memmap(payload, dtype=np.uint16, mode="r", shape=(rows.size, side, side))
    if counts.shape[0] != rows.size or counts.shape[1:] != (side, side):
        raise SystemExit(f"payload shape {counts.shape} does not match the case")

    # 1. the pixel set must be exactly the documented disk minus dead pixels.
    disk_rows, disk_cols = case_module.brightfield_disk(
        case.detector_shape_hint, case.bf_center, case.bf_radius
    )
    dead = case_module.detector_dead_pixels(case.source)
    dead_index = set(zip(dead[:, 0].tolist(), dead[:, 1].tolist()))
    keep = np.fromiter(
        ((int(r), int(c)) not in dead_index for r, c in zip(disk_rows.tolist(), disk_cols.tolist())),
        dtype=bool,
        count=disk_rows.size,
    )
    expected_rows, expected_cols = disk_rows[keep], disk_cols[keep]
    pixel_set_ok = bool(
        np.array_equal(rows, expected_rows) and np.array_equal(cols, expected_cols)
    )

    # 2. every stored value against a direct h5py read of the acquisition.
    exactness = case_module.verify_counts_against_hdf5(case, counts, rows, cols)
    counts_max = int(counts.max())
    counts_sum = int(np.asarray(counts, dtype=np.int64).sum())
    del counts

    counts_memmap = np.memmap(payload, dtype=np.uint16, mode="r", shape=(rows.size, side, side))
    column_sums = np.asarray(counts_memmap, dtype=np.int64).reshape(rows.size, -1).sum(axis=1)
    del counts_memmap

    fixture_manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    fixture_max = int(fixture_manifest["source"]["bf_columns"]["max_value"])

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
    scientific["counts_exactness"] = exactness
    scientific["counts_provenance"] = {
        "payload_sha256": sha256_file(owner_payload),
        "payload_bytes": int(owner_payload.stat().st_size),
        "symlink": str(payload),
        "exported_by": "quantem.gpu MPS loader export_brightfield (mps fixture)",
        "fixture": str(FIXTURE),
        "fixture_max_value": fixture_max,
        "verified_against_hdf5_here": bool(exactness.get("exact", False)),
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
        "pixel_set_is_documented_disk": pixel_set_ok,
        "bf_count_is_8937": scientific["bf_count"] == 8937,
        "active_bf_count_is_8937": scientific["active_bf_count"] == 8937,
        "all_aperture_weights_positive": bool(np.all(aperture > 0.0)),
        "counts_max_matches_fixture": counts_max == fixture_max,
        "counts_exact_vs_hdf5": bool(exactness.get("exact", False)),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("exactness:", json.dumps(exactness, sort_keys=True)[:400])
    if not all(checks.values()):
        raise SystemExit("production case failed its own construction checks")
    print(f"case: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
