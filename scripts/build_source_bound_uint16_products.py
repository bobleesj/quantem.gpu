#!/usr/bin/env python3
"""Build exact uint16 source contracts and CPU product oracles for one tilt."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  Registers the audited bitshuffle filter.
import numpy as np

MASK_PATH = "/entry/instrument/detector/detectorSpecific/pixel_mask"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _save_array(root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = root / f"{name}.npy"
    logical = np.ascontiguousarray(value)
    np.save(path, logical, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(logical.shape),
        "dtype": logical.dtype.str,
        "logical_sha256": hashlib.sha256(logical.tobytes(order="C")).hexdigest(),
        "file_bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
    }


def _load_inputs(
    audit_path: Path,
    inventory_path: Path,
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = json.loads(audit_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    datasets = inventory.get("datasets")
    if not isinstance(datasets, list) or not 0 <= ordinal < len(datasets):
        raise ValueError("ordinal is outside the frozen inventory")
    frozen = datasets[ordinal]
    if audit["master"] != {
        "path": frozen["master"]["path"],
        "file_bytes": frozen["master"]["bytes"],
        "sha256": frozen["master"]["sha256"],
    }:
        raise ValueError("master path, size, or hash disagrees with frozen inventory")
    if [row["sha256"] for row in audit["source_files"]] != [
        row["sha256"] for row in frozen["members"]
    ]:
        raise ValueError("ordered member hashes disagree with frozen inventory")
    if audit["source_shape"] != frozen["logical_shape"]:
        raise ValueError("shape disagrees with frozen inventory")
    if audit["source_dtype"] != frozen["source_dtype"] != "uint16":
        raise ValueError("source dtype is not the frozen uint16 contract")
    if audit["range_audit"].get("complete") is not True:
        raise ValueError("source audit does not cover every logical sample")
    for record in audit["source_files"]:
        source = Path(record["path"])
        if source.stat().st_size != int(record["file_bytes"]):
            raise ValueError(f"{source.name} changed size after the source audit")
    return audit, frozen


def _mask_geometry(
    mean_dp: np.ndarray,
    excluded: np.ndarray,
) -> tuple[dict[str, float | int | list[float] | str], dict[str, np.ndarray]]:
    probe = mean_dp.astype(np.float32)
    threshold = float(probe.mean()) + float(probe.std())
    probe_mask = (probe > threshold) & ~excluded
    area = int(probe_mask.sum())
    if area == 0:
        raise ValueError("source-bound probe detection selected no pixels")
    rows = np.arange(probe.shape[0], dtype=np.float32)[:, None]
    columns = np.arange(probe.shape[1], dtype=np.float32)[None, :]
    center_row = float((rows * probe_mask).sum() / area)
    center_column = float((columns * probe_mask).sum() / area)
    radius = float(np.sqrt(area / np.pi))

    def radial(
        center: tuple[float, float],
        inner: float,
        outer: float,
    ) -> np.ndarray:
        distance_squared = (rows - np.float32(center[0])) ** 2 + (
            columns - np.float32(center[1])
        ) ** 2
        return (
            (distance_squared >= np.float32(inner) ** 2)
            & (distance_squared <= np.float32(outer) ** 2)
            & ~excluded
        )

    masks = {
        "bf": radial((center_row, center_column), 0.0, radius),
        "abf": radial((center_row, center_column), 0.5 * radius, radius),
        "adf": radial((center_row, center_column), radius, 2.0 * radius),
        "custom": radial(
            (center_row + 0.75, center_column - 0.75),
            0.25 * radius,
            1.25 * radius,
        ),
    }
    calibration: dict[str, float | int | list[float] | str] = {
        "detector_center_px": [center_row, center_column],
        "bright_field_radius_px": radius,
        "method": (
            "mean-plus-standard-deviation binary disk centroid and "
            "area-equivalent radius on source-bound mean diffraction"
        ),
        "threshold": threshold,
        "selected_disk_pixels": area,
    }
    return calibration, masks


def build_source_bound_uint16_products(
    audit_path: Path,
    inventory_path: Path,
    ordinal: int,
    output: Path,
    *,
    batch_frames: int = 256,
) -> dict[str, Any]:
    """Build one exact source contract and independent CPU product set."""
    started = time.perf_counter()
    audit_path = audit_path.resolve()
    inventory_path = inventory_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output {output}")
    output.mkdir(parents=True)
    audit, frozen = _load_inputs(audit_path, inventory_path, ordinal)
    shape = tuple(int(value) for value in audit["source_shape"])
    scan_count = shape[0] * shape[1]

    master_path = Path(audit["master"]["path"])
    with h5py.File(master_path, "r") as master:
        detector_mask = np.asarray(master[MASK_PATH][...], dtype="<u4")
    detector_mask_sha256 = hashlib.sha256(detector_mask.tobytes(order="C")).hexdigest()
    if detector_mask_sha256 != frozen["detector_mask"]["sha256"]:
        raise ValueError("detector mask disagrees with frozen inventory")
    excluded = detector_mask != 0
    excluded_rows, excluded_columns = np.nonzero(excluded)
    masked_detector_pixels = [
        [int(row), int(column)]
        for row, column in zip(excluded_rows, excluded_columns, strict=True)
    ]

    working_digest = hashlib.sha256()
    detector_total = np.zeros(shape[2:], dtype=np.uint64)
    selected_flat = (shape[0] // 2) * shape[1] + shape[1] // 2
    selected_raw: np.ndarray | None = None
    selected_working: np.ndarray | None = None
    admitted_maximum = 0
    admitted_values_above_255 = 0
    global_frame = 0
    for record in audit["source_files"]:
        with h5py.File(record["path"], "r") as handle:
            dataset = handle[record["dataset_path"]]
            for begin in range(0, dataset.shape[0], batch_frames):
                stop = min(begin + batch_frames, dataset.shape[0])
                raw = np.ascontiguousarray(dataset[begin:stop], dtype="<u2")
                working = raw.copy()
                working[:, excluded] = 0
                admitted_maximum = max(admitted_maximum, int(working.max()))
                admitted_values_above_255 += int(np.count_nonzero(working > 255))
                working_digest.update(working.tobytes(order="C"))
                detector_total += working.sum(axis=0, dtype=np.uint64)
                if global_frame <= selected_flat < global_frame + len(raw):
                    local = selected_flat - global_frame
                    selected_raw = raw[local].copy()
                    selected_working = working[local].copy()
                global_frame += len(raw)
    if global_frame != scan_count or selected_raw is None or selected_working is None:
        raise ValueError("first source pass did not cover every scan position")
    if admitted_maximum > np.iinfo(np.uint16).max:
        raise ValueError("an admitted value exceeds exact uint16")

    mean_dp = detector_total.astype(np.float64) / float(scan_count)
    calibration_values, masks = _mask_geometry(mean_dp, excluded)
    planes = {name: np.zeros(scan_count, dtype=np.uint32) for name in masks}
    total_per_scan = np.zeros(scan_count, dtype=np.uint32)
    row_moment = np.zeros(scan_count, dtype=np.uint64)
    column_moment = np.zeros(scan_count, dtype=np.uint64)
    row_weights = np.broadcast_to(
        np.arange(shape[2], dtype=np.uint64)[:, None], shape[2:]
    ).ravel()
    column_weights = np.broadcast_to(
        np.arange(shape[3], dtype=np.uint64)[None, :], shape[2:]
    ).ravel()
    flat_masks = {name: mask.ravel() for name, mask in masks.items()}
    global_frame = 0
    for record in audit["source_files"]:
        with h5py.File(record["path"], "r") as handle:
            dataset = handle[record["dataset_path"]]
            for begin in range(0, dataset.shape[0], batch_frames):
                stop = min(begin + batch_frames, dataset.shape[0])
                working = np.ascontiguousarray(dataset[begin:stop], dtype="<u2")
                working[:, excluded] = 0
                count = len(working)
                flat = working.reshape(count, -1).astype(np.uint64, copy=False)
                destination = slice(global_frame, global_frame + count)
                totals = flat.sum(axis=1, dtype=np.uint64)
                if int(totals.max(initial=0)) > np.iinfo(np.uint32).max:
                    raise OverflowError("full-detector sum exceeds exact uint32")
                total_per_scan[destination] = totals
                row_moment[destination] = (flat * row_weights).sum(axis=1)
                column_moment[destination] = (flat * column_weights).sum(axis=1)
                for name, mask in flat_masks.items():
                    values = flat[:, mask].sum(axis=1, dtype=np.uint64)
                    if int(values.max(initial=0)) > np.iinfo(np.uint32).max:
                        raise OverflowError(f"{name} detector sum exceeds exact uint32")
                    planes[name][destination] = values
                global_frame += count
    if global_frame != scan_count:
        raise ValueError("product pass did not cover every scan position")

    source_identity = frozen["source_identity_sha256"]
    calibration = {
        "schema": "quantem.gpu.detector-calibration/v1",
        "source_identity_sha256": source_identity,
        "detector_center_px": calibration_values["detector_center_px"],
        "bright_field_radius_px": calibration_values["bright_field_radius_px"],
        "dpc_rotation_degrees": 0.0,
        "dpc_component_order_exchanged": False,
        "method": calibration_values["method"],
    }
    source_contract = {
        "schema": "quantem.gpu.compact-uint16-source-contract/v1",
        "source_identity_sha256": source_identity,
        "source_raw_logical_sha256": audit["range_audit"]["logical_source_sha256"],
        "source_shape": list(shape),
        "source_dtype": "uint16",
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "detector_mask_sha256": detector_mask_sha256,
        "masked_detector_pixels": masked_detector_pixels,
        "detector_calibration": calibration,
    }
    source_contract_path = output / "source-contract.json"
    source_contract_path.write_text(
        json.dumps(source_contract, indent=2, sort_keys=True) + "\n"
    )

    artifacts = {
        "selected_dp_raw_u16": _save_array(output, "selected-dp-raw-u16", selected_raw),
        "selected_dp_working_u16": _save_array(
            output, "selected-dp-working-u16", selected_working
        ),
        "detector_total_u64": _save_array(output, "detector-total-u64", detector_total),
        "mean_dp_f64": _save_array(output, "mean-dp-f64", mean_dp),
        "total_per_scan_u32": _save_array(
            output, "total-per-scan-u32", total_per_scan.reshape(shape[:2])
        ),
        "row_moment_u64": _save_array(
            output, "row-moment-u64", row_moment.reshape(shape[:2])
        ),
        "column_moment_u64": _save_array(
            output, "column-moment-u64", column_moment.reshape(shape[:2])
        ),
    }
    for name, mask in masks.items():
        artifacts[f"{name}_mask_u8"] = _save_array(
            output, f"{name}-mask-u8", mask.astype(np.uint8)
        )
        artifacts[f"{name}_plane_u32"] = _save_array(
            output, f"{name}-plane-u32", planes[name].reshape(shape[:2])
        )

    manifest = {
        "schema": "quantem.gpu.seven-tilt-source-product-uint16/v1",
        "status": "complete",
        "ordinal": ordinal,
        "source_identity_sha256": source_identity,
        "ordered_hdf5_source_identity_sha256": audit["source_identity_sha256"],
        "source_shape": list(shape),
        "source_dtype": "uint16",
        "working_dtype": "uint16",
        "raw_logical_sha256": source_contract["source_raw_logical_sha256"],
        "working_uint16_sha256": working_digest.hexdigest(),
        "working_definition": (
            "all source counts exact; authenticated detector-mask pixels "
            "excluded only from scientific products"
        ),
        "admitted_observed_maximum": admitted_maximum,
        "admitted_values_above_255": admitted_values_above_255,
        "detector_mask_sha256": detector_mask_sha256,
        "masked_detector_pixels": masked_detector_pixels,
        "calibration": {
            **calibration,
            "threshold": calibration_values["threshold"],
            "selected_disk_pixels": calibration_values["selected_disk_pixels"],
        },
        "product_definition": {
            "accumulator": "uint64, with checked exact uint32 detector-plane output",
            "radial_boundary": "float32 d2 >= inner^2 and d2 <= outer^2",
            "selected_scan": [shape[0] // 2, shape[1] // 2],
            "masked_pixels_excluded": True,
        },
        "artifacts": artifacts,
        "source_contract": {
            "path": source_contract_path.name,
            "sha256": _sha256_file(source_contract_path),
        },
        "audit_path": str(audit_path),
        "audit_sha256": _sha256_file(audit_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256_file(inventory_path),
        "producer_sha256": _sha256_file(Path(__file__)),
        "batch_frames": batch_frames,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_executed": False,
    }
    manifest_path = output / "exact-product-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_contract": str(source_contract_path),
        "source_contract_sha256": _sha256_file(source_contract_path),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--ordinal", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-frames", type=int, default=256)
    arguments = parser.parse_args()
    receipt = build_source_bound_uint16_products(
        arguments.audit,
        arguments.inventory,
        arguments.ordinal,
        arguments.output_dir,
        batch_frames=arguments.batch_frames,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
