#!/usr/bin/env python3
"""Benchmark exact compact-HDF5 residency and interaction on real CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import cupy as cp
import numpy as np

from quantem.gpu.io.backends.cuda.compact_h5 import load_compact_h5_cuda


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _mask(
    inner_inclusive: float,
    outer_inclusive: float,
    *,
    center_row: float = 95.5,
    center_column: float = 95.5,
    detector_shape: tuple[int, int] = (192, 192),
) -> np.ndarray:
    rows, columns = np.mgrid[: detector_shape[0], : detector_shape[1]]
    radius_squared = (rows - center_row) ** 2 + (columns - center_column) ** 2
    result = radius_squared <= outer_inclusive**2
    if inner_inclusive > 0:
        result &= radius_squared >= inner_inclusive**2
    return result.astype(np.uint8)


def _read_oracle(oracle: Path, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if oracle.suffix == ".npy":
        expected = np.load(oracle, allow_pickle=False)
    else:
        expected = np.fromfile(oracle, dtype=dtype).reshape(shape)
    if expected.shape != shape:
        raise ValueError(
            f"Oracle {oracle.name} has shape {expected.shape}, expected {shape}."
        )
    return expected


def _parity(values: np.ndarray, oracle: Path, dtype: str) -> dict[str, object]:
    expected = _read_oracle(oracle, dtype, values.shape)
    observed = values.astype(np.uint64, copy=False)
    expected_u64 = expected.astype(np.uint64, copy=False)
    difference = np.where(
        observed >= expected_u64,
        observed - expected_u64,
        expected_u64 - observed,
    )
    return {
        "oracle": oracle.name,
        "samples": int(values.size),
        "mismatch_count": int(np.count_nonzero(difference)),
        "maximum_absolute_difference": int(difference.max(initial=0)),
    }


def _oracle_manifest(oracle_dir: Path) -> dict[str, object] | None:
    path = oracle_dir / "exact-product-manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _oracle_artifact(
    oracle_dir: Path,
    manifest: dict[str, object],
    key: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or key not in artifacts:
        raise ValueError(f"Oracle manifest has no {key!r} artifact.")
    record = artifacts[key]
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise TypeError(f"Oracle manifest {key!r} record is invalid.")
    path = oracle_dir / record["path"]
    if _sha256_file(path) != record.get("file_sha256"):
        raise ValueError(f"Oracle artifact {path.name} failed its file SHA-256.")
    return path


def _timing_summary(values: list[float]) -> dict[str, object]:
    samples = np.asarray(values, dtype=np.float64)
    reduction_target_ms = 1_000.0 / 120.0
    return {
        "samples": int(samples.size),
        "minimum_ms": float(samples.min()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "maximum_ms": float(samples.max()),
        "exact_reduction_target_ms": reduction_target_ms,
        "p95_meets_exact_reduction_target": bool(
            np.percentile(samples, 95) <= reduction_target_ms
        ),
        "presentation_rate_claimed": False,
    }


def _exact_products(source, oracle_dir: Path) -> dict[str, object]:
    products: dict[str, object] = {}
    manifest = _oracle_manifest(oracle_dir)
    if manifest is not None:
        if manifest.get("source_identity_sha256") != (
            source.metadata.source_identity_sha256
        ):
            raise ValueError("Oracle manifest belongs to a different source.")
        if manifest.get("working_dtype") != source.metadata.manifest.get(
            "working_dtype"
        ):
            raise ValueError("Oracle and resident working dtypes disagree.")
        for name in ("bf", "abf", "adf", "custom"):
            mask_path = _oracle_artifact(oracle_dir, manifest, f"{name}_mask_u8")
            mask = np.load(mask_path, allow_pickle=False)
            metrics = source.update_virtual_detector(mask)
            products[name] = {
                "metrics": asdict(metrics),
                "mask_sha256": manifest["artifacts"][f"{name}_mask_u8"][
                    "logical_sha256"
                ],
                "parity": _parity(
                    source.virtual_detector_values(),
                    _oracle_artifact(
                        oracle_dir,
                        manifest,
                        f"{name}_plane_u32",
                    ),
                    "<u4",
                ),
            }
        return products
    for name, inner, outer in (
        ("bf", 0.0, 48.0),
        ("abf", 24.0, 48.0),
        ("adf", 48.0, 95.5),
    ):
        metrics = source.update_virtual_detector(_mask(inner, outer))
        products[name] = {
            "metrics": asdict(metrics),
            "parity": _parity(
                source.virtual_detector_values(),
                oracle_dir / f"{name}_u64_le.bin",
                "<u8",
            ),
        }
    return products


def _selected_oracle(oracle_dir: Path) -> tuple[Path, str]:
    manifest = _oracle_manifest(oracle_dir)
    if manifest is None:
        return oracle_dir / "selected_center_masked_u8_le.bin", "u1"
    return (
        _oracle_artifact(oracle_dir, manifest, "selected_dp_working_u16"),
        "<u2",
    )


def _interaction_geometry(source) -> tuple[float, float, float, tuple[int, int]]:
    calibration = source.metadata.detector_calibration
    if calibration is None:
        return 95.5, 95.5, 48.0, tuple(source.metadata.shape[2:])
    center = calibration["detector_center_px"]
    return (
        float(center[0]),
        float(center[1]),
        float(calibration["bright_field_radius_px"]),
        tuple(source.metadata.shape[2:]),
    )


def _activation(source, oracle_dir: Path, name: str = "bf") -> dict[str, object]:
    manifest = _oracle_manifest(oracle_dir)
    if manifest is None:
        center_row, center_column, radius, detector_shape = _interaction_geometry(
            source
        )
        mask = _mask(
            0.0,
            radius,
            center_row=center_row,
            center_column=center_column,
            detector_shape=detector_shape,
        )
        oracle = oracle_dir / f"{name}_u64_le.bin"
        dtype = "<u8"
    else:
        mask = np.load(
            _oracle_artifact(oracle_dir, manifest, f"{name}_mask_u8"),
            allow_pickle=False,
        )
        oracle = _oracle_artifact(oracle_dir, manifest, f"{name}_plane_u32")
        dtype = "<u4"
    started = time.perf_counter()
    metrics = source.update_virtual_detector(mask)
    values = source.virtual_detector_values()
    request_to_host_result_ms = (time.perf_counter() - started) * 1_000.0
    return {
        "product": name,
        "request_to_host_result_ms": request_to_host_result_ms,
        "detector_update": asdict(metrics),
        "parity": _parity(values, oracle, dtype),
        "display_present_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--expected-compact-sha256")
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--next-compact", type=Path)
    parser.add_argument("--next-expected-compact-sha256")
    parser.add_argument("--next-oracle-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-samples", type=int, default=120)
    args = parser.parse_args()

    device_id = cp.cuda.Device().id
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")

    source = load_compact_h5_cuda(
        args.compact,
        expected_whole_file_sha256=args.expected_compact_sha256,
    )
    selected = source.extract_diffraction(256, 256)
    products = _exact_products(source, args.oracle_dir)

    center_row, center_column, radius, detector_shape = _interaction_geometry(source)
    interaction: dict[str, object] = {}
    for name, inner, outer in (
        ("bf_translate", 0.0, radius),
        ("adf_translate", radius, 2.0 * radius),
    ):
        source.update_virtual_detector(
            _mask(
                inner,
                outer,
                center_row=center_row,
                center_column=center_column,
                detector_shape=detector_shape,
            )
        )
        samples: list[float] = []
        changed: list[int] = []
        for index in range(args.trajectory_samples):
            phase = index % 4
            row = center_row + (1.0 if phase == 1 else -1.0 if phase == 3 else 0.0)
            column = center_column + (
                1.0 if phase == 0 else -1.0 if phase == 2 else 0.0
            )
            metrics = source.update_virtual_detector(
                _mask(
                    inner,
                    outer,
                    center_row=row,
                    center_column=column,
                    detector_shape=detector_shape,
                )
            )
            samples.append(metrics.wall_ms)
            changed.append(metrics.changed_detector_pixels)
        interaction[name] = {
            **_timing_summary(samples),
            "minimum_changed_pixels": min(changed),
            "maximum_changed_pixels": max(changed),
        }

    resident_pool_bytes = source.memory_pool_used_bytes
    result = {
        "schema": "quantem.gpu.compact-h5-cuda-benchmark/v3",
        "cold_cache_controlled": False,
        "timing_definitions": {
            "resident_ready": "request start through complete authenticated CUDA residency",
            "detector_wall": "detector request through synchronized exact CUDA result",
            "request_to_host_result": "detector request through complete host readback",
            "presentation": "not measured; transport, display upload, and first present excluded",
        },
        "device": {"id": device_id, "name": device_name},
        "source": {
            "path_name": args.compact.name,
            "file_bytes": source.metadata.file_bytes,
            "shape": list(source.metadata.shape),
            "source_identity_sha256": source.metadata.source_identity_sha256,
            "qgix_schema_version": source.metadata.schema_version,
            "resident_bytes": source.metadata.resident_bytes,
        },
        "load": asdict(source.load_metrics),
        "selected_diffraction": _parity(selected, *_selected_oracle(args.oracle_dir)),
        "products": products,
        "interaction": interaction,
        "resident_memory_pool_used_bytes": resident_pool_bytes,
        "fft_dispatch_count": 0,
    }
    if args.next_compact:
        if not args.next_oracle_dir:
            raise SystemExit("--next-compact requires --next-oracle-dir")
        next_load_started = time.perf_counter()
        next_source = load_compact_h5_cuda(
            args.next_compact,
            expected_whole_file_sha256=args.next_expected_compact_sha256,
        )
        next_ready_ms = (time.perf_counter() - next_load_started) * 1_000.0
        next_selected = next_source.extract_diffraction(256, 256)
        next_products = _exact_products(next_source, args.next_oracle_dir)
        result["different_file_switch"] = {
            "mode": "multi_resident_a_and_b",
            "a_remained_resident": True,
            "b_resident_ready_ms": next_ready_ms,
            "source": {
                "path_name": args.next_compact.name,
                "file_bytes": next_source.metadata.file_bytes,
                "shape": list(next_source.metadata.shape),
                "source_identity_sha256": next_source.metadata.source_identity_sha256,
                "qgix_schema_version": next_source.metadata.schema_version,
                "resident_bytes": next_source.metadata.resident_bytes,
            },
            "load": asdict(next_source.load_metrics),
            "selected_diffraction": _parity(
                next_selected,
                *_selected_oracle(args.next_oracle_dir),
            ),
            "products": next_products,
            "resident_memory_pool_used_bytes": next_source.memory_pool_used_bytes,
        }
        a_b_a = [
            {"source": "A", **_activation(source, args.oracle_dir)},
            {"source": "B", **_activation(next_source, args.next_oracle_dir)},
            {"source": "A", **_activation(source, args.oracle_dir)},
        ]
        result["a_b_a"] = {
            "definition": (
                "request start through exact host result with A and B already "
                "fully resident; transport, display upload, and present excluded"
            ),
            "samples": a_b_a,
            "all_under_one_second": all(
                sample["request_to_host_result_ms"] <= 1_000.0 for sample in a_b_a
            ),
            "presentation_rate_claimed": False,
            "combined_private_pool_used_bytes": (
                source.memory_pool_used_bytes + next_source.memory_pool_used_bytes
            ),
        }
        next_source.release_resident_storage()
        result["different_file_switch"]["released_memory_pool_used_bytes"] = (
            next_source.memory_pool_used_bytes
        )
    source.release_resident_storage()
    result["released_memory_pool_used_bytes"] = source.memory_pool_used_bytes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    parity = [result["selected_diffraction"]]
    parity.extend(product["parity"] for product in products.values())
    if "different_file_switch" in result:
        switched = result["different_file_switch"]
        parity.append(switched["selected_diffraction"])
        parity.extend(product["parity"] for product in switched["products"].values())
        parity.extend(sample["parity"] for sample in result["a_b_a"]["samples"])
    if any(item["mismatch_count"] for item in parity):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
